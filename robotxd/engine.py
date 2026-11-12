"""Training loop (resume-safe) and prediction.

Every run writes to ``out_dir``:
    history.csv        one row per epoch (train loss terms, validation metrics, lr)
    best.pth           weights with the best validation selection metric
    last.pth           full state for resuming
    pred_val.npz       validation predictions of the best model
    pred_test.npz      test predictions of the best model   <- all reported numbers come from these
    run.json           config, seed, target-data access, PCA diagnostics
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd
import torch

from . import losses as L
from .metrics import compute_metrics
from .utils import save_json


@torch.no_grad()
def predict(model, loader, device, multilabel: bool) -> dict:
    model.eval()
    ys, ps, ms, ids = [], [], [], []
    for x, y, m, idx in loader:
        logits = model(x.to(device)).float()
        prob = torch.sigmoid(logits) if multilabel else torch.softmax(logits, 1)
        ys.append(y.numpy()), ps.append(prob.cpu().numpy()), ms.append(m.numpy()), ids.append(idx.numpy())
    return {"y_true": np.concatenate(ys), "y_prob": np.concatenate(ps),
            "mask": np.concatenate(ms), "index": np.concatenate(ids)}


def save_pred(pred: dict, path: str) -> None:
    np.savez_compressed(path, **pred)


def load_pred(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _reg_terms(cfg_loss: dict, latent, target_latent):
    terms = {}
    dec = cfg_loss.get("decorrelation", "none")
    if dec != "none":
        terms["decorrelation"] = cfg_loss.get("alpha", 0.01) * L.DECORRELATION[dec](latent)
    ali = cfg_loss.get("alignment", "none")
    if ali == "notebook_placeholder":
        terms["alignment"] = cfg_loss.get("beta", 0.005) * L.notebook_alignment_placeholder(latent)
    elif ali != "none":
        if target_latent is None:
            raise ValueError(f"alignment={ali!r} needs a target loader (UDA). Use alignment: none for source-only.")
        terms["alignment"] = cfg_loss.get("beta", 0.005) * L.ALIGNMENT[ali](latent, target_latent)
    g = cfg_loss.get("gamma", 0.0)
    if g:
        terms["latent_l2"] = g * L.latent_l2(latent)
    return terms


def fit(model, cfg: dict, train_loader, val_loader, device, out_dir: str, multilabel: bool,
        target_loader=None, class_weights=None, label_names=None, extra_meta: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    tcfg, lcfg = cfg.get("train", {}), cfg.get("loss", {})
    epochs = tcfg.get("epochs", 40)
    select = "macro_auroc" if multilabel else "macro_f1"

    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.get("lr", 1e-4), weight_decay=tcfg.get("weight_decay", 1e-5))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=tcfg.get("lr_factor", 0.5),
                                                       patience=tcfg.get("lr_patience", 5))
    cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    hist_path, best_path, last_path = (os.path.join(out_dir, f) for f in ("history.csv", "best.pth", "last.pth"))
    history, best, start = [], -np.inf, 0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        best, start = ck["best"], ck["epoch"]
        history = pd.read_csv(hist_path).to_dict("records") if os.path.exists(hist_path) else []
        print(f"Resuming from epoch {start}/{epochs} (best {select}={best:.4f})")

    tgt_iter = itertools.cycle(target_loader) if target_loader is not None else None
    for epoch in range(start, epochs):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for x, y, m, _ in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits, latent = model(x, return_latent=True)
                zt = None
                if tgt_iter is not None:
                    xt = next(tgt_iter)[0].to(device)  # labels of target batches are never read
                    _, zt = model(xt, return_latent=True)
                cls = L.classification_loss(logits.float(), y, m, multilabel, cw, lcfg.get("label_smoothing", 0.0))
                terms = _reg_terms(lcfg, latent.float(), None if zt is None else zt.float())
                loss = cls + sum(terms.values()) if terms else cls
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            for k, v in {"loss": loss, "cls": cls, **terms}.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1

        val = compute_metrics(predict(model, val_loader, device, multilabel), multilabel, label_names)
        sched.step(val[select])
        row = {"epoch": epoch + 1, **{f"train_{k}": v / n for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items() if "/" not in k},
               "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        if val[select] > best:
            best = val[select]
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "epoch": epoch + 1}, last_path)
        print(f"epoch {epoch + 1}/{epochs}  loss={row['train_loss']:.4f}  val_{select}={val[select]:.4f}  "
              f"best={best:.4f}  ({row['minutes']:.1f} min)")

    model.load_state_dict(torch.load(best_path, map_location=device))
    meta = {"selection_metric": select, "best_val": best, "epochs": epochs,
            "target_access": {
                "target_train_images": "unlabelled, alignment loss only" if target_loader is not None else "none",
                "target_val_images": "none", "target_test_images": "evaluation only",
                "batchnorm": "target batches pass through BN in train mode" if target_loader is not None else "source only",
                "model_selection": "source validation split", "thresholds": "source validation split"},
            **(extra_meta or {})}
    save_json({"config": cfg, **meta}, os.path.join(out_dir, "run.json"))
    return model
