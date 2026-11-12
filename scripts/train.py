#!/usr/bin/env python
"""Train one configuration with one seed and save predictions.

Examples
--------
python scripts/train.py --config configs/covid_robotxdpp.yaml --seed 42
python scripts/train.py --config configs/nih_to_chexpert_uda.yaml --seed 7 --set loss.beta=0.01
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from robotxd import analysis, engine, pca  # noqa: E402
from robotxd.builders import build_loaders, build_splits  # noqa: E402
from robotxd.models import build_model  # noqa: E402
from robotxd.utils import get_device, load_config, save_json, seed_everything  # noqa: E402


def parse_value(v: str):
    import yaml
    return yaml.safe_load(v)


def apply_overrides(cfg: dict, pairs) -> dict:
    cfg = copy.deepcopy(cfg)
    for p in pairs or []:
        key, val = p.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = parse_value(val)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="run name (default: config file name)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. loss.alignment=none")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    name = args.name or cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.out, name, f"seed{args.seed}")
    device = get_device(cfg.get("device", "auto"))
    seed_everything(args.seed)
    print(f"== {name} | seed {args.seed} | {device} | out={out_dir}")

    # ---------------- source data
    tr, va, te, label_names, class_names, multilabel = build_splits(cfg["data"])
    loaders = build_loaders(cfg["data"], (tr, va, te), label_names, class_names, multilabel)
    names = label_names if multilabel else class_names
    num_outputs = len(names)

    # ---------------- target data (transfer experiments only)
    target_loaders, target_names = None, None
    if cfg.get("target"):
        ttr, tva, tte, tln, tcn, tml = build_splits(cfg["target"])
        assert tml == multilabel, "source and target must both be single- or multi-label"
        target_names = tln if tml else tcn
        assert list(target_names) == list(names), (
            f"label spaces differ: {names} vs {target_names}. Use data.labels: shared on both sides.")
        target_loaders = build_loaders(cfg["target"], (ttr, tva, tte), tln, tcn, tml)

    uda = cfg.get("loss", {}).get("alignment", "none") not in ("none", "notebook_placeholder")
    if uda and target_loaders is None:
        raise SystemExit("alignment needs a `target:` section (unsupervised domain adaptation).")

    # ---------------- model
    model = build_model(cfg, num_outputs).to(device)
    meta = {"seed": args.seed, "names": list(names), "multilabel": multilabel, "protocol": "UDA" if uda else "source-only"}
    if getattr(model, "compression", None) == "pca":
        meta["pca"] = pca.fit_fixed_pca(model, loaders["train_eval"], device,
                                        max_batches=cfg.get("pca", {}).get("max_batches"))
        meta["pca"].pop("eigenvalues", None)

    class_weights = None
    if not multilabel and cfg.get("train", {}).get("class_weighted_loss", False):
        # the original notebook used BOTH a balanced sampler and a weighted loss; prefer one of the two
        counts = np.bincount(loaders["train"].dataset.targets, minlength=num_outputs)
        class_weights = 1.0 / counts

    model = engine.fit(model, cfg, loaders["train"], loaders["val"], device, out_dir, multilabel,
                       target_loader=target_loaders["train"] if uda else None, class_weights=class_weights,
                       label_names=names, extra_meta=meta)

    # ---------------- predictions (every reported number is computed from these files)
    engine.save_pred(engine.predict(model, loaders["val"], device, multilabel), os.path.join(out_dir, "pred_val.npz"))
    engine.save_pred(engine.predict(model, loaders["test"], device, multilabel), os.path.join(out_dir, "pred_test.npz"))
    te[["path", "patient"]].to_csv(os.path.join(out_dir, "test_index.csv"), index=False)
    if target_loaders is not None:
        engine.save_pred(engine.predict(model, target_loaders["test"], device, multilabel),
                         os.path.join(out_dir, "pred_target_test.npz"))
        tte[["path", "patient"]].to_csv(os.path.join(out_dir, "target_test_index.csv"), index=False)

    # ---------------- diagnostics
    diag = {}
    if getattr(model, "compression", None) == "pca":
        diag["pca_drift"] = pca.basis_drift_report(model, loaders["train_eval"], device,
                                                   max_batches=cfg.get("pca", {}).get("max_batches"))
    z = analysis.collect_latents(model, loaders["test"], device)
    diag["latent_test"] = analysis.latent_report(z)
    save_json(diag, os.path.join(out_dir, "diagnostics.json"))
    print("diagnostics:", {k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in diag.items()})
    open(os.path.join(out_dir, "DONE"), "w").close()


if __name__ == "__main__":
    main()
