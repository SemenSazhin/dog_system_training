#!/usr/bin/env python3
"""Регистрация прогона skeleton_cv / обучения в D:\\dog-ml\\registry\\experiments.jsonl.

Пример:
  python register_experiment.py --results D:\\dog-ml\\cv\\results_retrain_20261101.json --model lstm \\
      --note "после выгрузки 01.11" --dataset-snapshot dataset_20261101_223000.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

REGISTRY = Path(r"D:\dog-ml\registry")
PRODUCTION = REGISTRY / "production.json"


def mean_std(runs: list[dict], key: str) -> dict | None:
    vals = [float(r[key]) for r in runs if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": m, "std": 0.0, "n": 1}
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return {"mean": m, "std": var ** 0.5, "n": len(vals)}


def summarize_model(runs: list[dict]) -> dict:
    keys = ("auc", "ap", "f1_best", "auc_other", "cam_fpr_05", "win_auc", "auc_k1", "auc_k5", "auc_kmean")
    out = {}
    for k in keys:
        s = mean_std(runs, k)
        if s:
            out[k] = s
    return out


def load_production() -> dict:
    if PRODUCTION.exists():
        return json.loads(PRODUCTION.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True, help="results_<tag>.json из skeleton_cv")
    ap.add_argument("--model", default="lstm", help="какую модель из results взять")
    ap.add_argument("--note", default="")
    ap.add_argument("--dataset-snapshot", default="", help="имя файла из registry/snapshots")
    ap.add_argument("--weights", type=Path, default=None, help="опционально: .pth/.pt для архива")
    ap.add_argument("--set-candidate", action="store_true", help="записать как candidate в production.json")
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"нет файла: {args.results}")

    REGISTRY.mkdir(parents=True, exist_ok=True)
    (REGISTRY / "models").mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    runs = payload.get("results", {}).get(args.model)
    if not runs:
        raise SystemExit(f"в {args.results.name} нет results['{args.model}']")

    metrics = summarize_model(runs)
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model}"
    row = {
        "id": exp_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "model": args.model,
        "results_file": str(args.results),
        "tag": payload.get("args", {}).get("tag") or args.results.stem.replace("results_", ""),
        "dataset_snapshot": args.dataset_snapshot,
        "n_videos": payload.get("n_videos"),
        "n_windows": payload.get("n_windows"),
        "cfg": payload.get("cfg"),
        "args": payload.get("args"),
        "metrics": metrics,
    }

    if args.weights and args.weights.exists():
        dest = REGISTRY / "models" / f"{exp_id}{args.weights.suffix}"
        shutil.copy2(args.weights, dest)
        row["weights"] = str(dest)

    with (REGISTRY / "experiments.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prod = load_production()
    baseline = (prod.get("active") or {}).get("metrics") or {}
    cmp = {}
    for k, v in metrics.items():
        b = (baseline.get(k) or {}).get("mean")
        if b is not None:
            cmp[k] = {
                "baseline": b,
                "new": v["mean"],
                "delta": round(v["mean"] - b, 4),
                "better": (v["mean"] > b) if k != "cam_fpr_05" else (v["mean"] < b),
            }
    row["vs_production"] = cmp

    # перезапишем последнюю строку сравнения удобным summary-файлом
    (REGISTRY / "last_experiment.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.set_candidate:
        prod["candidate"] = {
            "experiment_id": exp_id,
            "metrics": metrics,
            "weights": row.get("weights"),
            "results_file": str(args.results),
            "note": args.note,
            "ts": row["ts"],
        }
        PRODUCTION.write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"registered {exp_id}")
    for k, v in metrics.items():
        print(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f} (n={v['n']})")
    if cmp:
        print("vs production:")
        for k, v in cmp.items():
            mark = "+" if v["better"] else "-"
            print(f"  {mark} {k}: {v['baseline']:.3f} -> {v['new']:.3f} (Δ {v['delta']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
