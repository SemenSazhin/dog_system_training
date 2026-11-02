#!/usr/bin/env python3
"""Сетка LSTM-конфигов на актуальном raw + регистрация в реестре.

Запуск:
  D:\\dog-ml\\venv\\Scripts\\python.exe -u run_lstm_grid.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = r"D:\dog-ml\venv\Scripts\python.exe"
SCRIPTS = Path(__file__).resolve().parent
CV = SCRIPTS / "skeleton_cv.py"
REG = SCRIPTS / "register_experiment.py"
RAW = r"D:\dog-ml\raw"
OUT = Path(r"D:\dog-ml\cv")
SNAP = "dataset_20261102_013728.json"
OUT.mkdir(parents=True, exist_ok=True)

# (tag, extra_args) — фокус на том, что давало прирост в прошлых экспериментах
JOBS = [
    ("grid_t30_mv07_min", ["--T", "30", "--min-valid", "0.7", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t30_mv05_min", ["--T", "30", "--min-valid", "0.5", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t30_mv09_min", ["--T", "30", "--min-valid", "0.9", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t45_mv07_min", ["--T", "45", "--min-valid", "0.7", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t45_mv09_min", ["--T", "45", "--min-valid", "0.9", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t45_mv09_base", ["--T", "45", "--min-valid", "0.9", "--no-asp"]),  # как retrain_20261102
    ("grid_t60_mv07_min", ["--T", "60", "--min-valid", "0.7", "--no-asp", "--no-conf", "--no-vel", "--no-aug"]),
    ("grid_t30_mv07_noconf", ["--T", "30", "--min-valid", "0.7", "--no-asp", "--no-conf"]),
    ("grid_t30_mv07_novel", ["--T", "30", "--min-valid", "0.7", "--no-asp", "--no-vel"]),
    ("grid_t30_mv07_noaug", ["--T", "30", "--min-valid", "0.7", "--no-asp", "--no-aug"]),
]


def already_done(tag: str) -> bool:
    p = OUT / f"results_{tag}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return "lstm" in d.get("results", {})
    except Exception:
        return False


def run_one(tag: str, extra: list[str]) -> int:
    if already_done(tag):
        print(f"[skip] {tag}", flush=True)
        return 0
    cmd = [
        PY, "-u", str(CV),
        "--raw", RAW, "--out", str(OUT), "--tag", tag,
        "--models", "lstm", "--seeds", "5", *extra,
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCH_NUM_THREADS="2")
    log = OUT / f"log_{tag}.txt"
    print(f"[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    print(f"[{time.strftime('%H:%M:%S')}] END {tag} rc={rc} {time.time()-t0:.0f}s", flush=True)
    if rc != 0:
        return rc
    note = f"grid {tag} " + " ".join(extra)
    subprocess.run(
        [PY, str(REG), "--results", str(OUT / f"results_{tag}.json"), "--model", "lstm",
         "--dataset-snapshot", SNAP, "--note", note, "--set-candidate"],
        check=False,
    )
    return 0


def summarize() -> Path:
    rows = []
    for tag, _ in JOBS:
        p = OUT / f"results_{tag}.json"
        # плюс уже сделанные retrain_*
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        runs = d.get("results", {}).get("lstm") or []
        if not runs:
            continue
        def mean(k):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return sum(vs) / len(vs) if vs else None
        rows.append({
            "tag": tag,
            "auc": mean("auc"),
            "ap": mean("ap"),
            "f1_best": mean("f1_best"),
            "auc_other": mean("auc_other"),
            "cam_fpr_05": mean("cam_fpr_05"),
            "n_videos": d.get("n_videos"),
            "n_windows": d.get("n_windows"),
            "args": d.get("args"),
        })
    for tag in ("retrain_20261102", "retrain_bestcfg_20261102"):
        p = OUT / f"results_{tag}.json"
        if not p.exists():
            continue
        if any(r["tag"] == tag for r in rows):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        runs = d.get("results", {}).get("lstm") or []
        if not runs:
            continue
        def mean(k):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return sum(vs) / len(vs) if vs else None
        rows.append({
            "tag": tag,
            "auc": mean("auc"),
            "ap": mean("ap"),
            "f1_best": mean("f1_best"),
            "auc_other": mean("auc_other"),
            "cam_fpr_05": mean("cam_fpr_05"),
            "n_videos": d.get("n_videos"),
            "n_windows": d.get("n_windows"),
            "args": d.get("args"),
        })
    rows.sort(key=lambda r: (r["auc"] is not None, r["auc"] or 0), reverse=True)
    out = OUT / f"grid_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== TOP by AUC ===", flush=True)
    for r in rows[:8]:
        print(
            f"{r['tag']}: auc={r['auc']:.3f} ap={r['ap']:.3f} "
            f"auc_other={r['auc_other']:.3f} cam_fpr={r['cam_fpr_05']:.3f} "
            f"vids={r['n_videos']}",
            flush=True,
        )
    return out


def main() -> int:
    # если bestcfg ещё бежит — ждём файл
    best = OUT / "results_retrain_bestcfg_20261102.json"
    for _ in range(180):  # до ~1.5ч
        if best.exists():
            break
        time.sleep(30)
    if best.exists() and not already_done("retrain_bestcfg_20261102"):
        pass
    if best.exists():
        subprocess.run(
            [PY, str(REG), "--results", str(best), "--model", "lstm",
             "--dataset-snapshot", SNAP, "--note", "bestcfg T=30 mv0.7 min features", "--set-candidate"],
            check=False,
        )
    fails = 0
    for tag, extra in JOBS:
        # bestcfg уже почти = grid_t30_mv07_min — skip дубликат если bestcfg есть
        if tag == "grid_t30_mv07_min" and best.exists():
            print(f"[skip] {tag} (== retrain_bestcfg)", flush=True)
            continue
        if tag == "grid_t45_mv09_base" and (OUT / "results_retrain_20261102.json").exists():
            print(f"[skip] {tag} (== retrain_20261102)", flush=True)
            continue
        rc = run_one(tag, extra)
        if rc != 0:
            fails += 1
    summary = summarize()
    print("summary:", summary, flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
