#!/usr/bin/env python3
"""Догоняющий прогон: LSTM по группам точек в базовой конфигурации.

Базовый групповой эксперимент (слайды 6-7) изначально шел только для tcn/stgcn.
Сырые скелеты уже извлечены, поэтому считается только lstm и вливается в
результаты существующих тегов grp_* (results_*.json + scores_*.npz), не трогая tcn/stgcn.

python run_grp_lstm.py [workers]
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PY = r"D:\dog-ml\venv\Scripts\python.exe"
CV = str(Path(__file__).with_name("skeleton_cv.py"))
RAW = r"D:\dog-ml\raw"
OUT = Path(r"D:\dog-ml\cv")
OUT.mkdir(parents=True, exist_ok=True)

S5 = ["--seeds", "5"]
GRP = {
    "only_spine": ["--keep", "spine"], "only_hips": ["--keep", "hips"], "only_tail": ["--keep", "tail"],
    "only_front": ["--keep", "front"], "only_rear": ["--keep", "rear"],
    "spine_hips": ["--keep", "spine", "hips"], "hips_tail": ["--keep", "hips", "tail"],
    "spine_tail": ["--keep", "spine", "tail"], "legs": ["--keep", "front", "rear"],
    "tail_rear": ["--keep", "tail", "rear"], "hips_tail_rear": ["--keep", "hips", "tail", "rear"],
    "drop_spine": ["--drop", "spine"], "drop_hips": ["--drop", "hips"], "drop_tail": ["--drop", "tail"],
    "drop_front": ["--drop", "front"], "drop_rear": ["--drop", "rear"],
}
JOBS = [("grp_all", [])] + [(f"grp_{k}", a) for k, a in GRP.items()]


def merge(tag):
    """Влить результаты lstm из временного тега в существующий results/scores tag."""
    tmp = f"{tag}__lstm"
    r_tmp, r_tag = OUT / f"results_{tmp}.json", OUT / f"results_{tag}.json"
    s_tmp, s_tag = OUT / f"scores_{tmp}.npz", OUT / f"scores_{tag}.npz"
    d = json.load(open(r_tag, encoding="utf-8"))
    src = json.load(open(r_tmp, encoding="utf-8"))
    assert "lstm" in src["results"], f"нет lstm в {r_tmp}"
    d["results"]["lstm"] = src["results"]["lstm"]
    json.dump(d, open(r_tag, "w", encoding="utf-8"), indent=1)
    z_new = np.load(s_tmp, allow_pickle=True)
    z_old = np.load(s_tag, allow_pickle=True)
    data = {k: z_old[k] for k in z_old.files}
    z_old.close()
    for k in z_new.files:
        if k.startswith("lstm"):
            data[k] = z_new[k]
    z_new.close()
    np.savez(s_tag, **data)
    for f in (r_tmp, s_tmp, OUT / f"log_{tmp}.txt"):
        try:
            f.exists() and f.unlink()
        except PermissionError:
            pass


def run(job):
    tag, extra = job
    r_tag = OUT / f"results_{tag}.json"
    if r_tag.exists() and "lstm" in json.load(open(r_tag, encoding="utf-8"))["results"]:
        print(f"[skip] {tag}: lstm уже есть", flush=True)
        return tag, "skip", 0
    tmp = f"{tag}__lstm"
    cmd = [PY, "-u", CV, "--raw", RAW, "--out", str(OUT), "--tag", tmp,
           "--models", "lstm", *S5, "--no-asp", *extra]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    t0 = time.time()
    with open(OUT / f"log_{tmp}.txt", "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc == 0:
        merge(tag)
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: merged, {time.time() - t0:.0f}s", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: rc={rc} (лог log_{tmp}.txt)", flush=True)
    return tag, rc, time.time() - t0


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"{len(JOBS)} jobs, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(run, JOBS):
            print("done:", r, flush=True)
    print("QUEUE_DONE", flush=True)
