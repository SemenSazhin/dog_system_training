#!/usr/bin/env python3
"""Reproduce Colab architecture comparison for Human Cleanup Detector.

Trains the same experiment grid as cleanup_detection/HumanCleanupDetector.ipynb
on synthetic pose sequences and writes metrics for the deck builder.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cleanup_detection.architectures import build_model, count_parameters  # noqa: E402

PROXY = ROOT / "datasets" / "cleanup_proxy" / "synthetic_pose_sequences.npz"
OUT_DIR = ROOT / "presentation" / "cleanup"
OUT_JSON = OUT_DIR / "architecture_comparison.json"
OUT_CSV = OUT_DIR / "architecture_comparison.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
WINDOW = 48
EPOCHS = 12
BATCH = 32
LR = 1e-3
WD = 1e-4

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
KEY_ANGLES = [
    (L_SHO, L_ELB, L_WRI),
    (R_SHO, R_ELB, R_WRI),
    (L_HIP, L_KNE, L_ANK),
    (R_HIP, R_KNE, R_ANK),
    (L_SHO, L_HIP, L_KNE),
    (R_SHO, R_HIP, R_KNE),
    (L_ELB, L_SHO, L_HIP),
    (R_ELB, R_SHO, R_HIP),
]

EXPERIMENTS = [
    ("lstm_base", "lstm", dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
    ("bilstm_base", "bilstm", dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
    ("gru_base", "gru", dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
    ("bigru_base", "bigru", dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=8e-4)),
    ("mamba_lite_base", "mamba_lite", dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
    ("transformer_cls", "transformer", dict(hidden_size=128, num_layers=2, nhead=4, dropout=0.3, num_classes=2, lr=8e-4)),
    ("transformer_wide", "transformer", dict(hidden_size=192, num_layers=3, nhead=6, dropout=0.35, num_classes=2, lr=5e-4)),
    ("temporal_transformer", "temporal_transformer", dict(hidden_size=128, num_layers=3, nhead=4, dropout=0.3, num_classes=2, lr=7e-4)),
]


def _angle(a, b, c):
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba) + 1e-8, np.linalg.norm(bc) + 1e-8
    cos = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos) / np.pi)


def pose_frame_features(kpts_xyv: np.ndarray) -> np.ndarray:
    xy = kpts_xyv[:, :2].astype(np.float32)
    conf = kpts_xyv[:, 2].astype(np.float32)
    mid_hip = (xy[L_HIP] + xy[R_HIP]) * 0.5
    mid_sho = (xy[L_SHO] + xy[R_SHO]) * 0.5
    scale = np.linalg.norm(mid_sho - mid_hip) + 1e-6
    rel = (xy - mid_hip) / scale
    feats = []
    feats.extend(rel.reshape(-1).tolist())
    feats.extend(conf.tolist())
    for i, j, k in KEY_ANGLES:
        feats.append(_angle(xy[i], xy[j], xy[k]))
    wrist_h = ((xy[L_WRI, 1] + xy[R_WRI, 1]) * 0.5 - mid_hip[1]) / scale
    torso_tilt = float(np.arctan2(mid_sho[0] - mid_hip[0], mid_hip[1] - mid_sho[1] + 1e-6) / np.pi)
    knee_bend = 0.5 * (_angle(xy[L_HIP], xy[L_KNE], xy[L_ANK]) + _angle(xy[R_HIP], xy[R_KNE], xy[R_ANK]))
    feats.extend([wrist_h, torso_tilt, knee_bend])
    return np.asarray(feats, dtype=np.float32)


def sequence_features(kpts_seq: np.ndarray) -> np.ndarray:
    fmat = np.stack([pose_frame_features(k) for k in kpts_seq], axis=0)
    dlt = np.zeros_like(fmat)
    dlt[1:] = fmat[1:] - fmat[:-1]
    return np.concatenate([fmat, dlt], axis=1)


class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def prepare_xy(x, y):
    if x.shape[-1] == 17 * 3:
        k = x.reshape(len(x), WINDOW, 17, 3)
        xs = np.stack([sequence_features(ki) for ki in k]).astype(np.float32)
    else:
        xs = x.astype(np.float32)
    return xs, y.astype(np.int64)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues, probs = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        pr = logits.argmax(1).cpu().numpy()
        preds.extend(pr.tolist())
        trues.extend(yb.numpy().tolist())
        probs.extend(p.tolist())
    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(trues, probs))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": auc,
    }


def train_one(tag, arch, params, train_loader, val_loader, test_loader, input_size):
    model_kwargs = {k: v for k, v in params.items() if k not in ("lr", "weight_decay")}
    model = build_model(arch, input_size, **model_kwargs).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", LR),
        weight_decay=params.get("weight_decay", WD),
    )
    crit = nn.CrossEntropyLoss()
    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate(model, val_loader)
        print(f"[{tag}] ep {ep+1}/{EPOCHS} loss={np.mean(losses):.4f} val_f1={val['f1']:.4f}")
        if val["f1"] >= best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader)
    n_params = count_parameters(model)
    return {
        "tag": tag,
        "arch": arch,
        "family": (
            "recurrent"
            if arch in ("lstm", "bilstm", "gru", "bigru")
            else ("ssm" if arch == "mamba_lite" else "transformer")
        ),
        "n_params": n_params,
        "params": params,
        **test,
    }


def main():
    random = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert PROXY.exists(), PROXY
    pack = np.load(PROXY)
    x_all, y_all = prepare_xy(pack["X"], pack["y"])
    input_size = int(x_all.shape[-1])
    print("device", DEVICE, "X", x_all.shape, "pos", int((y_all == 1).sum()), "neg", int((y_all == 0).sum()))

    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tmp, y_tmp, test_size=0.2, random_state=SEED, stratify=y_tmp
    )
    train_loader = DataLoader(SeqDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SeqDataset(x_va, y_va), batch_size=BATCH)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=BATCH)

    results = []
    for tag, arch, params in EXPERIMENTS:
        print("\n" + "=" * 70)
        print("RUN", tag)
        results.append(train_one(tag, arch, params, train_loader, val_loader, test_loader, input_size))

    results = sorted(results, key=lambda r: r["f1"], reverse=True)
    best = results[0]
    payload = {
        "task": "human_cleanup_detector",
        "notebook": "cleanup_detection/HumanCleanupDetector.ipynb",
        "colab": "https://colab.research.google.com/drive/1OBXxeuABaLc2AJexDtnmy4iNaBXjHF7E",
        "protocol": {
            "data": "synthetic COCO-17 pose sequences (cleanup squat+reach vs walk/stand)",
            "n_pos": 120,
            "n_neg": 120,
            "window": WINDOW,
            "feature_dim": input_size,
            "split": "64/16/20 stratified by label, seed 42",
            "epochs": EPOCHS,
            "optimizer": "AdamW",
            "selection": "best checkpoint by val F1",
            "metrics_on": "held-out test windows",
            "caveat": (
                "Цифры — воспроизведение сравнения архитектур Colab на synthetic proxy. "
                "Это проверка пайплайна и относительного ранжирования моделей, "
                "не итоговая оценка на реальных CCTV-роликах."
            ),
        },
        "best": best,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import csv

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["tag", "arch", "family", "n_params", "accuracy", "precision", "recall", "f1", "roc_auc"],
        )
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})

    print("\n=== COMPARISON ===")
    for r in results:
        print(
            f"{r['tag']:22} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} "
            f"acc={r['accuracy']:.4f} params={r['n_params']}"
        )
    print("BEST:", best["tag"], "F1=", round(best["f1"], 4))
    print("saved", OUT_JSON)


if __name__ == "__main__":
    main()
