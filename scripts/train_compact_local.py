#!/usr/bin/env python3
"""Локальное обучение CompactLSTM (как в Colab) на D:\\dog-ml\\raw + CUDA.

Совместимо с video-scan: признаки EfficientFeatureExtractor + дельты (вход 144).
Окно по умолчанию 60 шагов @15fps (~4с) — лучший balanced CV-конфиг.

Пример:
  D:\\dog-ml\\venv\\Scripts\\python.exe -u train_compact_local.py
  D:\\dog-ml\\venv\\Scripts\\python.exe -u train_compact_local.py --seq-len 60 --epochs 40 --finetune
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]  # Детекция Собак
VS = ROOT / "video-scan"
sys.path.insert(0, str(VS))
from compact_model import CompactLSTMPoseClassifier, EfficientFeatureExtractor, ImprovedPoseProcessor  # noqa: E402

RAW = Path(r"D:\dog-ml\raw")
OUT = Path(r"D:\dog-ml\registry\models")
VIDEO_SCAN_WEIGHTS = VS / "models" / "best_lstm_model.pth"
TARGET_FPS = 15.0


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor([self.y[i]], dtype=torch.float32)


def resample_kps(kps: np.ndarray, fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
    if fps <= 0:
        fps = 30.0
    n = len(kps)
    if n < 2:
        return kps
    dur = (n - 1) / fps
    m = max(2, int(round(dur * target_fps)) + 1)
    idx = np.linspace(0, n - 1, m)
    out = np.empty((m,) + kps.shape[1:], dtype=np.float32)
    base = np.arange(n)
    for j in range(kps.shape[1]):
        for c in range(kps.shape[2]):
            col = kps[:, j, c]
            # NaN -> 0 for interp
            good = np.isfinite(col)
            if good.sum() < 2:
                out[:, j, c] = 0.0
                continue
            out[:, j, c] = np.interp(idx, base[good], col[good])
    return out


def build_feat_seq(kps: np.ndarray, extractor: EfficientFeatureExtractor, pose_filter: ImprovedPoseProcessor):
    feats = []
    prev_kps = None
    prev_f = None
    for i in range(len(kps)):
        frame = kps[i].astype(np.float32)
        # NaN -> 0
        frame = np.nan_to_num(frame, nan=0.0)
        filtered = pose_filter.filter_keypoints(frame)
        if filtered is None:
            filtered = frame
        f = extractor.extract_features(filtered, prev_kps)
        delta = f - prev_f if prev_f is not None else np.zeros_like(f)
        prev_kps = filtered.copy()
        prev_f = f.copy()
        feats.append(np.concatenate([f, delta]))
    return np.asarray(feats, dtype=np.float32)


def windows_from_video(feat: np.ndarray, label: int, seq_len: int, stride: int, min_valid: float, vid: str):
    # valid step: any nonzero coords in first 30 dims roughly — use L2 of feat
    valid = np.linalg.norm(feat[:, :30], axis=1) > 1e-6
    out = []
    if len(feat) < seq_len:
        return out
    for start in range(0, len(feat) - seq_len + 1, stride):
        sl = slice(start, start + seq_len)
        if valid[sl].mean() < min_valid:
            continue
        out.append((feat[sl], label, vid))
    return out


def load_all(seq_len: int, stride: int, min_valid: float):
    extractor = EfficientFeatureExtractor(num_keypoints=20)
    pose_filter = ImprovedPoseProcessor(num_keypoints=20)
    index_path = RAW / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    items = []
    for p in sorted(RAW.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        cls = int(z["cls"]) if "cls" in z.files else int(p.name[0])
        name = str(z["name"]) if "name" in z.files else p.stem
        fps = float(z["fps"]) if "fps" in z.files else 30.0
        kps = z["kps"].astype(np.float32)
        kps = resample_kps(kps, fps)
        feat = build_feat_seq(kps, extractor, pose_filter)
        vid = p.stem
        items.extend(windows_from_video(feat, cls, seq_len, stride, min_valid, vid))
        if index.get(vid) is None:
            index[vid] = {"cls": cls, "name": name}
    if not items:
        raise SystemExit("нет окон — проверьте raw/seq_len/min_valid")
    X = np.stack([a[0] for a in items])
    y = np.array([a[1] for a in items], dtype=np.int64)
    groups = np.array([a[2] for a in items])
    feat_dim = X.shape[-1]
    print(f"windows={len(X)} pos={int(y.sum())} neg={int((1-y).sum())} vids={len(set(groups))} feat_dim={feat_dim} seq={seq_len}")
    return X, y, groups, feat_dim


def split_group(X, y, groups, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
    tr2, va = next(gss2.split(X[tr_idx], y[tr_idx], groups[tr_idx]))
    return (
        X[tr_idx][tr2], y[tr_idx][tr2],
        X[tr_idx][va], y[tr_idx][va],
        X[te_idx], y[te_idx],
    )


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    crit = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += crit(logits, yb).item() * len(xb)
        n += len(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
        ys.append(yb.cpu().numpy().ravel())
        ps.append(prob)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.float32)
    metrics = {
        "loss": loss_sum / max(n, 1),
        "acc": float((pred == y).mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
    }
    return metrics, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-valid", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--finetune", action="store_true", help="старт с текущих video-scan весов")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deploy", action="store_true", help="скопировать в video-scan/models и обновить SEQ_LENGTH")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    X, y, groups, feat_dim = load_all(args.seq_len, args.stride, args.min_valid)
    Xtr, ytr, Xva, yva, Xte, yte = split_group(X, y, groups, args.seed)
    print(f"split train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    train_loader = DataLoader(SeqDS(Xtr, ytr), batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDS(Xva, yva), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(SeqDS(Xte, yte), batch_size=args.batch, shuffle=False)

    model = CompactLSTMPoseClassifier(input_size=feat_dim).to(device)
    if args.finetune and VIDEO_SCAN_WEIGHTS.exists():
        sd = torch.load(VIDEO_SCAN_WEIGHTS, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        # encoder.0.weight shape[1] must match
        in_w = sd["encoder.0.weight"].shape[1]
        if in_w != feat_dim:
            print(f"finetune skip: weight in={in_w} != feat_dim={feat_dim}")
        else:
            model.load_state_dict(sd, strict=True)
            print("finetune: loaded", VIDEO_SCAN_WEIGHTS)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = nn.BCEWithLogitsLoss()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / f"compact_seq{args.seq_len}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_lstm_model.pth"
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        vm, _, _ = eval_loader(model, val_loader, device)
        sched.step(vm["f1"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in vm.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={vm['f1']:.3f} val_auc={vm['auc']} val_ap={vm['ap']}"
        )
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print("  saved", best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    tm, _, _ = eval_loader(model, test_loader, device)
    print("TEST", tm)
    meta = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "min_valid": args.min_valid,
        "target_fps": TARGET_FPS,
        "feat_dim": feat_dim,
        "best_val_f1": best_f1,
        "test": tm,
        "finetune": bool(args.finetune),
        "history": history,
        "weights": str(best_path),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.deploy:
        bak = VIDEO_SCAN_WEIGHTS.with_suffix(f".pth.bak_{stamp}")
        if VIDEO_SCAN_WEIGHTS.exists():
            shutil.copy2(VIDEO_SCAN_WEIGHTS, bak)
            print("backup", bak)
        shutil.copy2(best_path, VIDEO_SCAN_WEIGHTS)
        # SEQ_LENGTH в detector.py
        det = VS / "detector.py"
        text = det.read_text(encoding="utf-8")
        old = "SEQ_LENGTH = 120"
        new = f"SEQ_LENGTH = {args.seq_len}"
        if old in text:
            det.write_text(text.replace(old, new, 1), encoding="utf-8")
            print("updated", det, "->", new)
        # production registry
        prod_path = Path(r"D:\dog-ml\registry\production.json")
        prod = json.loads(prod_path.read_text(encoding="utf-8")) if prod_path.exists() else {}
        prev = prod.get("active")
        if prev:
            prod.setdefault("history", []).append(prev)
        prod["active"] = {
            "experiment_id": f"compact_local_{stamp}",
            "note": f"local CUDA train seq={args.seq_len} min_valid={args.min_valid}",
            "weights": str(VIDEO_SCAN_WEIGHTS),
            "metrics": {"test_auc": tm.get("auc"), "test_ap": tm.get("ap"), "test_f1": tm.get("f1")},
            "cv_candidate": (prod.get("candidate") or {}).get("experiment_id"),
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
        }
        prod_path.write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")
        print("deployed to video-scan weights + production.json")

    print("DONE", run_dir)


if __name__ == "__main__":
    main()
