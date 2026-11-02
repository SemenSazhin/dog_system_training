#!/usr/bin/env python3
"""Сравнение candidate vs active и (опционально) выкат LSTM в video-scan.

Критерий по умолчанию: AUC не хуже baseline-0.005 и cam_fpr_05 не хуже +0.02,
либо явный --force.

Пример:
  python promote_model.py --dry-run
  python promote_model.py --weights D:\\dog-ml\\registry\\models\\....pth --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

REGISTRY = Path(r"D:\dog-ml\registry")
PRODUCTION = REGISTRY / "production.json"
VIDEO_SCAN_MODELS = Path(r"C:\Users\Zhenya\Desktop\Детекция Собак\video-scan\models")
DEFAULT_NAME = "best_lstm_model.pth"


def load_prod() -> dict:
    if PRODUCTION.exists():
        return json.loads(PRODUCTION.read_text(encoding="utf-8"))
    return {}


def decide(active: dict, candidate: dict) -> tuple[bool, list[str]]:
    reasons = []
    am = (active or {}).get("metrics") or {}
    cm = (candidate or {}).get("metrics") or {}
    if not cm:
        return False, ["у candidate нет metrics"]

    auc_a = (am.get("auc") or {}).get("mean")
    auc_c = (cm.get("auc") or {}).get("mean")
    fpr_a = (am.get("cam_fpr_05") or {}).get("mean")
    fpr_c = (cm.get("cam_fpr_05") or {}).get("mean")

    ok = True
    if auc_a is not None and auc_c is not None:
        if auc_c + 1e-12 < auc_a - 0.005:
            ok = False
            reasons.append(f"AUC хуже: {auc_c:.3f} < {auc_a:.3f}-0.005")
        else:
            reasons.append(f"AUC ок: {auc_c:.3f} vs {auc_a:.3f}")
    elif auc_c is not None:
        reasons.append(f"AUC candidate={auc_c:.3f} (baseline AUC нет)")
    else:
        ok = False
        reasons.append("нет AUC у candidate")

    if fpr_a is not None and fpr_c is not None:
        if fpr_c > fpr_a + 0.02:
            ok = False
            reasons.append(f"cam_fpr_05 хуже: {fpr_c:.3f} > {fpr_a:.3f}+0.02")
        else:
            reasons.append(f"cam_fpr_05 ок: {fpr_c:.3f} vs {fpr_a:.3f}")

    return ok, reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--name", default=DEFAULT_NAME, help="имя файла в video-scan/models")
    ap.add_argument("--apply", action="store_true", help="реально копировать веса и обновить production.json")
    ap.add_argument("--force", action="store_true", help="игнорировать пороги сравнения")
    ap.add_argument("--dry-run", action="store_true", help="только показать решение")
    args = ap.parse_args()
    if not args.apply:
        args.dry_run = True

    prod = load_prod()
    active = prod.get("active") or {}
    candidate = prod.get("candidate") or {}
    ok, reasons = decide(active, candidate)
    print("active:", active.get("experiment_id") or active.get("note") or "(пусто)")
    print("candidate:", candidate.get("experiment_id") or candidate.get("note") or "(пусто)")
    for r in reasons:
        print("-", r)

    weights = args.weights
    if weights is None and candidate.get("weights"):
        weights = Path(candidate["weights"])

    if not ok and not args.force:
        print("РЕШЕНИЕ: не выкатывать")
        return 2

    if weights is None or not Path(weights).exists():
        print("РЕШЕНИЕ: метрики ок, но нет файла весов (--weights)")
        return 3

    dest = VIDEO_SCAN_MODELS / args.name
    backup = VIDEO_SCAN_MODELS / f"{args.name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"будет: {weights} -> {dest}")
    if args.dry_run and not args.apply:
        print("DRY-RUN (добавьте --apply)")
        return 0

    VIDEO_SCAN_MODELS.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.copy2(dest, backup)
        print("backup:", backup)
    shutil.copy2(weights, dest)

    prod["active"] = {
        **candidate,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "video_scan_path": str(dest),
        "forced": bool(args.force),
    }
    prod["history"] = list(prod.get("history") or [])
    if active:
        prod["history"].append(active)
    prod["candidate"] = {}
    PRODUCTION.write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")
    print("выкачено в video-scan; обновите контейнер: docker compose up -d --build  или docker cp + restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
