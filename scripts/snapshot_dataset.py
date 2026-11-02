#!/usr/bin/env python3
"""Снимок датасета: сколько клипов в классах 0/1, откуда они, что добавилось недавно.

Пишет JSON в D:\\dog-ml\\registry\\snapshots\\ и строку в dataset_ledger.jsonl.

Пример:
  python snapshot_dataset.py
  python snapshot_dataset.py --data "G:\\Мой диск\\Система контроля за собаками\\data" --note "после выгрузки 01.11"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_DATA = Path(r"G:\Мой диск\Система контроля за собаками\data")
REGISTRY = Path(r"D:\dog-ml\registry")
EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CAM_RE = re.compile(
    r"(?:nodf|df|hc)_([a-z0-9_]+?)_\d{2}_\d{2}_\d{4}",
    re.IGNORECASE,
)


def file_id(path: Path) -> str:
    st = path.stat()
    raw = f"{path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def camera_of(name: str) -> str:
    m = CAM_RE.search(name)
    if m:
        return m.group(1)
    if name.lower().startswith(("nodf_", "df_", "hc_")):
        parts = name.split("_")
        if len(parts) >= 3:
            return "_".join(parts[1:-4]) if len(parts) > 5 else "_".join(parts[1:])
    return "other_or_manual"


def scan_class(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            st = p.stat()
            out.append(
                {
                    "name": p.name,
                    "rel": str(p.relative_to(folder.parent)).replace("\\", "/"),
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "mtime_epoch": st.st_mtime,
                    "id": file_id(p),
                    "camera": camera_of(p.name),
                    # Префикс df_/nodf_ — автолейбл video-scan; класс папки — вердикт человека.
                    # Расхождение не ошибка, а hard example (ложный+/ложный-).
                    "auto_human_disagree": (
                        (folder.name == "1" and p.name.lower().startswith("nodf_"))
                        or (folder.name == "0" and p.name.lower().startswith("df_"))
                    ),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--note", default="")
    ap.add_argument("--since-days", type=int, default=1, help="сколько дней считать 'новыми'")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"нет папки датасета: {args.data}")

    REGISTRY.mkdir(parents=True, exist_ok=True)
    (REGISTRY / "snapshots").mkdir(parents=True, exist_ok=True)

    cls0 = scan_class(args.data / "0")
    cls1 = scan_class(args.data / "1")
    now = time.time()
    since = now - args.since_days * 86400

    def summarize(items: list[dict]) -> dict:
        cams = Counter(i["camera"] for i in items)
        new_items = [i for i in items if i["mtime_epoch"] >= since]
        return {
            "n": len(items),
            "bytes": int(sum(i["size"] for i in items)),
            "new_n": len(new_items),
            "cameras": dict(cams.most_common()),
            "auto_human_disagree": [i["name"] for i in items if i["auto_human_disagree"]],
            "newest": sorted(items, key=lambda x: x["mtime_epoch"], reverse=True)[:8],
        }

    snap = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data),
        "note": args.note,
        "since_days": args.since_days,
        "class_0": summarize(cls0),
        "class_1": summarize(cls1),
        "total": len(cls0) + len(cls1),
        "balance": {
            "neg": len(cls0),
            "pos": len(cls1),
            "pos_share": round(len(cls1) / max(1, len(cls0) + len(cls1)), 4),
        },
        "all_ids": {
            "0": sorted(i["id"] for i in cls0),
            "1": sorted(i["id"] for i in cls1),
        },
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REGISTRY / "snapshots" / f"dataset_{stamp}.json"
    # ids тяжёлые для ledger — в ledger краткая запись
    out_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger_row = {
        "ts": snap["ts"],
        "snapshot": out_path.name,
        "note": args.note,
        "n0": snap["class_0"]["n"],
        "n1": snap["class_1"]["n"],
        "new0": snap["class_0"]["new_n"],
        "new1": snap["class_1"]["new_n"],
        "pos_share": snap["balance"]["pos_share"],
        "disagree_n": len(snap["class_0"]["auto_human_disagree"]) + len(snap["class_1"]["auto_human_disagree"]),
    }
    with (REGISTRY / "dataset_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")

    print(f"snapshot -> {out_path}")
    print(
        f"класс 0 (нет): {snap['class_0']['n']} (новых {snap['class_0']['new_n']}) | "
        f"класс 1 (есть): {snap['class_1']['n']} (новых {snap['class_1']['new_n']}) | "
        f"доля pos={snap['balance']['pos_share']:.1%}"
    )
    disagree = snap["class_0"]["auto_human_disagree"] + snap["class_1"]["auto_human_disagree"]
    if disagree:
        print(f"расхождение автолейбла video-scan и вердикта человека: {len(disagree)} (это hard examples)")
        for name in disagree[:8]:
            print(" ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
