"""
Download proxy datasets for human "cleaned up after dog" pose/action training.

Exact public video datasets of people picking up dog feces are extremely rare.
This script downloads the closest open proxies + metadata into ./datasets/.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CHARADES_DIR = ROOT / "charades"
SHITSPOTTER_DIR = ROOT / "shitspotter"
STIPRA_DIR = ROOT / "stipra"
MANIFESTS_DIR = ROOT / "manifests"
PROXY_DIR = ROOT / "cleanup_proxy"

CHARADES_ANNOTATIONS_URL = (
    "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
)

# Classes useful as cleanup / bend / throw proxies
CHARADES_CLEANUP_LIKE = {
    "c015": "Throwing a bag somewhere",
    "c016": "Opening a bag",
    "c019": "Throwing a broom somewhere",
    "c020": "Holding a broom",
    "c021": "Taking a broom from somewhere",
    "c022": "Putting a broom somewhere",
    "c035": "Throwing clothes somewhere",
    "c037": "Throwing a dish somewhere",
    "c046": "Throwing a laptop somewhere",
    "c052": "Throwing a pillow/cushion somewhere",
    "c061": "Throwing a shoe/sneakers somewhere",
    "c068": "Throwing a towel/s somewhere",
    "c074": "Throwing a blanket somewhere",
    "c092": "Throwing a box somewhere",
    "c100": "Throwing something somewhere",
    "c106": "Taking something from somewhere",
    "c107": "Putting something on a table",
    "c108": "Throwing something somewhere",
    "c118": "Standing up from somewhere",
    "c119": "Standing up from somewhere",
    "c148": "Sitting down",
    "c151": "Standing up",
}


def _download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest}")
        return dest
    print(f"[get] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
                    pbar.update(len(block))
        tmp.replace(dest)
    return dest


def download_charades_annotations() -> dict:
    CHARADES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _download_file(CHARADES_ANNOTATIONS_URL, CHARADES_DIR / "Charades.zip")
    extract_dir = CHARADES_DIR / "Charades"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CHARADES_DIR)

    classes_path = extract_dir / "Charades_v1_classes.txt"
    train_csv = extract_dir / "Charades_v1_train.csv"
    test_csv = extract_dir / "Charades_v1_test.csv"

    class_map = {}
    if classes_path.exists():
        for line in classes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            code, name = line.split(" ", 1)
            class_map[code] = name

    cleanup_rows = []
    for split, path in (("train", train_csv), ("test", test_csv)):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row.get("actions") or ""
                matched = []
                for code in class_map:
                    if code in actions and (
                        code in CHARADES_CLEANUP_LIKE
                        or any(
                            k in class_map[code].lower()
                            for k in ("throw", "pick", "tidy", "put", "take", "stand", "sit")
                        )
                    ):
                        matched.append(code)
                if matched:
                    cleanup_rows.append(
                        {
                            "split": split,
                            "id": row.get("id"),
                            "actions": actions,
                            "matched_codes": matched,
                            "matched_names": [class_map.get(c, c) for c in matched],
                            "length": row.get("length"),
                            "scene": row.get("scene"),
                        }
                    )

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFESTS_DIR / "charades_cleanup_like.json"
    payload = {
        "source": CHARADES_ANNOTATIONS_URL,
        "note": (
            "Charades has no exact 'pick up dog poop' class. "
            "These are throw/put/take/tidy/stand proxies for human cleanup motion."
        ),
        "n_videos": len(cleanup_rows),
        "videos": cleanup_rows[:5000],
        "class_map_subset": {k: class_map.get(k, v) for k, v in CHARADES_CLEANUP_LIKE.items()},
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Charades cleanup-like videos: {len(cleanup_rows)} -> {out}")
    return payload


def download_shitspotter_sample(max_files: int = 40) -> dict:
    """Download one webdataset shard and extract a small image sample."""
    SHITSPOTTER_DIR.mkdir(parents=True, exist_ok=True)
    sample_dir = SHITSPOTTER_DIR / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)

    shard = hf_hub_download(
        repo_id="erotemic/shitspotter",
        repo_type="dataset",
        filename="webdataset_shards/train/train-000000.tar",
        local_dir=str(SHITSPOTTER_DIR),
    )

    extracted = []
    with tarfile.open(shard, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".jpg")]
        for member in members[:max_files]:
            dest = sample_dir / Path(member.name).name
            if not dest.exists():
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                dest.write_bytes(fobj.read())
            extracted.append(dest.name)

    meta = {
        "source": "https://huggingface.co/datasets/erotemic/shitspotter",
        "note": (
            "ShitSpotter is before/after outdoor dog-feces PHOTOS, not action videos. "
            "Useful as domain context / detection prior, not as human-pose sequences."
        ),
        "shard": shard,
        "n_extracted": len(extracted),
        "files": extracted,
    }
    out = MANIFESTS_DIR / "shitspotter_sample.json"
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] ShitSpotter sample images: {len(extracted)} -> {sample_dir}")
    return meta


def download_stipra_metadata_only() -> dict:
    """Save Stipra card + download instructions (full videos.zip ~21.7GB)."""
    STIPRA_DIR.mkdir(parents=True, exist_ok=True)
    readme = hf_hub_download(
        repo_id="TeknTrash/Stipra",
        repo_type="dataset",
        filename="README.md",
        local_dir=str(STIPRA_DIR),
    )
    meta = {
        "source": "https://huggingface.co/datasets/TeknTrash/Stipra",
        "videos_zip_gb": 21.7,
        "n_videos": 1273,
        "license": "osl-3.0",
        "relevance": (
            "Closest open video dataset: people disposing household products into garbage. "
            "Useful positive proxy for cleanup / bend / bag / discard motions."
        ),
        "readme_path": readme,
        "full_download_command": (
            "huggingface-cli download TeknTrash/Stipra --repo-type dataset "
            "--include videos.zip --local-dir datasets/stipra"
        ),
    }
    out = MANIFESTS_DIR / "stipra_info.json"
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (STIPRA_DIR / "DOWNLOAD.md").write_text(
        "# Stipra (TeknTrash)\n\n"
        "Full archive is ~21.7 GB (`videos.zip`, 1273 mp4).\n\n"
        "```bash\n"
        "pip install huggingface_hub\n"
        "huggingface-cli download TeknTrash/Stipra --repo-type dataset "
        "--include videos.zip --local-dir datasets/stipra\n"
        "```\n\n"
        "Then unzip and treat clips as **positive cleanup-like** examples.\n",
        encoding="utf-8",
    )
    print(f"[ok] Stipra metadata saved -> {out}")
    return meta


def write_dataset_catalog() -> None:
    catalog = {
        "task": "Detect whether a person cleaned up after a dog (human pose / action on video)",
        "exact_match_found": False,
        "sequence_architectures": [
            "lstm",
            "bilstm",
            "gru",
            "bigru",
            "mamba_lite",
            "transformer",
            "temporal_transformer",
        ],
        "pretrained_human_pose": [
            {
                "name": "YOLOv8n-pose / YOLOv8s-pose / YOLOv8m-pose",
                "source": "Ultralytics",
                "weights": ["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"],
                "keypoints": 17,
                "train_data": "COCO Keypoints",
            },
            {
                "name": "YOLOv11n-pose",
                "source": "Ultralytics",
                "weights": ["yolo11n-pose.pt"],
                "keypoints": 17,
                "train_data": "COCO Keypoints",
            },
            {
                "name": "MediaPipe Pose Landmarker",
                "source": "Google MediaPipe",
                "keypoints": 33,
                "note": "Good CPU realtime alternative; different skeleton than COCO-17",
            },
        ],
        "datasets": [
            {
                "id": "stipra",
                "url": "https://huggingface.co/datasets/TeknTrash/Stipra",
                "type": "video",
                "size": "~21.7 GB / 1273 videos",
                "use": "positive proxy: dispose/throw trash",
            },
            {
                "id": "shitspotter",
                "url": "https://huggingface.co/datasets/erotemic/shitspotter",
                "type": "images",
                "size": "~30GB+ shards",
                "use": "dog feces before/after photos (not human action)",
            },
            {
                "id": "charades",
                "url": "https://allenai.org/data/charades",
                "type": "video+annotations",
                "use": "throw/put/take/tidy/stand/sit proxies",
            },
            {
                "id": "kinetics_related",
                "url": "https://github.com/cvdfoundation/kinetics-dataset",
                "type": "video urls",
                "use": "squat / sweeping floor / garbage collecting / picking fruit",
            },
            {
                "id": "stair_actions",
                "url": "https://actions.stair.center",
                "type": "video",
                "use": "everyday home actions incl. throwing trash",
            },
        ],
        "label_scheme_recommended": {
            "0": "no_cleanup",
            "1": "cleanup",
            "notes": (
                "Prefer clip-level labels from videos. "
                "If only own CCTV data exists, annotate windows where person bends, "
                "reaches ground near dog waste, bags it, and stands up."
            ),
        },
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / "DATASETS_CATALOG.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "README.md").write_text(
        "# Datasets for human cleanup-after-dog detection\n\n"
        "Exact public datasets of people picking up dog feces are essentially nonexistent.\n"
        "This folder stores the closest open proxies and manifests.\n\n"
        "## Downloaded / prepared here\n"
        "- `charades/` — Charades annotations + cleanup-like video id list\n"
        "- `shitspotter/sample_images/` — small image sample (before/after dog feces)\n"
        "- `stipra/` — metadata + instructions for full 21.7GB trash-disposal videos\n"
        "- `cleanup_proxy/` — synthetic pose sequences for local pipeline tests\n"
        "- `manifests/` — catalogs and filtered lists\n\n"
        "## Download\n"
        "```bash\n"
        "py -3 datasets/download_datasets.py --all\n"
        "py -3 datasets/download_datasets.py --stipra-full   # optional, ~22GB\n"
        "```\n",
        encoding="utf-8",
    )
    print(f"[ok] catalog -> {path}")


def generate_synthetic_pose_sequences(
    n_pos: int = 120,
    n_neg: int = 120,
    window: int = 48,
    n_kp: int = 17,
) -> None:
    """Synthetic COCO-17 pose sequences: cleanup (squat+reach) vs walk/stand."""
    import numpy as np

    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def base_pose(t: float, mode: str) -> np.ndarray:
        # [K, 3] = x,y,conf  normalized
        kp = np.zeros((n_kp, 3), dtype=np.float32)
        # torso
        kp[0] = [0.50, 0.18, 0.95]  # nose
        kp[5] = [0.45, 0.30, 0.95]  # L shoulder
        kp[6] = [0.55, 0.30, 0.95]  # R shoulder
        kp[11] = [0.46, 0.55, 0.95]  # L hip
        kp[12] = [0.54, 0.55, 0.95]  # R hip
        if mode == "cleanup":
            # squat depth + arm reach down
            squat = 0.12 * np.sin(np.pi * min(max((t - 0.2) / 0.5, 0.0), 1.0))
            reach = 0.20 * np.sin(np.pi * min(max((t - 0.25) / 0.45, 0.0), 1.0))
            kp[[5, 6, 11, 12], 1] += squat
            kp[0, 1] += squat * 0.8
            kp[9] = [0.42, 0.70 + reach, 0.9]   # L wrist down
            kp[10] = [0.58, 0.70 + reach, 0.9]  # R wrist down
            kp[7] = [0.43, 0.50 + reach * 0.5, 0.9]
            kp[8] = [0.57, 0.50 + reach * 0.5, 0.9]
            kp[13] = [0.45, 0.75 + squat, 0.9]
            kp[14] = [0.55, 0.75 + squat, 0.9]
            kp[15] = [0.44, 0.92, 0.9]
            kp[16] = [0.56, 0.92, 0.9]
        else:
            # walk / stand with mild sway
            sway = 0.03 * np.sin(2 * np.pi * t * 2)
            kp[:, 0] += sway
            kp[9] = [0.40, 0.48, 0.9]
            kp[10] = [0.60, 0.48, 0.9]
            kp[7] = [0.42, 0.40, 0.9]
            kp[8] = [0.58, 0.40, 0.9]
            kp[13] = [0.46, 0.72 + 0.02 * np.sin(2 * np.pi * t * 3), 0.9]
            kp[14] = [0.54, 0.72 + 0.02 * np.cos(2 * np.pi * t * 3), 0.9]
            kp[15] = [0.45, 0.92, 0.9]
            kp[16] = [0.55, 0.92, 0.9]
        noise = rng.normal(0, 0.008, size=kp.shape).astype(np.float32)
        noise[:, 2] = 0
        kp = np.clip(kp + noise, 0.0, 1.0)
        kp[:, 2] = np.clip(kp[:, 2], 0.5, 1.0)
        return kp

    def make_seq(label: int) -> np.ndarray:
        mode = "cleanup" if label == 1 else "other"
        frames = []
        for i in range(window):
            t = i / max(window - 1, 1)
            frames.append(base_pose(t, mode).reshape(-1))
        return np.stack(frames, axis=0)

    X, y = [], []
    for _ in range(n_pos):
        X.append(make_seq(1))
        y.append(1)
    for _ in range(n_neg):
        X.append(make_seq(0))
        y.append(0)

    X_np = np.stack(X).astype(np.float32)
    y_np = np.asarray(y, dtype=np.int64)
    np.savez_compressed(PROXY_DIR / "synthetic_pose_sequences.npz", X=X_np, y=y_np)
    meta = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "window": window,
        "feature_dim": int(X_np.shape[-1]),
        "keypoints": n_kp,
        "labels": {"0": "no_cleanup", "1": "cleanup"},
        "note": "Synthetic COCO-17 pose sequences for pipeline/unit tests before real video labeling.",
    }
    (PROXY_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] synthetic sequences {X_np.shape} -> {PROXY_DIR}")


def maybe_download_stipra_full() -> None:
    print("[warn] Stipra full archive ~21.7GB — starting download...")
    path = hf_hub_download(
        repo_id="TeknTrash/Stipra",
        repo_type="dataset",
        filename="videos.zip",
        local_dir=str(STIPRA_DIR),
    )
    print(f"[ok] downloaded {path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--charades", action="store_true")
    parser.add_argument("--shitspotter", action="store_true")
    parser.add_argument("--stipra-meta", action="store_true")
    parser.add_argument("--stipra-full", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.all or not any(
        [args.charades, args.shitspotter, args.stipra_meta, args.stipra_full, args.synthetic]
    ):
        args.charades = args.shitspotter = args.stipra_meta = args.synthetic = True

    write_dataset_catalog()
    if args.charades:
        download_charades_annotations()
    if args.shitspotter:
        download_shitspotter_sample()
    if args.stipra_meta:
        download_stipra_metadata_only()
    if args.synthetic:
        generate_synthetic_pose_sequences()
    if args.stipra_full:
        maybe_download_stipra_full()


if __name__ == "__main__":
    main()
