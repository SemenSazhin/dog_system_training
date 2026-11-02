#!/usr/bin/env python3
"""Перенос проверенных клипов video-scan с Яндекс Диска в датасет на Google Диске (диск G:, Google Drive for desktop).

Переносятся только клипы с вердиктом из verdicts.json ("0" - нет дефекации, "1" - есть дефекация).
Вердикты ставятся после просмотра кадров и подтверждения человеком; без вердикта клип не трогается.
Порядок для каждого клипа: скачать (если нет локальной копии) -> скопировать в data\\<класс> на G: ->
сверить md5 копии -> только после этого перенести исходник на Яндексе в корзину (обратимо, не навсегда).

Без --apply ничего не копируется и не удаляется, пишется только отчёт.
--trash-bad дополнительно отправляет в корзину клипы, не прошедшие техническую проверку (см. check_clip).
Папки health_check, *_with_boxes и defecation этот скрипт не переносит.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ENV = Path(r"C:\Users\Zhenya\Desktop\Детекция Собак\video-scan\.env")
BASE = "disk:/Датасет Собак"
WORK = Path(r"D:\dog-ml\yd_sync")
CLIPS = WORK / "clips"
VERDICTS = WORK / "verdicts.json"      # {путь на Яндексе: "0" | "1" | "skip"}
STATE = WORK / "state.json"
DATA_DIR = Path(r"G:\Мой диск\Система контроля за собаками\data")
API = "https://cloud-api.yandex.net/v1/disk/resources"
MIN_SEC, MIN_SIDE, MIN_READ, MIN_MOTION = 3.0, 360, 0.9, 0.15
DOG_WEIGHTS = r"D:\dog-ml\bench\yolo26s.pt"


def yandex_token():
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("YANDEX_DISK_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("YANDEX_DISK_TOKEN не найден в .env")


def ycall(token, method, url):
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"OAuth {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return json.loads(body) if body else {}


def ylist(token, path):
    out, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"path": path, "limit": 500, "offset": offset,
                                    "fields": "_embedded.items.name,_embedded.items.type,_embedded.items.size,_embedded.items.md5,_embedded.items.path,_embedded.total"})
        emb = ycall(token, "GET", f"{API}?{q}")["_embedded"]
        out += emb["items"]
        offset += 500
        if offset >= emb["total"]:
            return out


def ydownload(token, path, dest, tries=3):
    for k in range(tries):
        try:
            href = ycall(token, "GET", "https://cloud-api.yandex.net/v1/disk/resources/download?"
                         + urllib.parse.urlencode({"path": path}))["href"]
            urllib.request.urlretrieve(href, dest)
            return
        except (urllib.error.URLError, OSError):
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def ytrash(token, path):
    # permanently=false: файл идёт в корзину Яндекс Диска и восстанавливается
    ycall(token, "DELETE", f"{API}?" + urllib.parse.urlencode({"path": path, "permanently": "false"}))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_clip(path, dog_model):
    cap = cv2.VideoCapture(str(path))
    try:
        return _check_clip(cap, dog_model)
    finally:
        cap.release()


def _check_clip(cap, dog_model):
    if not cap.isOpened():
        return False, "не открывается"
    n, fps = int(cap.get(7)), cap.get(5) or 0
    w, h = int(cap.get(3)), int(cap.get(4))
    if n <= 0 or fps <= 0:
        return False, "нет кадров или fps"
    if n / fps < MIN_SEC:
        return False, f"короткий ({n / fps:.1f} с)"
    if min(w, h) < MIN_SIDE:
        return False, f"малое разрешение {w}x{h}"
    idx = np.linspace(0, n - 1, 6).astype(int)
    frames = []
    for i in idx:
        cap.set(1, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    if len(frames) / len(idx) < MIN_READ:
        return False, "кадры не читаются"
    g = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(g, g[1:])]))
    if motion < MIN_MOTION:
        return False, f"статичный кадр (движение {motion:.2f})"
    if dog_model is not None:
        found = sum(len(dog_model.predict(f, imgsz=1280, conf=0.25, classes=[16], verbose=False)[0].boxes) > 0
                    for f in frames)
        if found == 0:
            return False, "собака не найдена"
    return True, "ok"


def copy_verified(local, dest_dir):
    """Копирует клип в dest_dir, сверяет md5 копии; возвращает (путь, ok). Существующий файл с тем же md5 считается копией."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_md5 = md5(local)
    dest = dest_dir / local.name.split("__", 2)[-1]
    if dest.exists():
        if md5(dest) == src_md5:
            return dest, True
        dest = dest_dir / f"{dest.stem}_{src_md5[:6]}{dest.suffix}"
    shutil.copy2(local, dest)
    return dest, md5(dest) == src_md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="копировать на G: и переносить исходники в корзину")
    ap.add_argument("--trash-bad", action="store_true", help="в корзину и клипы, не прошедшие техническую проверку")
    ap.add_argument("--no-dog-check", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.exists() else {}
    if a.apply and not DATA_DIR.exists():
        raise SystemExit(f"{DATA_DIR} недоступна: запустите Google Drive for desktop")
    token = yandex_token()
    dog = None
    if a.trash_bad and not a.no_dog_check:
        from ultralytics import YOLO
        dog = YOLO(DOG_WEIGHTS)

    listed = {}
    for folder in ("no_defecation",):
        for cam in ylist(token, f"{BASE}/{folder}"):
            if cam["type"] == "dir":
                for f in ylist(token, cam["path"]):
                    if f["type"] == "file":
                        listed[f["path"]] = (folder, cam["name"], f)
    report = {"pending_no_verdict": 0, "copied": [], "trashed": [], "bad": [], "errors": []}
    for p, (folder, cam, f) in listed.items():
        if state.get(p, {}).get("done"):
            continue
        v = verdicts.get(p)
        local = CLIPS / f"{folder}__{cam}__{f['name']}"
        try:
            if v in ("0", "1"):
                if not local.exists():
                    ydownload(token, p, local)
                if not a.apply:
                    report["copied"].append((p, v, "dry-run"))
                    continue
                dest, ok = copy_verified(local, DATA_DIR / v)
                if not ok:
                    report["errors"].append((p, "md5 копии не совпал, исходник не тронут"))
                    continue
                ytrash(token, p)
                state[p] = {"done": True, "class": v, "dest": str(dest)}
                report["copied"].append((p, v, str(dest)))
            elif v is None:
                report["pending_no_verdict"] += 1
                if a.trash_bad:
                    if not local.exists():
                        ydownload(token, p, local)
                    ok, why = check_clip(local, dog)
                    if not ok:
                        report["bad"].append((p, why))
                        if a.apply:
                            ytrash(token, p)
                            state[p] = {"done": True, "trashed": why}
                            report["trashed"].append(p)
        except (urllib.error.URLError, OSError, Exception) as e:  # noqa: BLE001
            report["errors"].append((p, str(e)[:120]))
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORK / f"report_{time.strftime('%Y%m%d_%H%M')}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"на Яндексе {len(listed)}; без вердикта {report['pending_no_verdict']}; "
          f"{'скопировано' if a.apply else 'к копированию (dry-run)'} {len(report['copied'])}; "
          f"негодных {len(report['bad'])}, в корзину {len(report['trashed'])}, ошибок {len(report['errors'])}")
    for p, why in report["errors"][:5]:
        print("  ошибка:", p.split("/")[-1], "-", why)


if __name__ == "__main__":
    sys.exit(main())
