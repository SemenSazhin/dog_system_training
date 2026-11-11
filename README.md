# Ноутбуки в Google Colab

Файл генерируется автоматически (`scripts/colab_sync.py`), вручную не править.
Ноутбуки из репозитория синхронизируются с Google Drive в обе стороны: пуш обновляет Colab, правки из Colab подтягиваются обратно в репозиторий.

| Ноутбук | Colab | GitHub |
|---|---|---|
| `new/PoopDetector.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1quFz_T_M6EZG9kWFrGbUSZ6RfPhkp7Tq) | [файл](https://github.com/SemenSazhin/dog_system_training/blob/master/new/PoopDetector.ipynb) |
| `old/PoopDetector.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A62n8eOV7ZUIaP7cRsfSmIPFNopE2_L7) | [файл](https://github.com/SemenSazhin/dog_system_training/blob/master/old/PoopDetector.ipynb) |
| `cleanup_detection/HumanCleanupDetector.ipynb` | — | [файл](https://github.com/SemenSazhin/dog_system_training/blob/master/cleanup_detection/HumanCleanupDetector.ipynb) |

## Human cleanup (убрал ли человек за собакой)

Отдельный пайплайн в `cleanup_detection/`:
- human pose (YOLO pose COCO-17)
- сравнение `LSTM / GRU / MambaLite / Transformer / TemporalTransformer`
- датасеты и proxies в `datasets/`
- инструкция: `cleanup_detection/README.md`
