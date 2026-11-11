# Human Cleanup Detector

Отдельный пайплайн для задачи: **убрал ли человек за собакой** (по позе человека на видео).

## Ноутбук
- [`HumanCleanupDetector.ipynb`](HumanCleanupDetector.ipynb) — обучение, сравнение архитектур, метрики, инференс

## Архитектуры
| Семья | Модели |
|---|---|
| Recurrent | LSTM, BiLSTM, GRU, BiGRU |
| SSM | MambaLite (pure PyTorch) |
| **Transformer** | `TransformerEncoder` (CLS), `TemporalTransformer` (PoseFormer-lite) |

Модуль с классами: [`architectures.py`](architectures.py)

## Pretrained human pose
- `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolo11n-pose.pt` (COCO-17)
- Опционально MediaPipe Pose (33 landmarks)

## Датасеты
См. [`../datasets/README.md`](../datasets/README.md) и `../datasets/manifests/DATASETS_CATALOG.json`.

Точного open-датасета «человек убирает за собакой» почти нет. Используются proxies + свои labeled videos в:
- `datasets/labeled_videos/cleanup/`
- `datasets/labeled_videos/no_cleanup/`
