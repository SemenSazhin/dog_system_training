# ShitSpotter sample

Источник: https://huggingface.co/datasets/erotemic/shitspotter

Это **фото** dog feces before/after (не видео действий человека).
Полезно как доменный prior / detection, но **не** как основной датасет для pose-последовательности cleanup.

Полные webdataset shards большие (~30GB+). Метаданные категорий уже скачаны скриптом `datasets/download_datasets.py`.

Чтобы вытащить небольшой image-sample:

```bash
py -3 datasets/download_datasets.py --shitspotter
```

(скачает один train shard и извлечёт до 40 jpg в `sample_images/`)
