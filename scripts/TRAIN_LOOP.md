# Цикл: датасет → переобучение → сравнение → video-scan

Реестр лежит в `D:\dog-ml\registry\`:
- `snapshots/` — срезы датасета
- `dataset_ledger.jsonl` — краткая история объёма данных
- `experiments.jsonl` — метрики прогонов
- `production.json` — active / candidate модель для video-scan

## 0) Сегодняшняя выгрузка уже на G:

`G:\Мой диск\Система контроля за собаками\data\`
- `0/` — нет дефекации
- `1/` — есть дефекация

Перед обучением гляньте подозрительные имена (`nodf_` в классе `1` и наоборот).

## 1) Снимок датасета

```powershell
D:\dog-ml\venv\Scripts\python.exe `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\snapshot_dataset.py" `
  --note "после выгрузки 01.11"
```

## 2) Извлечь скелеты (только новые ролики)

```powershell
D:\dog-ml\venv\Scripts\python.exe `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\extract_raw.py" `
  "G:\Мой диск\Система контроля за собаками\data" `
  "C:\Users\Zhenya\Desktop\Детекция Собак\video-scan\models\yolo26s.pt" `
  "C:\Users\Zhenya\Desktop\Детекция Собак\video-scan\models\dog_pose_model_yolo8_14.pt" `
  "D:\dog-ml\raw"
```

Скрипт пропускает уже существующие `.npz`.

## 3) Честный CV-прогон LSTM (база для сравнения)

Лучшая простая конфигурация из отчёта: короткое окно, без лишних усложнений.

```powershell
D:\dog-ml\venv\Scripts\python.exe -u `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\skeleton_cv.py" `
  --raw D:\dog-ml\raw `
  --out D:\dog-ml\cv `
  --tag retrain_YYYYMMDD `
  --models lstm `
  --seeds 5 `
  --no-asp
```

## 4) Записать эксперимент в реестр

```powershell
D:\dog-ml\venv\Scripts\python.exe `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\register_experiment.py" `
  --results D:\dog-ml\cv\results_retrain_YYYYMMDD.json `
  --model lstm `
  --dataset-snapshot dataset_....json `
  --note "retrain после выгрузки" `
  --set-candidate
```

Если есть готовые веса production-LSTM (из Colab fine-tune) — добавьте `--weights path\to\best_lstm_model.pth`.

## 5) Решение: выкатывать ли в video-scan

```powershell
D:\dog-ml\venv\Scripts\python.exe `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\promote_model.py" --dry-run

# если лучше:
D:\dog-ml\venv\Scripts\python.exe `
  "C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\scripts\promote_model.py" `
  --weights D:\dog-ml\registry\models\<id>.pth --apply
```

Потом в `video-scan`:

```powershell
docker cp ".\models\best_lstm_model.pth" video-scan:/app/models/best_lstm_model.pth
docker restart video-scan
```

## Как читать «лучше»

Минимум для выката:
- `auc` не хуже baseline более чем на 0.005
- `cam_fpr_05` не хуже более чем на +0.02

На маленьком датасете колебания ±0.02 AUC — шум; смотрите ещё `auc_other` и ложные на камерах.

## Важно про две модели

1. **CV-модель в `skeleton_cv.py`** — для честного сравнения метрик (OOF по видео).
2. **Production `best_lstm_model.pth`** — обычно учится/дообучается в Colab (`new/PoopDetector.ipynb`, fine-tune) на том же датасете и уже потом кладётся в `video-scan`.

Реестр связывает оба шага: CV говорит «данные/протокол лучше», Colab-веса — то, что реально крутится на камерах.
