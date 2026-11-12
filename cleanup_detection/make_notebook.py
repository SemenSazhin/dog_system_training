# -*- coding: utf-8 -*-
"""Build HumanCleanupDetector.ipynb"""
import json
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).with_name("HumanCleanupDetector.ipynb")


def md(s: str):
    lines = dedent(s).strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(s: str):
    s = dedent(s).strip("\n")
    lines = s.split("\n")
    src = [l + ("\n" if i < len(lines) - 1 else "") for i, l in enumerate(lines)]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = []

cells.append(
    md(
        """
        # Human Cleanup Detector

        Пайплайн как у детекции дефекации собак, но для **поз человека**:
        определяет, **убрал ли человек за собакой**.

        ## Архитектуры для сравнения
        - Recurrent: `LSTM`, `BiLSTM`, `GRU`, `BiGRU`
        - SSM: `MambaLite` (pure PyTorch)
        - **Transformers**: `TransformerEncoder` (CLS-token) и `TemporalTransformer` (PoseFormer-lite)

        ## Также
        - Pretrained human pose: YOLO pose COCO-17 (`yolov8n/s-pose`, `yolo11n-pose`)
        - Трекеры: IoU / DeepSORT
        - Метрики, сохранение в `models/`, инференс на видео

        > Точного публичного видеодатасета «уборка за собакой» почти нет.
        > Proxies: Stipra, Charades throw/put/tidy, ShitSpotter (фото) + synthetic + свои ролики.
        """
    )
)

cells.append(
    code(
        r"""
        # @title Установка зависимостей
        !pip install -q ultralytics opencv-python-headless tqdm scikit-learn matplotlib seaborn pandas deep_sort_realtime huggingface_hub requests

        import os, sys, json, math, time, random, zipfile, shutil
        from pathlib import Path
        from collections import deque
        from dataclasses import dataclass, asdict
        from datetime import datetime

        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report, confusion_matrix, roc_auc_score
        )
        import pandas as pd
        from tqdm.auto import tqdm

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('DEVICE:', DEVICE)
        """
    )
)

cells.append(
    code(
        r"""
        # @title Config
        @dataclass
        class Config:
            pose_models: tuple = ('yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolo11n-pose.pt')
            num_keypoints: int = 17
            window_size: int = 48
            person_conf: float = 0.4
            batch_size: int = 32
            epochs: int = 12
            lr: float = 1e-3
            weight_decay: float = 1e-4
            seed: int = 42
            repo_root: str = ''
            data_root: str = ''
            models_root: str = ''
            videos_pos: str = ''
            videos_neg: str = ''

        cfg = Config()
        if Path('/content').exists():
            cfg.repo_root = '/content/dogsproject'
        else:
            here = Path.cwd()
            cfg.repo_root = str(here.parent if here.name == 'cleanup_detection' else here)

        cfg.data_root = str(Path(cfg.repo_root) / 'datasets')
        cfg.models_root = str(Path(cfg.repo_root) / 'cleanup_detection' / 'models')
        cfg.videos_pos = str(Path(cfg.data_root) / 'labeled_videos' / 'cleanup')
        cfg.videos_neg = str(Path(cfg.data_root) / 'labeled_videos' / 'no_cleanup')
        for p in [cfg.models_root, cfg.videos_pos, cfg.videos_neg]:
            Path(p).mkdir(parents=True, exist_ok=True)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
        """
    )
)

cells.append(
    md(
        """
        ## Датасеты
        | Источник | Тип | Зачем |
        |---|---|---|
        | Stipra (~22GB, 1273 видео) | выброс мусора | лучший open proxy cleanup-движения |
        | Charades | throw/put/tidy | proxy наклона/выброса |
        | ShitSpotter | фото before/after | домен, не action |
        | Свои CCTV | cleanup / no_cleanup | целевой датасет |
        """
    )
)

cells.append(
    code(
        r"""
        # @title Load / bootstrap data
        DATA = Path(cfg.data_root)
        PROXY = DATA / 'cleanup_proxy' / 'synthetic_pose_sequences.npz'
        MANIFEST = DATA / 'manifests' / 'DATASETS_CATALOG.json'

        if Path('/content').exists() and not (DATA / 'manifests').exists():
            !git clone --depth 1 https://github.com/SemenSazhin/dog_system_training.git /content/dogsproject
            cfg.repo_root = '/content/dogsproject'
            cfg.data_root = '/content/dogsproject/datasets'
            cfg.models_root = '/content/dogsproject/cleanup_detection/models'
            DATA = Path(cfg.data_root)
            PROXY = DATA / 'cleanup_proxy' / 'synthetic_pose_sequences.npz'
            MANIFEST = DATA / 'manifests' / 'DATASETS_CATALOG.json'

        sys.path.insert(0, str(Path(cfg.repo_root)))
        try:
            from datasets.download_datasets import generate_synthetic_pose_sequences, write_dataset_catalog
            if not PROXY.exists():
                generate_synthetic_pose_sequences()
            write_dataset_catalog()
        except Exception as e:
            print('helper:', e)

        if MANIFEST.exists():
            cat = json.loads(MANIFEST.read_text(encoding='utf-8'))
            print('pretrained pose:', cat.get('pretrained_human_pose'))
            print('datasets:', [d['id'] for d in cat.get('datasets', [])])

        assert PROXY.exists(), PROXY
        pack = np.load(PROXY)
        X_syn, y_syn = pack['X'], pack['y']
        print('synthetic', X_syn.shape, 'pos', int((y_syn == 1).sum()), 'neg', int((y_syn == 0).sum()))
        """
    )
)

cells.append(
    md(
        """
        ## Features + модели (включая Transformer)
        COCO-17 → углы + относительные координаты + Δ.
        Особо важны для cleanup: сгиб коленей, наклон корпуса, высота запястий.
        """
    )
)

cells.append(
    code(
        r'''
        # @title Features + architectures (LSTM/GRU/Mamba/Transformer)
        L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
        L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

        KEY_ANGLES = [
            (L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI),
            (L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK),
            (L_SHO, L_HIP, L_KNE), (R_SHO, R_HIP, R_KNE),
            (L_ELB, L_SHO, L_HIP), (R_ELB, R_SHO, R_HIP),
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
            Fmat = np.stack([pose_frame_features(k) for k in kpts_seq], axis=0)
            dlt = np.zeros_like(Fmat)
            dlt[1:] = Fmat[1:] - Fmat[:-1]
            return np.concatenate([Fmat, dlt], axis=1)


        class LSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False, num_classes=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
                od = hidden_size * (2 if bidirectional else 1)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(od, od // 2), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(od // 2, num_classes))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1])


        class GRUClassifier(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False, num_classes=2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
                od = hidden_size * (2 if bidirectional else 1)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(od, od // 2), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(od // 2, num_classes))

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1])


        class _MambaBlock(nn.Module):
            def __init__(self, d_model, d_state=16, expand=2):
                super().__init__()
                d_inner = d_model * expand
                self.in_proj = nn.Linear(d_model, d_inner * 2)
                self.conv = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
                self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner)
                self.dt_proj = nn.Linear(d_inner, d_inner)
                self.out_proj = nn.Linear(d_inner, d_model)
                self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
                self.D = nn.Parameter(torch.ones(d_inner))
                self.d_state, self.d_inner = d_state, d_inner

            def forward(self, x):
                xz = self.in_proj(x)
                x_b, z = xz.chunk(2, dim=-1)
                x_b = F.silu(self.conv(x_b.transpose(1, 2)).transpose(1, 2))
                dbl = self.x_proj(x_b)
                delta, B, C = torch.split(dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
                delta = F.softplus(self.dt_proj(delta))
                A = -torch.exp(self.A_log)
                ys = []
                h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
                for t in range(x.size(1)):
                    dt = delta[:, t].unsqueeze(-1)
                    a_bar = torch.exp(dt * A)
                    b_bar = dt * B[:, t].unsqueeze(1)
                    h = a_bar * h + b_bar * x_b[:, t].unsqueeze(-1)
                    yt = torch.einsum('bdn,bn->bd', h, C[:, t]) + self.D * x_b[:, t]
                    ys.append(yt)
                y = torch.stack(ys, 1) * F.silu(z)
                return self.out_proj(y)


        class MambaLiteClassifier(nn.Module):
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, d_state=16):
                super().__init__()
                self.input_proj = nn.Linear(input_size, hidden_size)
                self.blocks = nn.ModuleList([_MambaBlock(hidden_size, d_state=d_state) for _ in range(num_layers)])
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_size // 2, num_classes))

            def forward(self, x):
                h = self.input_proj(x)
                for b in self.blocks:
                    h = h + b(self.norm(h))
                return self.head(h.mean(1))


        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=512, dropout=0.1):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

            def forward(self, x):
                return self.dropout(x + self.pe[:, :x.size(1)])


        class TransformerEncoderClassifier(nn.Module):
            # Transformer encoder + CLS token
            def __init__(self, input_size, hidden_size=128, num_layers=2, nhead=4, dropout=0.3, num_classes=2, max_len=512):
                super().__init__()
                if hidden_size % nhead:
                    hidden_size = nhead * max(1, hidden_size // nhead)
                self.input_proj = nn.Linear(input_size, hidden_size)
                self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))
                nn.init.normal_(self.cls, std=0.02)
                self.pos = PositionalEncoding(hidden_size, max_len + 1, dropout)
                layer = nn.TransformerEncoderLayer(
                    hidden_size, nhead, hidden_size * 4, dropout, 'gelu',
                    batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers)
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden_size // 2, num_classes),
                )

            def forward(self, x):
                h = self.input_proj(x)
                h = torch.cat([self.cls.expand(x.size(0), -1, -1), h], 1)
                h = self.encoder(self.pos(h))
                return self.head(self.norm(h[:, 0]))


        class TemporalTransformerClassifier(nn.Module):
            # PoseFormer-lite: temporal Transformer + mean/max pooling
            def __init__(self, input_size, hidden_size=128, num_layers=3, nhead=4, dropout=0.3, num_classes=2, max_len=512):
                super().__init__()
                if hidden_size % nhead:
                    hidden_size = nhead * max(1, hidden_size // nhead)
                self.input_proj = nn.Sequential(
                    nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(dropout)
                )
                self.pos = PositionalEncoding(hidden_size, max_len, dropout)
                layer = nn.TransformerEncoderLayer(
                    hidden_size, nhead, hidden_size * 4, dropout, 'gelu',
                    batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers)
                self.norm = nn.LayerNorm(hidden_size)
                self.head = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden_size, num_classes),
                )

            def forward(self, x):
                h = self.norm(self.encoder(self.pos(self.input_proj(x))))
                return self.head(torch.cat([h.mean(1), h.max(1).values], -1))


        def build_model(name, input_size, **kw):
            name = name.lower()
            common = {k: kw[k] for k in ('hidden_size', 'num_layers', 'dropout', 'num_classes') if k in kw}
            if name == 'lstm':
                return LSTMClassifier(input_size, bidirectional=False, **common)
            if name == 'bilstm':
                return LSTMClassifier(input_size, bidirectional=True, **common)
            if name == 'gru':
                return GRUClassifier(input_size, bidirectional=False, **common)
            if name == 'bigru':
                return GRUClassifier(input_size, bidirectional=True, **common)
            if name == 'mamba_lite':
                return MambaLiteClassifier(input_size, **common)
            if name == 'transformer':
                tkw = dict(common)
                if 'nhead' in kw:
                    tkw['nhead'] = kw['nhead']
                return TransformerEncoderClassifier(input_size, **tkw)
            if name == 'temporal_transformer':
                tkw = dict(common)
                if 'nhead' in kw:
                    tkw['nhead'] = kw['nhead']
                return TemporalTransformerClassifier(input_size, **tkw)
            raise KeyError(name)


        print('Architectures ready: lstm, bilstm, gru, bigru, mamba_lite, transformer, temporal_transformer')
        '''
    )
)

cells.append(
    md(
        """
        ## Обучение и сравнение архитектур
        Включая **transformer** и **temporal_transformer**.
        """
    )
)

cells.append(
    code(
        r"""
        # @title Prepare splits + train/eval
        class SeqDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.long)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, i):
                return self.X[i], self.y[i]


        def prepare_xy(X, y):
            # synthetic stores flattened COCO-17 xyz => reshape then featureize
            T = cfg.window_size
            if X.shape[-1] == 17 * 3:
                k = X.reshape(len(X), T, 17, 3)
                Xs = np.stack([sequence_features(ki) for ki in k]).astype(np.float32)
            else:
                Xs = X.astype(np.float32)
            return Xs, y.astype(np.int64)


        X_all, y_all = prepare_xy(X_syn, y_syn)
        INPUT_SIZE = X_all.shape[-1]
        print('X_all', X_all.shape)

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=cfg.seed, stratify=y_all
        )
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tmp, y_tmp, test_size=0.2, random_state=cfg.seed, stratify=y_tmp
        )

        train_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=cfg.batch_size)
        test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=cfg.batch_size)


        def evaluate(model, loader):
            model.eval()
            preds, trues, probs = [], [], []
            with torch.no_grad():
                for xb, yb in loader:
                    logits = model(xb.to(DEVICE))
                    p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                    pr = logits.argmax(1).cpu().numpy()
                    preds.extend(pr.tolist())
                    trues.extend(yb.numpy().tolist())
                    probs.extend(p.tolist())
            acc = accuracy_score(trues, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average='binary', zero_division=0)
            try:
                auc = roc_auc_score(trues, probs)
            except Exception:
                auc = float('nan')
            return {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
                'report': classification_report(trues, preds, target_names=['no_cleanup', 'cleanup'], zero_division=0),
                'cm': confusion_matrix(trues, preds).tolist(),
                'samples': [
                    {'i': i, 'true': int(trues[i]), 'pred': int(preds[i]), 'prob_cleanup': round(float(probs[i]), 4)}
                    for i in range(min(3, len(trues)))
                ],
            }


        def train_one(tag, arch, params, epochs=None):
            epochs = epochs or cfg.epochs
            model_kwargs = {k: v for k, v in params.items() if k not in ('lr', 'weight_decay')}
            model = build_model(arch, INPUT_SIZE, **model_kwargs).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=params.get('lr', cfg.lr),
                weight_decay=params.get('weight_decay', cfg.weight_decay),
            )
            crit = nn.CrossEntropyLoss()
            best_f1, best_state = -1.0, None
            for ep in range(epochs):
                model.train()
                losses = []
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = crit(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(loss.item())
                val = evaluate(model, val_loader)
                print(f"[{tag}] ep {ep+1}/{epochs} loss={np.mean(losses):.4f} val_f1={val['f1']:.4f}")
                if val['f1'] >= best_f1:
                    best_f1 = val['f1']
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            test = evaluate(model, test_loader)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            exp_dir = Path(cfg.models_root) / tag
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': arch,
                'params': params,
                'input_size': INPUT_SIZE,
                'metrics': {k: test[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cm')},
            }, exp_dir / 'model.pth')
            md_lines = [
                f'# {tag}', '',
                f'- arch: {arch}',
                f"- accuracy: {test['accuracy']:.4f}",
                f"- precision: {test['precision']:.4f}",
                f"- recall: {test['recall']:.4f}",
                f"- f1: {test['f1']:.4f}",
                f"- roc_auc: {test['roc_auc']:.4f}",
                f'- params: {n_params}', '',
                '## Hyperparams',
            ] + [f'- {k}: {v}' for k, v in params.items()] + [
                '', '## Confusion matrix', str(test['cm']), '',
                '## Report', '```', test['report'], '```', '',
                '## 3 samples',
            ] + [f'- {s}' for s in test['samples']]
            (exp_dir / 'metrics.md').write_text('\n'.join(md_lines), encoding='utf-8')
            return {
                'tag': tag, 'arch': arch, 'n_params': n_params, 'dir': str(exp_dir),
                'accuracy': test['accuracy'], 'precision': test['precision'],
                'recall': test['recall'], 'f1': test['f1'], 'roc_auc': test['roc_auc'],
                'lr': params.get('lr', cfg.lr),
            }


        EXPERIMENTS = [
            ('lstm_base', 'lstm', dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
            ('bilstm_base', 'bilstm', dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
            ('gru_base', 'gru', dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
            ('bigru_base', 'bigru', dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=8e-4)),
            ('mamba_lite_base', 'mamba_lite', dict(hidden_size=128, num_layers=2, dropout=0.3, num_classes=2, lr=1e-3)),
            # Transformers
            ('transformer_cls', 'transformer', dict(hidden_size=128, num_layers=2, nhead=4, dropout=0.3, num_classes=2, lr=8e-4)),
            ('transformer_wide', 'transformer', dict(hidden_size=192, num_layers=3, nhead=6, dropout=0.35, num_classes=2, lr=5e-4)),
            ('temporal_transformer', 'temporal_transformer', dict(hidden_size=128, num_layers=3, nhead=4, dropout=0.3, num_classes=2, lr=7e-4)),
        ]

        results = []
        for tag, arch, params in EXPERIMENTS:
            print('\n' + '=' * 70)
            print('RUN', tag)
            results.append(train_one(tag, arch, params))

        df = pd.DataFrame(results).sort_values('f1', ascending=False)
        print('\n=== ARCHITECTURE COMPARISON (incl. Transformers) ===')
        print(df.to_string(index=False))
        best = max(results, key=lambda r: r['f1'])
        print('BEST:', best['tag'], 'F1=', round(best['f1'], 4))
        df.to_csv(Path(cfg.models_root) / 'architecture_comparison.csv', index=False)
        (Path(cfg.models_root) / 'summary.md').write_text(
            '# Architecture comparison\n\n' + df.to_markdown(index=False) +
            f"\n\nBest: **{best['tag']}** ({best['arch']}), F1={best['f1']:.4f}\n",
            encoding='utf-8',
        )
        """
    )
)

cells.append(
    md(
        """
        ## Pose backends + трекеры
        Комбинации YOLO-pose весов и трекеров (IoU / DeepSORT) на видео.
        """
    )
)

cells.append(
    code(
        r"""
        # @title Pose/tracker combo smoke-test
        from ultralytics import YOLO

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            HAS_DEEPSORT = True
        except Exception:
            HAS_DEEPSORT = False
            print('DeepSORT unavailable')


        class IoUTracker:
            def __init__(self, iou_thr=0.3, max_age=20):
                self.iou_thr, self.max_age = iou_thr, max_age
                self.tracks, self._next = {}, 1

            @staticmethod
            def iou(a, b):
                x1, y1 = max(a[0], b[0]), max(a[1], b[1])
                x2, y2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6
                return inter / ua

            def update(self, boxes):
                assigned, alive = set(), {}
                for tid, tr in self.tracks.items():
                    best, bi = 0, None
                    for i, b in enumerate(boxes):
                        if i in assigned:
                            continue
                        v = self.iou(tr['box'], b)
                        if v > best:
                            best, bi = v, i
                    if bi is not None and best >= self.iou_thr:
                        alive[tid] = {'box': boxes[bi], 'age': 0}
                        assigned.add(bi)
                    elif tr['age'] + 1 <= self.max_age:
                        alive[tid] = {'box': tr['box'], 'age': tr['age'] + 1}
                for i, b in enumerate(boxes):
                    if i not in assigned:
                        alive[self._next] = {'box': b, 'age': 0}
                        self._next += 1
                self.tracks = alive
                return [(tid, t['box']) for tid, t in self.tracks.items() if t['age'] == 0]


        def extract_person_pose_sequence(video_path, pose_weights='yolov8n-pose.pt', tracker='iou', max_frames=240):
            pose = YOLO(pose_weights)
            track = IoUTracker()
            ds = DeepSort(max_age=30) if (tracker == 'deepsort' and HAS_DEEPSORT) else None
            cap = cv2.VideoCapture(str(video_path))
            seqs = {}
            frames = 0
            while frames < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                res = pose(frame, verbose=False)[0]
                boxes, kps = [], []
                if res.boxes is not None and len(res.boxes):
                    for i in range(len(res.boxes)):
                        if float(res.boxes.conf[i]) < cfg.person_conf:
                            continue
                        boxes.append(res.boxes.xyxy[i].cpu().numpy().tolist())
                        kps.append(res.keypoints.data[i].cpu().numpy() if res.keypoints is not None else np.zeros((17, 3), np.float32))
                if ds is not None:
                    dets = [([b[0], b[1], b[2] - b[0], b[3] - b[1]], 0.9, 'person') for b in boxes]
                    for tr in ds.update_tracks(dets, frame=frame):
                        if not tr.is_confirmed() or not boxes:
                            continue
                        l, t, r, btm = tr.to_ltrb()
                        i = int(np.argmin([abs((b[0] + b[2]) / 2 - (l + r) / 2) for b in boxes]))
                        seqs.setdefault(tr.track_id, []).append(kps[i])
                else:
                    for tid, box in track.update(boxes):
                        if not boxes:
                            continue
                        i = int(np.argmin([abs(sum(np.array(box) - np.array(bb))) for bb in boxes]))
                        seqs.setdefault(tid, []).append(kps[i])
            cap.release()
            if not seqs:
                return None
            tid = max(seqs, key=lambda k: len(seqs[k]))
            arr = np.stack(seqs[tid], 0)
            if len(arr) < cfg.window_size:
                pad = np.repeat(arr[-1:], cfg.window_size - len(arr), 0)
                arr = np.concatenate([arr, pad], 0)
            else:
                arr = arr[-cfg.window_size:]
            return sequence_features(arr)


        video_candidates = list(Path(cfg.videos_pos).glob('*.mp4')) + list(Path(cfg.videos_neg).glob('*.mp4'))
        combo_rows = []
        if video_candidates:
            v = video_candidates[0]
            for pose_w in cfg.pose_models:
                for tr in (['iou', 'deepsort'] if HAS_DEEPSORT else ['iou']):
                    t0 = time.time()
                    try:
                        feat = extract_person_pose_sequence(v, pose_w, tr, max_frames=90)
                        ok, shape = feat is not None, (None if feat is None else list(feat.shape))
                    except Exception as e:
                        ok, shape = False, str(e)
                    row = {'pose': pose_w, 'tracker': tr, 'ok': ok, 'feat_shape': shape, 'sec': round(time.time() - t0, 2)}
                    combo_rows.append(row)
                    print(row)
            pd.DataFrame(combo_rows).to_csv(Path(cfg.models_root) / 'pose_tracker_combos.csv', index=False)
        else:
            print('Нет mp4 в labeled_videos. Добавьте ролики в:')
            print(cfg.videos_pos)
            print(cfg.videos_neg)
            print('Pretrained pose: yolov8n-pose.pt / yolov8s-pose.pt / yolo11n-pose.pt')
        """
    )
)

cells.append(
    md(
        """
        ## Инференс всех архитектур на видео
        """
    )
)

cells.append(
    code(
        r"""
        # @title Video inference (all architectures)
        def resolve_arch(name: str) -> str:
            for a in ['temporal_transformer', 'transformer', 'mamba_lite', 'bilstm', 'bigru', 'lstm', 'gru']:
                if name.startswith(a) or a in name:
                    return a
            return name


        def load_sequence_model(row):
            ckpt_path = Path(row['dir']) / 'model.pth'
            # PyTorch 2.6+ defaults weights_only=True; our ckpt stores numpy scalars in metrics/params
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            arch = resolve_arch(ckpt.get('arch', row['arch']))
            params = {k: v for k, v in ckpt.get('params', {}).items() if k not in ('lr', 'weight_decay')}
            model = build_model(arch, ckpt['input_size'], **params).to(DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            return model, ckpt


        def load_best_sequence_model():
            return load_sequence_model(best)


        def extract_pose_stream(video_path, pose_weights='yolov8n-pose.pt'):
            """Run YOLO-pose once; yield (frame, boxes, kps) for all models."""
            pose = YOLO(pose_weights)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f'Cannot open video: {video_path}')
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w, h = int(cap.get(3)), int(cap.get(4))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = pose(frame, verbose=False)[0]
                boxes, kps = [], []
                if res.boxes is not None:
                    for i in range(len(res.boxes)):
                        if float(res.boxes.conf[i]) < cfg.person_conf:
                            continue
                        boxes.append(res.boxes.xyxy[i].cpu().numpy().tolist())
                        if res.keypoints is not None:
                            kps.append(res.keypoints.data[i].cpu().numpy())
                        else:
                            kps.append(np.zeros((17, 3), np.float32))
                frames.append((frame, boxes, kps))
            cap.release()
            return frames, fps, w, h


        def run_video_cleanup_detector(
            video_path,
            output_path,
            model=None,
            row=None,
            pose_weights='yolov8n-pose.pt',
            thr=0.5,
            pose_cache=None,
            tag='',
        ):
            if model is None:
                model, _ = load_sequence_model(row or best)
            if pose_cache is None:
                frames, fps, w, h = extract_pose_stream(video_path, pose_weights)
            else:
                frames, fps, w, h = pose_cache

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            track = IoUTracker()
            buffers = {}
            probs_hist = []
            n_cleanup = 0
            n_scored = 0

            for frame, boxes, kps in frames:
                vis = frame.copy()
                label, p = 'no_cleanup', 0.0
                for tid, box in track.update(boxes):
                    if not boxes:
                        continue
                    i = int(np.argmin([abs(sum(np.array(box) - np.array(bb))) for bb in boxes]))
                    buffers.setdefault(tid, deque(maxlen=cfg.window_size)).append(kps[i])
                    if len(buffers[tid]) == cfg.window_size:
                        feat = sequence_features(np.stack(buffers[tid], 0))
                        with torch.no_grad():
                            logits = model(torch.tensor(feat, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                            p = float(torch.softmax(logits, 1)[0, 1].cpu())
                        label = 'cleanup' if p >= thr else 'no_cleanup'
                        probs_hist.append(p)
                        n_scored += 1
                        n_cleanup += int(label == 'cleanup')
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 200, 0) if label == 'cleanup' else (0, 165, 255)
                    title = f'{tag} id:{tid} {label} {p:.2f}'.strip()
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, title, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                writer.write(vis)
            writer.release()

            mean_p = float(np.mean(probs_hist)) if probs_hist else 0.0
            cleanup_rate = float(n_cleanup / n_scored) if n_scored else 0.0
            print(f"saved {output_path} | mean_p={mean_p:.3f} cleanup_rate={cleanup_rate:.3f} scored={n_scored}")
            return {
                'output_path': str(output_path),
                'mean_p': mean_p,
                'cleanup_rate': cleanup_rate,
                'n_scored_windows': n_scored,
                'n_cleanup_windows': n_cleanup,
            }


        def run_video_all_architectures(
            video_path='/content/test.mp4',
            out_dir=None,
            pose_weights='yolov8n-pose.pt',
            thr=0.5,
            model_rows=None,
        ):
            model_rows = model_rows or results
            out_dir = Path(out_dir or (Path(cfg.models_root) / 'video_runs'))
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f'Extracting pose once from {video_path} ...')
            frames, fps, w, h = extract_pose_stream(video_path, pose_weights)
            pose_cache = (frames, fps, w, h)
            print(f'frames={len(frames)} fps={fps:.2f} size={w}x{h}; running {len(model_rows)} models')

            summary = []
            for row in model_rows:
                tag = row['tag']
                print(f'\\n=== {tag} ({row["arch"]}) ===')
                model, _ = load_sequence_model(row)
                out_path = out_dir / f'{tag}.mp4'
                stats = run_video_cleanup_detector(
                    video_path,
                    out_path,
                    model=model,
                    thr=thr,
                    pose_cache=pose_cache,
                    tag=tag,
                )
                summary.append({
                    'tag': tag,
                    'arch': row['arch'],
                    'train_f1': row.get('f1'),
                    'train_accuracy': row.get('accuracy'),
                    **stats,
                    'is_best': tag == best['tag'],
                })
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sdf = pd.DataFrame(summary).sort_values('mean_p', ascending=False)
            csv_path = out_dir / 'video_inference_summary.csv'
            sdf.to_csv(csv_path, index=False)
            print('\\n=== VIDEO INFERENCE ALL ARCHITECTURES ===')
            print(sdf.to_string(index=False))
            print('summary:', csv_path)
            print('best-by-train still:', best['tag'])
            return sdf


        print('Models for video test:', [r['tag'] for r in results])
        print('Best-by-train F1:', best['tag'], best['arch'])
        # Uncomment / set path:
        # video_summary = run_video_all_architectures('/content/test.mp4')
        # # single model still available:
        # run_video_cleanup_detector('/content/test.mp4', '/content/cleanup_out_best.mp4', row=best)
        """
    )
)

cells.append(
    md(
        """
        ## Выгрузка моделей и метрик
        """
    )
)

cells.append(
    code(
        r"""
        # @title Export zip
        try:
            from google.colab import files
        except Exception:
            files = None

        out_zip = Path('/content') / f"cleanup_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        if not Path('/content').exists():
            out_zip = Path(cfg.models_root).parent / out_zip.name

        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in Path(cfg.models_root).rglob('*'):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(Path(cfg.models_root).parent)))
        print('zip:', out_zip)
        if files is not None:
            files.download(str(out_zip))
        """
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
