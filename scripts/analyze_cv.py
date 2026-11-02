#!/usr/bin/env python3
"""Сводка результатов очереди skeleton_cv: агрегаты, графики, разбор ошибок по роликам.

python analyze_cv.py [cv_dir] [out_dir]  ->  out_dir/stats.json, out_dir/*.png, out_dir/errors.md
"""
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CV = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\dog-ml\cv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\Zhenya\Desktop\Детекция Собак\Обучение\123141ewq1\presentation\cv")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, ORANGE, GREY, GREEN, RED, PURPLE = "#1F4E79", "#E07B39", "#9AA5B1", "#4C9F70", "#B5483A", "#7D6BAA"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
METRICS = ["auc", "ap", "f1_best", "auc_other", "cam_fpr_05", "win_auc", "auc_k1", "auc_k5", "auc_kmean"]
GROUP_NAMES = {"spine": "позвоночник", "hips": "таз", "tail": "хвост", "front": "передние лапы", "rear": "задние лапы"}
JOINT_GROUP = ["spine"] * 6 + ["hips"] * 4 + ["tail"] * 3 + ["front"] * 3 + ["rear"] * 4

R = {}  # tag -> {model -> {metric: (mean, std, n)}}
META = {}
for f in sorted(CV.glob("results_*.json")):
    tag = f.stem[len("results_"):]
    if tag == "smoke":
        continue
    d = json.load(open(f, encoding="utf-8"))
    META[tag] = dict(cfg=d["cfg"], n_windows=d["n_windows"], n_videos=d["n_videos"])
    R[tag] = {}
    for model, runs in d["results"].items():
        agg = {}
        for k in METRICS:
            v = [r[k] for r in runs if r.get(k) is not None]
            if v:
                agg[k] = (float(np.mean(v)), float(np.std(v)), len(v))
        R[tag][model] = agg



from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef, precision_recall_curve,
                             roc_curve)

EXTRA = ["prec05", "rec05", "spec05", "f1_05", "bal_acc05", "mcc05", "rec_fpr05", "rec_fpr10", "prec_rec80",
         "brier", "ece"]
CURVES = {}  # (tag, model) -> (fpr_grid, tpr_mean, rec_grid, prec_mean)


def op_metrics(y, sc):
    pred = sc >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    fpr, tpr, _ = roc_curve(y, sc)
    pr, rc, _ = precision_recall_curve(y, sc)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (sc >= a) & (sc < b if b < 1 else sc <= b)
        if m.any():
            ece += m.mean() * abs(sc[m].mean() - y[m].mean())
    return dict(prec05=prec, rec05=rec, spec05=spec, f1_05=2 * prec * rec / max(prec + rec, 1e-9),
                bal_acc05=(rec + spec) / 2, mcc05=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
                rec_fpr05=float(tpr[fpr <= 0.05].max()), rec_fpr10=float(tpr[fpr <= 0.10].max()),
                prec_rec80=float(pr[rc >= 0.8].max()) if (rc >= 0.8).any() else 0.0,
                brier=brier_score_loss(y, sc), ece=ece), (fpr, tpr, rc, pr)


FG = np.linspace(0, 1, 101)
for tag in list(R):
    f = CV / f"scores_{tag}.npz"
    if not f.exists():
        continue
    z = np.load(f, allow_pickle=True)
    cls_all = z["vid_cls"]
    for model in list(R[tag]):
        if f"{model}_ids" not in z.files:
            continue
        ids = z[f"{model}_ids"]
        y = cls_all[ids]
        per, tprs, precs = [], [], []
        for key in z.files:
            if key.startswith(f"{model}_s"):
                m, (fpr, tpr, rc, pr) = op_metrics(y, z[key])
                per.append(m)
                tprs.append(np.interp(FG, fpr, tpr))
                order = np.argsort(rc)
                precs.append(np.interp(FG, rc[order], pr[order]))
        if per:
            for k in EXTRA:
                v = [p[k] for p in per]
                R[tag][model][k] = (float(np.mean(v)), float(np.std(v)), len(v))
            CURVES[(tag, model)] = (FG, np.mean(tprs, 0), FG, np.mean(precs, 0))

def g(tag, model, metric="auc"):
    try:
        return R[tag][model][metric][0]
    except KeyError:
        return None


def gs(tag, model, metric="auc"):
    try:
        return R[tag][model][metric][1]
    except KeyError:
        return None


stats = {t: {m: {k: dict(mean=v[0], std=v[1], n=v[2]) for k, v in a.items()} for m, a in ms.items()}
         for t, ms in R.items()}
json.dump(dict(stats=stats, meta=META), open(OUT / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("tags:", len(R), sorted(R))

META_AUC = g("zoo_gbm", "meta") or g("zoo_gbm_noasp", "meta")
ASP_AUC = g("zoo_gbm", "aspect")


def hline(ax, y, color, label):
    if y is not None:
        ax.axhline(y, color=color, ls="--", lw=1.3)
        ax.text(ax.get_xlim()[1], y, f" {label} {y:.2f}", color=color, va="center", ha="left", fontsize=10)


def bars(ax, labels, means, stds, colors, fmt="{:.2f}"):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.62)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.012, fmt.format(m), ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x, labels)


# ------------------------------------------------------------------ 1. зоопарк моделей
def chart_zoo():
    rows = []
    for tag, suffix in (("zoo_nets", "+рамка"), ("zoo_nets_noasp", "только точки")):
        for m in ("lstm", "tcn", "stgcn", "transformer", "mamba"):
            if g(tag, m) is not None:
                rows.append((m, suffix, g(tag, m), gs(tag, m)))
    for tag, suffix in (("zoo_gbm", "+рамка"), ("zoo_gbm_noasp", "только точки")):
        if g(tag, "gbm") is not None:
            rows.append(("gbm", suffix, g(tag, "gbm"), gs(tag, "gbm")))
    if not rows:
        return
    names = ["gbm", "lstm", "tcn", "stgcn", "transformer", "mamba"]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    w = 0.36
    for j, (suffix, col) in enumerate((("+рамка", NAVY), ("только точки", ORANGE))):
        vals = [(r[2], r[3]) if (r := next((x for x in rows if x[0] == n and x[1] == suffix), None)) else (0, 0)
                for n in names]
        pos = np.arange(len(names)) + (j - 0.5) * w
        ax.bar(pos, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=3, color=col, label=suffix)
        for p, v in zip(pos, vals):
            if v[0]:
                ax.text(p, v[0] + v[1] + 0.01, f"{v[0]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(-0.6, len(names) - 0.4)
    hline(ax, META_AUC, RED, "только метаданные")
    hline(ax, ASP_AUC, GREY, "только форма рамки")
    ax.set_ylabel("ROC-AUC по видео (среднее ± std по сидам)")
    ax.set_title("Модели на честном протоколе (разбиение по источникам)", loc="left", fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "zoo.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 2. группы точек
def chart_groups():
    ms = [("lstm", PURPLE), ("tcn", NAVY), ("stgcn", ORANGE)]
    ref = {m: g("grp_all", m) for m, _ in ms}
    if all(v is None for v in ref.values()):
        return
    only = [("only_spine", "позвоночник"), ("only_hips", "таз"), ("only_tail", "хвост"), ("only_front", "перед. лапы"),
            ("only_rear", "задн. лапы"), ("spine_hips", "позв.+таз"), ("hips_tail", "таз+хвост"),
            ("spine_tail", "позв.+хвост"), ("legs", "все лапы"), ("tail_rear", "хвост+задн.л."),
            ("hips_tail_rear", "таз+хвост+задн.л.")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, (m, col) in zip(axes, ms):
        labs, vals, errs = ["все 20"], [ref[m] or 0], [gs("grp_all", m) or 0]
        for k, lab in only:
            if g(f"grp_{k}", m) is not None:
                labs.append(lab); vals.append(g(f"grp_{k}", m)); errs.append(gs(f"grp_{k}", m))
        bars(ax, labs, vals, errs, [col] * len(labs))
        ax.set_xticklabels(labs, rotation=40, ha="right")
        ax.set_ylim(0.4, 1.0)
        if META_AUC:
            ax.axhline(META_AUC, color=RED, ls="--", lw=1.2)
        ax.set_title(f"{m}: только выбранные точки", loc="left")
    axes[0].set_ylabel("ROC-AUC по видео")
    fig.tight_layout()
    fig.savefig(OUT / "groups_only.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [("drop_spine", "без позвоночника"), ("drop_hips", "без таза"), ("drop_tail", "без хвоста"),
            ("drop_front", "без перед. лап"), ("drop_rear", "без задн. лап")]
    w = 0.26
    for j, (m, col) in enumerate(ms):
        d = [((g(f"grp_{k}", m) or np.nan) - (ref[m] or np.nan)) for k, _ in keys]
        pos = np.arange(len(keys)) + (j - 1) * w
        ax.bar(pos, d, w, color=col, label=m)
        for p, v in zip(pos, d):
            if np.isfinite(v):
                ax.text(p, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xticks(np.arange(len(keys)), [l for _, l in keys])
    ax.set_ylabel("изменение ROC-AUC")
    ax.set_title("Что теряем, если убрать группу (относительно всех 20 точек)", loc="left")
    ax.set_ylim(-0.035, 0.04)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "groups_drop.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 3. окно
def chart_window():
    tags = [("win_T30", "T=30"), ("zoo_nets", "T=45"), ("win_T60", "T=60"), ("win_T90", "T=90")]
    tags2 = [("zoo_nets", "T=45, 0.9"), ("win_T45_mv70", "T=45, 0.7"), ("win_T45_mv50", "T=45, 0.5"),
             ("win_T30", "T=30, 0.9"), ("win_T30_mv70", "T=30, 0.7"), ("win_T30_mv50", "T=30, 0.5"),
             ("win_T90_mv70", "T=90, 0.7")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, tg, ttl in ((axes[0], tags, "Длина окна (при 15 кадр/с)"), (axes[1], tags2, "Порог доли кадров с позой")):
        have = [(l, t) for t, l in tg if t in R]
        w = 0.26
        for j, (m, col) in enumerate((("lstm", PURPLE), ("tcn", NAVY), ("stgcn", ORANGE))):
            vals = [g(t, m) or 0 for _, t in have]
            errs = [gs(t, m) or 0 for _, t in have]
            pos = np.arange(len(have)) + (j - 1) * w
            ax.bar(pos, vals, w, yerr=errs, capsize=2, color=col, label=m)
        ax.set_xticks(np.arange(len(have)), [f"{l}\n({META[t]['n_videos']} видео)" for l, t in have])
        ax.set_ylim(0.4, 1.0)
        ax.set_title(ttl, loc="left")
    axes[0].set_ylabel("ROC-AUC по видео")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "window.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 4. абляции признаков
def chart_ablation():
    items = [("grp_all", "базовый (без рамки)"), ("abl_no_conf", "без confidence"), ("abl_no_vel", "без скоростей"),
             ("abl_no_aug", "без аугментаций"), ("abl_minimal", "минимум"), ("abl_ep20", "20 эпох"),
             ("abl_ep80", "80 эпох")]
    have = [(l, t) for t, l in items if t in R]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.27
    for j, (m, col) in enumerate((("lstm", PURPLE), ("tcn", NAVY), ("stgcn", ORANGE))):
        def val(t, fn):
            v = fn(t, m)
            if v is None and t == "grp_all":  # lstm в базовом варианте считался в zoo_nets_noasp
                v = fn("zoo_nets_noasp", m)
            return v or 0

        vals = [val(t, g) for _, t in have]
        errs = [val(t, gs) for _, t in have]
        pos = np.arange(len(have)) + (j - 1) * w
        ax.bar(pos, vals, w, yerr=errs, capsize=2, color=col, label=m)
    ax.set_xticks(np.arange(len(have)), [l for l, _ in have], rotation=15, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC по видео")
    ax.set_title("Какие части признаков и обучения важны", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 5. домен
def chart_domain():
    pairs = [("zoo_nets", "dom_excl_cam", "с рамкой"), ("zoo_nets_noasp", "dom_excl_cam_noasp", "только точки")]
    if not any(b in R for _, b, _ in pairs):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labs, a_auc, b_auc, a_fp, b_fp = [], [], [], [], []
    for a, b, lab in pairs:
        for m in ("lstm", "tcn", "stgcn"):
            if g(a, m) is not None and g(b, m) is not None:
                labs.append(f"{m}\n{"+рамка" if lab == "с рамкой" else "точки"}")
                a_auc.append(g(a, m)); b_auc.append(g(b, m))
                a_fp.append(g(a, m, "cam_fpr_05") or 0); b_fp.append(g(b, m, "cam_fpr_05") or 0)
    x = np.arange(len(labs))
    axes[0].bar(x - 0.2, a_auc, 0.4, color=NAVY, label="камеры в обучении")
    axes[0].bar(x + 0.2, b_auc, 0.4, color=ORANGE, label="камер в обучении нет")
    axes[0].set_xticks(x, labs); axes[0].set_ylim(0.4, 1.0); axes[0].set_title("ROC-AUC по видео", loc="left")
    axes[0].legend()
    axes[1].bar(x - 0.2, a_fp, 0.4, color=NAVY)
    axes[1].bar(x + 0.2, b_fp, 0.4, color=ORANGE)
    axes[1].set_xticks(x, labs)
    axes[1].set_title("Ложные срабатывания на видео с камер (порог 0,5)", loc="left")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "domain.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 6. агрегация по видео
def chart_agg():
    fig, ax = plt.subplots(figsize=(11, 5))
    labs = ["k=1 (макс.)", "k=3", "k=5", "среднее"]
    keys = ["auc_k1", "auc", "auc_k5", "auc_kmean"]
    have = [m for m in ("lstm", "tcn", "stgcn", "transformer", "mamba") if g("zoo_nets_noasp", m) is not None]
    if not have:
        return
    w = 0.8 / len(have)
    for j, m in enumerate(have):
        vals = [g("zoo_nets_noasp", m, k) or 0 for k in keys]
        ax.bar(np.arange(4) + (j - len(have) / 2 + 0.5) * w, vals, w, label=m)
    ax.set_xticks(np.arange(4), labs)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC по видео")
    ax.set_title("Как собирать оценку видео из оценок окон", loc="left")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "aggregation.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 7. внимание по суставам
def chart_joints():
    f = CV / "joint_zoo_nets_noasp.npy"
    if not f.exists():
        f = CV / "joint_grp_all.npy"
    if not f.exists():
        return
    w = np.load(f)
    cols = {"spine": NAVY, "hips": ORANGE, "tail": GREEN, "front": PURPLE, "rear": RED}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(np.arange(20), w * 100, color=[cols[g_] for g_ in JOINT_GROUP])
    ax.set_xticks(np.arange(20))
    ax.set_xlabel("индекс ключевой точки")
    ax.set_ylabel("доля внимания, %")
    ax.axhline(5, color="#222", ls=":", lw=1)
    ax.text(19.6, 5.1, "равномерно (5%)", ha="right", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()]
    ax.legend(handles, [GROUP_NAMES[k] for k in cols], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False)
    ax.set_title("Куда смотрит ST-GCN: усреднённое внимание по суставам на тесте", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "joint_attention.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    grp = {k: float(sum(w[i] for i, g_ in enumerate(JOINT_GROUP) if g_ == k)) for k in cols}
    json.dump(grp, open(OUT / "joint_groups.json", "w", encoding="utf-8"), ensure_ascii=False)


# ------------------------------------------------------------------ 8. ошибки по роликам
def errors():
    f = CV / "scores_zoo_nets_noasp.npz"
    if not f.exists():
        return
    z = np.load(f, allow_pickle=True)
    names, cls, dom = z["vid_names"], z["vid_cls"], z["vid_dom"]
    acc = {}
    for m in ("lstm", "tcn", "stgcn"):
        for key in z.files:
            if key.startswith(f"{m}_s"):
                ids = z[f"{m}_ids"]
                for i, s in zip(ids, z[key]):
                    acc.setdefault(int(i), []).append(float(s))
    mean = {i: np.mean(v) for i, v in acc.items()}
    ids = np.array(sorted(mean))
    sc = np.array([mean[i] for i in ids])
    y = cls[ids]
    lines = ["# Ошибки по роликам (среднее по lstm/tcn/stgcn и сидам, только точки)", ""]
    fp = [(sc[k], ids[k]) for k in np.argsort(-sc) if y[k] == 0][:15]
    fn = [(sc[k], ids[k]) for k in np.argsort(sc) if y[k] == 1][:15]
    lines += ["## Нормальные видео с наибольшей оценкой дефекации (ложные срабатывания)", "",
              "| оценка | домен | видео |", "|---|---|---|"]
    lines += [f"| {s:.2f} | {dom[i]} | {names[i]} |" for s, i in fp]
    lines += ["", "## Видео с дефекацией с наименьшей оценкой (пропуски)", "", "| оценка | домен | видео |", "|---|---|---|"]
    lines += [f"| {s:.2f} | {dom[i]} | {names[i]} |" for s, i in fn]
    open(OUT / "errors.md", "w", encoding="utf-8").write("\n".join(lines))
    stat = dict(n=int(len(ids)), fp_hi=int(sum(1 for k in range(len(ids)) if y[k] == 0 and sc[k] >= 0.5)),
                n_neg=int((y == 0).sum()), fn_lo=int(sum(1 for k in range(len(ids)) if y[k] == 1 and sc[k] < 0.5)),
                n_pos=int((y == 1).sum()),
                fp_names=[names[i] for _, i in fp[:10]], fn_names=[names[i] for _, i in fn[:10]])
    json.dump(stat, open(OUT / "errors.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def table():
    cols = [("auc", "AUC"), ("ap", "AP"), ("prec05", "prec@0.5"), ("rec05", "rec@0.5"), ("f1_05", "F1@0.5"),
            ("spec05", "spec@0.5"), ("rec_fpr10", "rec@FPR10%"), ("prec_rec80", "prec@rec80%"), ("mcc05", "MCC"),
            ("brier", "Brier"), ("ece", "ECE"), ("auc_other", "AUC не-камеры"), ("cam_fpr_05", "ложные на камерах")]
    rows = ["| эксперимент | модель | " + " | ".join(c[1] for c in cols) + " | видео |",
            "|" + "---|" * (len(cols) + 3)]
    for tag in sorted(R):
        for m, a in R[tag].items():
            if "auc" not in a:
                continue
            cells = [f"{a[k][0]:.3f}" if k in a else "-" for k, _ in cols]
            rows.append(f"| {tag} | {m} | " + " | ".join(cells) + f" | {META[tag]['n_videos']} |")
    open(OUT / "table_all.md", "w", encoding="utf-8").write("\n".join(rows))


def chart_curves():
    tag = "zoo_nets_noasp" if any(t == "zoo_nets_noasp" for t, _ in CURVES) else None
    if tag is None:
        return
    cols = {"lstm": PURPLE, "tcn": NAVY, "stgcn": ORANGE, "transformer": GREEN, "mamba": RED}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for m, c in cols.items():
        if (tag, m) in CURVES:
            fg, tp, rg, pr = CURVES[(tag, m)]
            axes[0].plot(fg, tp, color=c, label=f"{m} (AUC {g(tag, m):.2f})", lw=2)
            axes[1].plot(rg, pr, color=c, label=m, lw=2)
    for key, c, lab in (("zoo_gbm_noasp", GREY, "gbm"),):
        if (key, "gbm") in CURVES:
            fg, tp, rg, pr = CURVES[(key, "gbm")]
            axes[0].plot(fg, tp, color=c, label=f"gbm (AUC {g(key, 'gbm'):.2f})", lw=2, ls="--")
            axes[1].plot(rg, pr, color=c, label="gbm", lw=2, ls="--")
    axes[0].plot([0, 1], [0, 1], ":", color="#555")
    axes[0].axvline(0.10, color="#999", lw=1)
    axes[0].set(xlabel="доля ложных тревог (нормальные видео)", ylabel="доля найденных дефекаций",
                title="ROC (среднее по сидам)")
    axes[0].legend(loc="lower right")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-Recall", ylim=(0.3, 1.02))
    names = [m for m in cols if m in R.get(tag, {})]
    x = np.arange(len(names))
    w = 0.2
    for j, (k, lab, c) in enumerate((("prec05", "precision", NAVY), ("rec05", "recall", ORANGE),
                                      ("spec05", "specificity", GREEN), ("rec_fpr10", "recall@FPR10%", PURPLE))):
        axes[2].bar(x + (j - 1.5) * w, [g(tag, m, k) or 0 for m in names], w, label=lab, color=c)
    axes[2].set_xticks(x, names)
    axes[2].set(ylim=(0, 1.05), title="Рабочая точка (порог 0,5) и recall при 10% ложных тревог")
    axes[2].legend(ncol=2, loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=170)
    plt.close(fig)



def chart_best():
    if "best_nets" not in R:
        return
    names = ["lstm", "tcn", "transformer", "mamba", "stgcn"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = axes[0]
    series = [("zoo_nets_noasp", "базовая (T=45, с conf/vel/aug)", GREY), ("best_nets", "лучшая конфигурация", NAVY),
              ("best_rep", "повтор: новые разбиения", ORANGE)]
    w = 0.26
    for j, (tag, lab, c) in enumerate(series):
        if tag not in R:
            continue
        vals = [g(tag, m) or 0 for m in names]
        errs = [gs(tag, m) or 0 for m in names]
        pos = np.arange(len(names)) + (j - 1) * w
        ax.bar(pos, vals, w, yerr=errs, capsize=3, color=c, label=lab)
        for p_, v in zip(pos, vals):
            if v:
                ax.text(p_, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    if "best_gbm" in R:
        ax.axhline(g("best_gbm", "gbm"), color=GREEN, ls="--", lw=1.4)
        ax.text(4.45, g("best_gbm", "gbm") + 0.005, f"gbm {g('best_gbm', 'gbm'):.2f}", color=GREEN, ha="right", fontsize=10)
    if "best_ctrl" in R:
        ax.axhline(g("best_ctrl", "meta"), color=RED, ls="--", lw=1.4)
        ax.text(4.45, g("best_ctrl", "meta") - 0.025, f"только метаданные {g('best_ctrl', 'meta'):.2f}", color=RED,
                ha="right", fontsize=10)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC по видео")
    ax.set_title("Лучшая конфигурация и проверка на новых разбиениях", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    ax = axes[1]
    mets = [("prec05", "precision"), ("rec05", "recall"), ("spec05", "specificity"), ("rec_fpr10", "recall@FPR10%")]
    rows = [("best_nets", "lstm", NAVY, "lstm"), ("best_nets", "tcn", ORANGE, "tcn")]
    for tag, m, c, lab in rows:
        pass
    w2 = 0.2
    for j, (tag, m, lab, c) in enumerate((("best_nets", "lstm", "lstm", NAVY), ("best_nets", "tcn", "tcn", ORANGE),
                                           ("best_gbm", "gbm", "gbm", GREEN), ("best_ctrl", "meta", "метаданные", RED))):
        if tag in R and m in R[tag]:
            ax.bar(np.arange(4) + (j - 1.5) * w2, [g(tag, m, k) or 0 for k, _ in mets], w2, label=lab, color=c)
    ax.set_xticks(np.arange(4), [l for _, l in mets])
    ax.set_ylim(0, 1.05)
    ax.set_title("Рабочие метрики лучшей конфигурации (порог 0,5)", loc="left")
    ax.legend(ncol=4, loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "best.png", dpi=170)
    plt.close(fig)


def chart_best_groups():
    base_tag = "best_nets"
    if base_tag not in R or not any(t.startswith("bgrp_") for t in R):
        return
    only = [("only_spine", "позвоночник"), ("only_hips", "таз"), ("only_tail", "хвост"), ("only_front", "перед. лапы"),
            ("only_rear", "задн. лапы"), ("spine_hips", "позв.+таз"), ("hips_tail", "таз+хвост")]
    drop = [("drop_spine", "без позв."), ("drop_hips", "без таза"), ("drop_tail", "без хвоста"),
            ("drop_front", "без перед.л."), ("drop_rear", "без задн.л.")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for ax, items, ttl in ((axes[0], only, "Только выбранные точки"), (axes[1], drop, "Все точки без группы")):
        labs = ["все 20"] + [l for _, l in items]
        w = 0.38
        for j, (m, c) in enumerate((("lstm", PURPLE), ("tcn", NAVY))):
            vals = [g(base_tag, m) or 0] + [g(f"bgrp_{k}", m) or 0 for k, _ in items]
            errs = [gs(base_tag, m) or 0] + [gs(f"bgrp_{k}", m) or 0 for k, _ in items]
            ax.bar(np.arange(len(labs)) + (j - 0.5) * w, vals, w, yerr=errs, capsize=2, color=c, label=m)
        ax.set_xticks(np.arange(len(labs)), labs, rotation=30, ha="right")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(ttl, loc="left")
        if "best_ctrl" in R:
            ax.axhline(g("best_ctrl", "meta"), color=RED, ls="--", lw=1.2)
    axes[0].set_ylabel("ROC-AUC по видео")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "best_groups.png", dpi=170)
    plt.close(fig)


for fn in (chart_curves, chart_best, chart_best_groups, chart_zoo, chart_groups, chart_window, chart_ablation, chart_domain, chart_agg, chart_joints, errors, table):
    try:
        fn()
        print("ok", fn.__name__)
    except Exception as e:  # noqa: BLE001
        print("FAILED", fn.__name__, repr(e))
