#!/usr/bin/env python3
"""Презентация Human Cleanup Detector (обучение только на synthetic).

Числа: presentation/cleanup/architecture_comparison.json
Выход: presentation/cleanup_detector_report.pptx

python scripts/build_cleanup_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "presentation" / "cleanup"
OUT = ROOT / "presentation" / "cleanup_detector_report.pptx"
METRICS = json.loads((CV / "architecture_comparison.json").read_text(encoding="utf-8"))
META = json.loads((ROOT / "datasets" / "cleanup_proxy" / "meta.json").read_text(encoding="utf-8"))

NAVY, ORANGE, GREY, INK = "#1F4E79", "#E07B39", "#9AA5B1", "#1B2733"
W, H = Inches(13.333), Inches(7.5)

N_POS = META.get("n_pos", 120)
N_NEG = META.get("n_neg", 120)
N_ALL = N_POS + N_NEG
WINDOW = META.get("window", 48)
N_KP = META.get("keypoints", 17)
FEAT = METRICS["protocol"]["feature_dim"]
N_TRAIN, N_VAL, N_TEST = 154, 38, 48


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{p}f}".replace(".", ",")


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text if isinstance(text, list) else [text]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, "Calibri"
        r.font.color.rgb = rgb(color)
        p.space_after = Pt(6)
    return tb


def bullets(slide, x, y, w, h, items, size=16):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "•  " + it
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = rgb(INK)
        p.space_after = Pt(8)
    return tb


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.adjustments[0] = 0.06
    s.shadow.inherit = False
    return s


def table(slide, x, y, rows, col_w, size=13, row_h=0.38, hl=()):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, Inches(sum(col_w)), Inches(row_h * len(rows)))
    t = shp.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size, f.name = Pt(size), "Calibri"
            f.bold = i == 0 or i in hl
            f.color.rgb = rgb("#FFFFFF") if i == 0 else rgb(INK)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(
                NAVY if i == 0 else ("#DCE8F5" if i in hl else ("#F3F5F7" if i % 2 else "#FFFFFF"))
            )


def box(slide, text, top=6.25, h=0.7):
    rect(slide, Inches(0.6), Inches(top), Inches(12.1), Inches(h), "#F3F5F7")
    add_text(
        slide,
        Inches(0.8),
        Inches(top + 0.02),
        Inches(11.7),
        Inches(h - 0.04),
        text,
        size=13,
        anchor=MSO_ANCHOR.MIDDLE,
    )


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
N = [0]


def new_slide(title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    N[0] += 1
    add_text(s, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.85), title, size=28, bold=True, color=NAVY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.15), Inches(1.2), Emu(38000))
    ln.fill.solid()
    ln.fill.fore_color.rgb = rgb(ORANGE)
    ln.line.fill.background()
    add_text(
        s,
        Inches(12.2),
        Inches(7.05),
        Inches(0.8),
        Inches(0.35),
        str(N[0]),
        size=12,
        color=GREY,
        align=PP_ALIGN.RIGHT,
    )
    return s


def metrics_chart():
    rows = METRICS["results"]
    names = [r["tag"].replace("_", "\n") for r in rows]
    f1 = [r["f1"] for r in rows]
    auc = [r["roc_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, f1, w, color=NAVY, label="F1")
    ax.bar(x + w / 2, auc, w, color=ORANGE, label="ROC-AUC")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("метрика на тестовой части")
    ax.set_title(f"Сравнение моделей на отложенном synthetic-тесте ({N_TEST} окон)", loc="left", fontsize=12)
    ax.legend(loc="lower right")
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (a, b) in enumerate(zip(f1, auc)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    p = CV / "arch_metrics.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def class_sketch_chart():
    t = np.linspace(0, 1, WINDOW)
    squat = 0.12 * np.sin(np.pi * np.clip((t - 0.2) / 0.5, 0, 1))
    reach = 0.20 * np.sin(np.pi * np.clip((t - 0.25) / 0.45, 0, 1))
    wrist_c = 0.70 + reach
    hip_c = 0.55 + squat
    sway = 0.03 * np.sin(2 * np.pi * t * 2)
    wrist_n = np.full_like(t, 0.48) + 0.01 * np.sin(2 * np.pi * t)
    hip_n = np.full_like(t, 0.55)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6), sharey=True)
    ax = axes[0]
    ax.plot(t, hip_c, color=NAVY, lw=2.2, label="бёдра")
    ax.plot(t, wrist_c, color=ORANGE, lw=2.2, label="запястья")
    ax.set_title("cleanup: человек приседает и тянется вниз", loc="left", fontsize=12)
    ax.set_xlabel("время внутри короткого клипа")
    ax.set_ylabel("положение по вертикали")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(t, hip_n, color=NAVY, lw=2.2, label="бёдра")
    ax.plot(t, wrist_n, color=ORANGE, lw=2.2, label="запястья")
    ax.plot(t, 0.50 + sway, color=GREY, lw=1.5, ls="--", label="лёгкое покачивание")
    ax.set_title("no_cleanup: стоит или просто идёт", loc="left", fontsize=12)
    ax.set_xlabel("время внутри короткого клипа")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    for a in axes:
        a.set_ylim(0.35, 0.95)
    fig.tight_layout()
    p = CV / "synthetic_classes.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ================================================================= 1. титул
s = prs.slides.add_slide(prs.slide_layouts[6])
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = rgb(NAVY)
bg.line.fill.background()
add_text(
    s,
    Inches(0.9),
    Inches(2.0),
    Inches(11.5),
    Inches(1.8),
    "Детекция уборки за собакой человеком",
    size=40,
    bold=True,
    color="#FFFFFF",
)
add_text(
    s,
    Inches(0.9),
    Inches(3.8),
    Inches(11.5),
    Inches(1.0),
    "Пилот на синтетическом датасете поз: идея, модели, что получилось",
    size=22,
    color="#C9D8E8",
)
add_text(
    s,
    Inches(0.9),
    Inches(5.4),
    Inches(11.5),
    Inches(0.5),
    f"{N_ALL} коротких клипов · {N_POS} уборка / {N_NEG} без уборки · Colab HumanCleanupDetector",
    size=16,
    color=ORANGE,
)

# ================================================================= 2. постановка
s = new_slide("О чём задача")
bullets(
    s,
    Inches(0.6),
    Inches(1.7),
    Inches(7.0),
    Inches(4.5),
    [
        "Нужно понять по движению человека: убирает он за собакой или нет",
        "Смотрим не на сами фекалии, а на жест: присед, рука к земле, подъём",
        "На вход модели идёт не «картинка целиком», а скелет человека во времени",
        "В этом прогоне учили только на искусственных клипах поз — без реальных камер",
        "Зачем так: быстро проверить идею и сравнить несколько моделей на одном наборе",
    ],
    size=16,
)
rect(s, Inches(8.0), Inches(1.85), Inches(4.7), Inches(4.2), "#F3F5F7")
add_text(s, Inches(8.25), Inches(2.05), Inches(4.2), Inches(0.5), "Коротко", size=18, bold=True, color=NAVY)
add_text(
    s,
    Inches(8.25),
    Inches(2.7),
    Inches(4.2),
    Inches(3.0),
    [
        "Уборка = характерный жест",
        "во времени.",
        "",
        "Модель учится отличать",
        "«нагнулся и потянулся»",
        "от «просто стоит / идёт».",
    ],
    size=16,
)

# ================================================================= 3. что такое скелет / coco-17 + природа датасета
s = new_slide("Что за датасет и что такое COCO-17")
bullets(
    s,
    Inches(0.6),
    Inches(1.55),
    Inches(12.1),
    Inches(2.2),
    [
        "Это не видео с улицы и не фото людей. Мы программно нарисовали движение скелета: точки тела по кадрам",
        "COCO-17 — просто стандартный набор из 17 точек человека: нос, плечи, локти, запястья, бёдра, колени, лодыжки",
        "Так же выглядит выход YOLO-pose: поэтому синтетика сразу в том же формате, в котором потом будет реальное видео",
        "Каждый пример — короткий клип на 48 кадров. Класс задали сами при генерации: cleanup или no_cleanup",
    ],
    size=15,
)
table(
    s,
    Inches(0.6),
    Inches(4.0),
    [
        ["Параметр", "Значение"],
        ["Сколько примеров", f"{N_ALL}: {N_POS} cleanup и {N_NEG} no_cleanup"],
        ["Что внутри примера", f"{WINDOW} кадров × {N_KP} точек (x, y и уверенность)"],
        ["Откуда взялось", "сгенерировано кодом, seed 42, лёгкий шум на координаты"],
        ["Файл", "datasets/cleanup_proxy/synthetic_pose_sequences.npz"],
    ],
    [3.6, 8.5],
    size=13,
    row_h=0.42,
)
box(
    s,
    "Простыми словами: это учебный тренажёр. Мы сами нарисовали «как выглядит уборка» и «как выглядит обычная ходьба», "
    "и на этом учили модель.",
    top=6.3,
    h=0.7,
)

# ================================================================= 4. два класса
s = new_slide("Два класса внутри synthetic")
sketch = class_sketch_chart()
s.shapes.add_picture(str(sketch), Inches(0.45), Inches(1.5), width=Inches(12.4))
bullets(
    s,
    Inches(0.6),
    Inches(5.35),
    Inches(12.1),
    Inches(1.4),
    [
        "cleanup: в середине клипа корпус опускается, руки уходят вниз — как будто человек нагнулся поднять",
        "no_cleanup: человек стоит или слегка идёт, руки около корпуса, без заметного наклона к земле",
        "Классы специально сделаны разными, чтобы проверить: видит ли модель этот жест вообще",
    ],
    size=14,
)

# ================================================================= 5. признаки и обучение
s = new_slide("Как готовили признаки и учили")
bullets(
    s,
    Inches(0.6),
    Inches(1.6),
    Inches(6.5),
    Inches(4.5),
    [
        "Из 17 точек считали удобные признаки: где суставы относительно корпуса, углы рук и ног",
        "Отдельно смотрели на вещи, важные для уборки: насколько низко руки, наклон корпуса, сгиб коленей",
        "Добавляли изменение признаков от кадра к кадру — чтобы модель чувствовала движение, а не позу «застывшую»",
        f"На выходе каждое окно — последовательность признаков размерности {FEAT}",
    ],
    size=15,
)
rect(s, Inches(7.5), Inches(1.7), Inches(5.2), Inches(4.3), "#F3F5F7")
add_text(s, Inches(7.75), Inches(1.9), Inches(4.7), Inches(0.45), "Как делили и учили", size=17, bold=True, color=NAVY)
add_text(
    s,
    Inches(7.75),
    Inches(2.5),
    Inches(4.7),
    Inches(3.2),
    [
        f"Обучение: {N_TRAIN} окон",
        f"Проверка по ходу: {N_VAL}",
        f"Финальный тест: {N_TEST}",
        "",
        "Одинаковый сплит для всех моделей",
        "12 эпох, AdamW, batch 32",
        "Лучшие веса брали по F1 на проверке",
    ],
    size=15,
)

# ================================================================= 6. архитектуры
s = new_slide("Какие модели сравнивали")
table(
    s,
    Inches(0.6),
    Inches(1.6),
    [
        ["Группа", "Модели", "Зачем брали"],
        ["Обычные рекуррентные", "LSTM, BiLSTM, GRU, BiGRU", "простой и понятный baseline"],
        ["SSM", "MambaLite", "современная альтернатива LSTM"],
        ["Transformer", "Transformer с CLS-токеном, wide-вариант", "смотрит на всё окно сразу"],
        ["Transformer", "TemporalTransformer", "вариант «как для поз» с pooling"],
    ],
    [3.0, 4.6, 4.5],
    size=14,
    row_h=0.55,
)
bullets(
    s,
    Inches(0.6),
    Inches(4.8),
    Inches(12.1),
    Inches(1.8),
    [
        "Все модели ели одни и те же признаки и один и тот же train/val/test",
        "Смысл сравнения: не «какая модель моднее», а даёт ли усложнение выигрыш на этой задаче",
    ],
    size=15,
)

# ================================================================= 7. метрики
s = new_slide("Метрики: что показали цифры")
chart = metrics_chart()
s.shapes.add_picture(str(chart), Inches(0.5), Inches(1.5), width=Inches(12.3))
box(
    s,
    "На отложенном synthetic-тесте все восемь моделей дали F1 = 1,0 и ROC-AUC = 1,0. "
    "То есть жест «нагнулся / потянулся» от обычной ходьбы они отделяют идеально — на этом учебном наборе.",
    top=5.85,
    h=0.95,
)

# ================================================================= 8. таблица
s = new_slide("Таблица сравнения")
rows = [["Модель", "Группа", "Параметры", "F1", "ROC-AUC", "Accuracy"]]
for r in METRICS["results"]:
    rows.append(
        [
            r["tag"],
            r["family"],
            f"{r['n_params']:,}".replace(",", " "),
            fmt(r["f1"], 2),
            fmt(r["roc_auc"], 2),
            fmt(r["accuracy"], 2),
        ]
    )
table(s, Inches(0.6), Inches(1.55), rows, [3.2, 1.8, 2.0, 1.4, 1.6, 1.6], size=12, row_h=0.42)
add_text(
    s,
    Inches(0.6),
    Inches(5.55),
    Inches(12.1),
    Inches(1.2),
    [
        "По качеству все одинаковые. Разница только в размере модели.",
        "На таком простом наборе сложные Transformer и Mamba ничего не выиграли у обычной LSTM.",
    ],
    size=15,
)

# ================================================================= 9. видео
s = new_slide("Видео-примеры")
for i, title in enumerate(["Пример cleanup", "Пример no_cleanup", "Как модель размечает ролик"]):
    x = 0.7 + i * 4.1
    rect(s, Inches(x), Inches(1.8), Inches(3.7), Inches(3.3), "#EEF2F6", line=GREY)
    add_text(
        s,
        Inches(x + 0.15),
        Inches(3.05),
        Inches(3.4),
        Inches(0.9),
        ["[сюда видео / кадры]", title],
        size=14,
        color=GREY,
        align=PP_ALIGN.CENTER,
    )
bullets(
    s,
    Inches(0.6),
    Inches(5.4),
    Inches(12.1),
    Inches(1.3),
    [
        "Сюда лучше поставить живые ролики: нагнулся / не нагнулся",
        "Важно: модель училась на синтетике, поэтому видео — это демо, а не новая таблица метрик",
    ],
    size=15,
)

# ================================================================= 10. итог
s = new_slide("Что в итоге")
left = [
    "Собрали учебный набор из 240 синтетических клипов поз",
    "Описали уборку как присед и руку вниз, а «норму» как стояние/ходьбу",
    "Прогнали через единый пайплайн признаков восемь моделей",
    "Померили все на одном отложенном тесте",
]
right = [
    "Идея сработала: модели уверенно видят этот жест на synthetic",
    "Все архитектуры дали одинаково высокий результат",
    "Для такого набора хватает простой LSTM — усложнять смысла не было",
    "Это пока тренажёр, не полевая проверка на камерах",
]
rect(s, Inches(0.6), Inches(1.6), Inches(5.9), Inches(4.0), "#F3F5F7")
rect(s, Inches(6.8), Inches(1.6), Inches(5.9), Inches(4.0), "#F3F5F7")
add_text(s, Inches(0.85), Inches(1.75), Inches(5.4), Inches(0.45), "Что сделали", size=18, bold=True, color=NAVY)
add_text(s, Inches(7.05), Inches(1.75), Inches(5.4), Inches(0.45), "Как это читать", size=18, bold=True, color=ORANGE)
bullets(s, Inches(0.85), Inches(2.35), Inches(5.4), Inches(3.0), left, size=14)
bullets(s, Inches(7.05), Inches(2.35), Inches(5.4), Inches(3.0), right, size=14)
box(
    s,
    "Если коротко: мы не утверждаем, что система уже готова к камерам. Мы показали, что пайплайн живой, "
    "жест уборки модель выучивает, а на этом учебном наборе сложные сети не лучше простой LSTM.",
    top=5.95,
    h=0.95,
)


def save(path=OUT):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(path)
    except PermissionError:
        alt = path.with_name(path.stem + "_v2" + path.suffix)
        prs.save(alt)
        print("locked, saved", alt, "slides:", len(prs.slides))
        return alt
    print("saved", path, "slides:", len(prs.slides))
    return path


if __name__ == "__main__":
    save()
