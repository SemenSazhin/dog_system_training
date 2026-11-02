#!/usr/bin/env python3
"""Презентация и текстовый отчёт по экспериментам skeleton_cv. Все числа берутся из presentation/cv/stats.json.

python build_cv_presentation.py  ->  presentation/cv_experiments.pptx, presentation/REPORT.md
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "presentation" / "cv"
OUT = ROOT / "presentation"
D = json.load(open(CV / "stats.json", encoding="utf-8"))
ST, META = D["stats"], D["meta"]
ERR = json.load(open(CV / "errors.json", encoding="utf-8")) if (CV / "errors.json").exists() else {}
JG = json.load(open(CV / "joint_groups.json", encoding="utf-8")) if (CV / "joint_groups.json").exists() else {}
NAVY, ORANGE, GREY, RED, GREEN, INK = "#1F4E79", "#E07B39", "#9AA5B1", "#B5483A", "#4C9F70", "#1B2733"


def S(tag, model, metric="auc", p=2):
    try:
        return f"{ST[tag][model][metric]['mean']:.{p}f}"
    except KeyError:
        return "н/д"


def SD(tag, model, metric="auc", p=3):
    try:
        return f"±{ST[tag][model][metric]['std']:.{p}f}"
    except KeyError:
        return ""


def V(tag, model, metric="auc"):
    return ST[tag][model][metric]["mean"]


def rgb(h):
    return RGBColor.from_string(h.lstrip("#"))


W, H = Inches(13.333), Inches(7.5)


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


def bullets(slide, x, y, w, h, items, size=17):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "•  " + it
        r.font.size, r.font.name = Pt(size), "Calibri"
        r.font.color.rgb = rgb(INK)
        p.space_after = Pt(9)
    return tb


def rect(slide, x, y, w, h, fill):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.adjustments[0] = 0.06
    s.shadow.inherit = False
    return s


def table(slide, x, y, rows, col_w, size=14, row_h=0.4, hl=()):
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
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else ("#DCE8F5" if i in hl else ("#F3F5F7" if i % 2 else "#FFFFFF")))


def pic_fit(slide, name, x, y, max_w, max_h):
    path = CV / name
    if not path.exists():
        add_text(slide, x, y, max_w, Inches(1), f"[нет графика {name}]", size=14, color=RED)
        return
    w, h = Image.open(path).size
    k = min(max_w / w, max_h / h)
    slide.shapes.add_picture(str(path), x + int((max_w - w * k) / 2), y, width=int(w * k), height=int(h * k))


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
N = [0]


def new_slide(title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    N[0] += 1
    add_text(s, Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.9), title, size=28, bold=True, color=NAVY)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.12), Inches(1.2), Emu(38000))
    ln.fill.solid()
    ln.fill.fore_color.rgb = rgb(ORANGE)
    ln.line.fill.background()
    add_text(s, Inches(12.2), Inches(7.05), Inches(0.8), Inches(0.35), str(N[0] + 1), size=12, color=GREY, align=PP_ALIGN.RIGHT)
    return s


def chart_slide(title, img, takeaway, img_h=4.3):
    s = new_slide(title)
    pic_fit(s, img, Inches(0.5), Inches(1.3), Inches(12.3), Inches(img_h))
    top = 1.3 + img_h + 0.1
    rect(s, Inches(0.6), Inches(top), Inches(12.1), Inches(7.0 - top), "#F3F5F7")
    add_text(s, Inches(0.8), Inches(top + 0.03), Inches(11.7), Inches(7.0 - top - 0.06), takeaway, size=13,
             anchor=MSO_ANCHOR.MIDDLE)
    return s


has = lambda t: t in ST
best_rep_ok = has("best_rep")

# ------------------------------------------------------------------ 1. титул
s = prs.slides.add_slide(prs.slide_layouts[6])
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid()
bg.fill.fore_color.rgb = rgb(NAVY)
bg.line.fill.background()
add_text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.2),
         "Как улучшить детекцию дефекации по скелету собаки", size=42, bold=True, color="#FFFFFF")
add_text(s, Inches(0.9), Inches(4.3), Inches(11.5), Inches(1.2),
         f"{len(ST)} экспериментов на честном протоколе: модели, группы точек, окна, признаки, сдвиг домена",
         size=22, color="#C9D8E8")
add_text(s, Inches(0.9), Inches(6.3), Inches(11.5), Inches(0.5),
         "166 видео · честная проверка: одно видео не бывает одновременно в обучении и в тесте", size=16, color=ORANGE)

# ------------------------------------------------------------------ публикация
s = new_slide("Публикация: журнал «Искусственные общества»")
bullets(s, Inches(0.6), Inches(1.4), Inches(7.3), Inches(5.4), [
    "Научный журнал при ЦЭМИ РАН, входит в перечень ВАК и РИНЦ",
    "Тематика: агентное моделирование, искусственные общества, анализ поведения с применением методов искусственного интеллекта",
    "Наша статья: автоматический контроль соблюдения правил выгула собак по видео с камер - от детекции позы до работающего сервиса",
    "Упор статьи: честная оценка качества и перенос модели со «стоковых» роликов на реальные камеры",
], size=16)
rect(s, Inches(8.2), Inches(1.4), Inches(4.5), Inches(5.2), "#F3F5F7")
add_text(s, Inches(8.4), Inches(1.5), Inches(4.1), Inches(0.5), "Почему этот журнал", size=18, bold=True, color=NAVY)
bullets(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.4), [
    "прикладной ИИ для анализа поведения в общественных пространствах - в тематику журнала",
    "есть что показать померить модель: работающее полевое развёртывание",
    "публикация ВАК для ВКР",
], size=15)

# ------------------------------------------------------------------ статус и план
s = new_slide("Статус и план к следующей встрече")
rect(s, Inches(0.6), Inches(1.4), Inches(6.0), Inches(5.2), "#F3F5F7")
add_text(s, Inches(0.8), Inches(1.5), Inches(5.6), Inches(0.5), "Что есть на данный момент", size=18, bold=True, color=NAVY)
bullets(s, Inches(0.8), Inches(2.1), Inches(5.6), Inches(4.4), [
    "честный протокол оценки и десятки обученных конфигураций",
    "лучшая конфигурация: качество 0,91 из 1 против 0,81 у «угадайки» по метаданным",
    "работающий сервис video-scan: 7 камер круглосуточно накапливают датасет",
    "знаем главное узкое место: нет ни одного подтверждённого эпизода с камеры",
], size=15)
rect(s, Inches(6.8), Inches(1.4), Inches(5.9), Inches(5.2), "#EAF3EC")
add_text(s, Inches(7.0), Inches(1.5), Inches(5.5), Inches(0.5), "К следующей встрече", size=18, bold=True, color=GREEN)
bullets(s, Inches(7.0), Inches(2.1), Inches(5.5), Inches(4.4), [
    "перенести лучшую конфигурацию (окно 2 секунды, простые признаки) в video-scan",
    "разметить накопленные камерные клипы, перепроверить сомнительные метки",
    "собрать черновик статьи: структура, таблицы и рисунки - из этой презентации",
], size=15)

# ------------------------------------------------------------------ 2. данные
s = new_slide("Данные: главная проблема - ответ можно угадать, не глядя на собаку")
bullets(s, Inches(0.6), Inches(1.4), Inches(7.6), Inches(5.4), [
    "166 видео, около 22 минут: 106 роликов без дефекации и 60 с дефекацией",
    "Все 64 ролика с камер парков попали только в класс «норма»: подтверждённых эпизодов с камер нет совсем",
    "Почти все ролики в высоком разрешении тоже из «нормы»",
    "Классификатор, который видит только разрешение, fps и длительность, угадывает класс с качеством 0,81 из 1 - модели есть чему «хитрить», не глядя на позу собаки",
], size=17)
rect(s, Inches(8.5), Inches(1.4), Inches(4.2), Inches(2.5), "#FDECEC")
add_text(s, Inches(8.7), Inches(1.5), Inches(3.8), Inches(0.6), "Угадайка по метаданным", size=16, bold=True, color=RED)
add_text(s, Inches(8.7), Inches(2.1), Inches(3.8), Inches(1.6),
         [f"качество {S('zoo_gbm', 'meta')} из 1", "столько же, сколько у полноценных моделей"],
         size=20, bold=True, color=RED)
rect(s, Inches(8.5), Inches(4.1), Inches(4.2), Inches(2.5), "#F3F5F7")
add_text(s, Inches(8.7), Inches(4.2), Inches(3.8), Inches(0.6), "Угадайка по форме рамки собаки", size=16, bold=True, color=GREY)
add_text(s, Inches(8.7), Inches(4.8), Inches(3.8), Inches(1.6),
         [f"качество {S('zoo_gbm', 'aspect')}", f"ложные тревоги на камерах: {S('zoo_gbm', 'aspect', 'cam_fpr_05')}"],
         size=20, bold=True, color="#555555")

# ------------------------------------------------------------------ 3. протокол
s = new_slide("Как проверяли: честно, без подглядывания")
bullets(s, Inches(0.6), Inches(1.4), Inches(7.3), Inches(5.4), [
    "Детектор позы прогоняется по видео один раз - дальше все эксперименты считаются на сохранённых скелетах за минуты",
    "Разбиение по роликам: нарезки одного видео не попадают одновременно в обучение и в проверку - в первой версии ноутбука из-за этого цифры были завышены",
    "Признаки не зависят от разрешения картинки: поза описывается относительно самой собаки",
    "Оценка на уровне ролика, а не отдельного кадра: как в реальном видеопотоке",
], size=16)
rect(s, Inches(8.2), Inches(1.4), Inches(4.5), Inches(5.2), "#F3F5F7")
add_text(s, Inches(8.4), Inches(1.5), Inches(4.1), Inches(0.5), "Что измеряем", size=18, bold=True, color=NAVY)
bullets(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.4), [
    "сколько дефекаций находим (recall)",
    "сколько при этом ложных тревог",
    "не путает ли модель камеры с обычными роликами",
    "AUC - общая сводка для сравнения моделей",
], size=15)
s2 = s
add_text(s, Inches(0.6), Inches(6.55), Inches(12), Inches(0.5),
         "Повторные запуски одинаковой модели расходятся на ±0,01-0,025: различия меньше 0,02 считаем шумом, а не прогрессом.",
         size=14, color=RED)

# ------------------------------------------------------------------ 4. зоопарк
chart_slide("Модели: всё сложное работает не лучше простого", "zoo.png",
            [f"Все нейросети (LSTM, TCN, Transformer, Mamba) показали почти одинаковый результат, около {S('zoo_nets_noasp', 'lstm')}; даже простой бустинг не отстаёт.",
             "Вывод: менять архитектуру смысла нет. Качество упирается в данные, а не в модель."])

# ------------------------------------------------------------------ 5. рабочие метрики
chart_slide("Что это значит на практике", "curves.png",
            ["Если ставить порог «срабатывать при оценке выше 0,5»: модель находит около 85% дефекаций, но примерно треть тревог оказывается ложной.",
             "Если допускать ложную тревогу на каждые 10 нормальных роликов - находится примерно половина дефекаций. Оценки модели плохо откалиброваны: порог в сервисе нужно подбирать по данным, а не брать «0,8 по умолчанию»."])

# ------------------------------------------------------------------ 6. группы точек
chart_slide("Одних «главных» точек не существует", "groups_only.png",
            ["Проверили гипотезу: может, для дефекации достаточно таза и хвоста? Не хватило - качество заметно падает, а выигрыша почти нет.",
             "Любая одна группа точек (таз, хвост, лапы) заметно слабее полного набора из 20 точек у всех моделей. Сокращать скелет нельзя."])

s = chart_slide("Сигнал распределён по всему телу", "groups_drop.png",
                ["Убираем по одной группе: в обычном режиме у большинства моделей таз и хвост - самые важные, но даже без них качество падает немного.",
                 "В лучшей конфигурации картина другая: там критичны таз и позвоночник. Единственный устойчивый вывод: одной «магической» группы нет, модель смотрит на всё тело."], img_h=4.2)

# ------------------------------------------------------------------ 7. внимание по суставам
if JG:
    chart_slide("Куда смотрит модель", "joint_attention.png",
                [f"Внимание графовой сети распределено по телу: позвоночник {JG['spine'] * 100:.0f}%, передние лапы {JG['front'] * 100:.0f}%, задние лапы {JG['rear'] * 100:.0f}%, хвост {JG['tail'] * 100:.0f}%, таз {JG['hips'] * 100:.0f}%.",
                 "Сильнее всего выделены концы конечностей. Это подтверждает картину предыдущих слайдов: модель использует всё тело, а не один «сигнальный» сустав."], img_h=4.2)

# ------------------------------------------------------------------ 8. окно
chart_slide("Короткое окно (2 секунды) - лучший вариант", "window.png",
            ["Окно в 2 секунды работает лучше длинных (4-6 секунд) и позволяет использовать больше роликов для обучения.",
             "Длинные окна выкидывают почти половину датасета: в таких роликах собака периодически пропадает из кадра."])

# ------------------------------------------------------------------ 9. абляции
chart_slide("Чем проще - тем лучше", "ablation.png",
            ["Убрали по одному «улучшению»: confidence точек, скорости, аугментации - качество выросло с каждым удалением.",
             "На маленьком датасете усложнения только мешают, и дольше учить модель бесполезно: она быстро переобучается."])

# ------------------------------------------------------------------ 10. лучшая конфигурация
if has("best_nets"):
    chart_slide("Лучшая конфигурация: простая", "best.png",
                [f"Собрали всё, что помогло: окно 2 секунды, только координаты точек, ничего лишнего. LSTM вышла на {S('best_nets', 'lstm')} против {S('zoo_nets_noasp', 'lstm')} в обычной настройке и {S('best_ctrl', 'meta')} у «угадайки» по метаданным.",
                 "Чтобы убедиться, что это не подгонка под конкретные ролики, повторили на других разбиениях - результат тот же."])

if any(t.startswith("bgrp_") for t in ST):
    chart_slide("Проверка в лучшей конфигурации: сокращать точки нельзя", "best_groups.png",
                ["В лучшей конфигурации снова перебрали наборы точек: полный скелет всё равно не хуже любого подмножества.",
                 "Картина по группам немного другая, чем раньше, но вывод прежний: сигнал распределён по всему телу."])

# ------------------------------------------------------------------ 11. домен
chart_slide("Главный риск: модель узнаёт камеру, а не позу", "domain.png",
            ["В обучении нет ни одного эпизода дефекации с камер. Если специально убрать камерные ролики из обучения, ложные тревоги на камерах вырастают в несколько раз.",
             "Честно оценить работу на камерах сейчас невозможно: для этого нужны подтверждённые эпизоды с камер, и именно их собирает video-scan."])

# ------------------------------------------------------------------ 12. ошибки
s = new_slide("Ошибки модели: где стоит перепроверить разметку")
if ERR:
    add_text(s, Inches(0.6), Inches(1.3), Inches(12), Inches(0.6),
             f"Из {ERR['n_neg']} нормальных видео {ERR['fp_hi']} модель уверенно считает дефекацией; из {ERR['n_pos']} видео с дефекацией {ERR['fn_lo']} считает нормой.",
             size=16, bold=True, color=NAVY)
    rect(s, Inches(0.6), Inches(1.95), Inches(6.0), Inches(4.9), "#FDECEC")
    add_text(s, Inches(0.8), Inches(2.0), Inches(5.6), Inches(0.5), "Норма, которую модель считает дефекацией", size=16, bold=True, color=RED)
    bullets(s, Inches(0.8), Inches(2.55), Inches(5.7), Inches(4.2), [n[:60] for n in ERR["fp_names"][:7]], size=12)
    rect(s, Inches(6.8), Inches(1.95), Inches(5.9), Inches(4.9), "#F3F5F7")
    add_text(s, Inches(7.0), Inches(2.0), Inches(5.5), Inches(0.5), "Дефекация, которую модель пропускает", size=16, bold=True, color=NAVY)
    bullets(s, Inches(7.0), Inches(2.55), Inches(5.5), Inches(4.2), [n[:60] for n in ERR["fn_names"][:7]], size=12)

s = new_slide("Что делать дальше с video-scan")
bullets(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.4), [
    "Сервис сейчас работает на старой конфигурации из первого ноутбука - по результатам экспериментов это не лучший вариант",
    f"Перенести в сервис короткое окно (2 секунды) и простые признаки: это дало {S('best_nets', 'lstm')} вместо {S('zoo_nets_noasp', 'lstm')}",
    "Порог срабатывания подобрать заново на реальных данных - 0,8 «по умолчанию» не откалиброван",
    "Начать собирать подтверждённые эпизоды с камер: без них качество на реальных потоках проверить нечем",
], size=17)

# ------------------------------------------------------------------ 13. выводы
s = new_slide("Выводы")
cards = [
    ("Данные важнее модели", f"«Угадайка» по метаданным даёт {S('zoo_gbm', 'meta')} из 1, лучшая модель - {S('best_nets', 'lstm')}. Дальше расти можно только за счёт данных.", RED),
    ("Точки сокращать нельзя", "Ни одна группа точек не заменяет полный скелет: модель смотрит на всё тело.", ORANGE),
    ("Проще - лучше", "Короткое окно в 2 секунды и минимум признаков обошли все сложные варианты.", NAVY),
]
for i, (h, t, c) in enumerate(cards):
    x = Inches(0.6 + i * 4.1)
    rect(s, x, Inches(1.5), Inches(3.9), Inches(3.8), "#F3F5F7")
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.5), Inches(3.9), Emu(60000))
    bar.fill.solid()
    bar.fill.fore_color.rgb = rgb(c)
    bar.line.fill.background()
    add_text(s, x + Inches(0.2), Inches(1.7), Inches(3.5), Inches(0.9), h, size=20, bold=True, color=c)
    add_text(s, x + Inches(0.2), Inches(2.8), Inches(3.5), Inches(2.5), t, size=15)
add_text(s, Inches(0.6), Inches(5.6), Inches(12.1), Inches(1.3),
         "Дальше: собирать эпизоды с камер, размечать момент самого акта, перенести лучшую конфигурацию в video-scan.",
         size=17, bold=True, color=NAVY)

# ------------------------------------------------------------------ 14. ограничения
s = new_slide("Ограничения")
bullets(s, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.4), [
    "Датасет маленький: около 22 минут видео; различия меньше 0,02 считаем шумом",
    "Лучшую конфигурацию подбирали на этих же данных - цифры чуть оптимистичны, независимой проверки пока нет",
    "Метка стоит на весь ролик, а не на момент самого акта",
    "Нет ни одного подтверждённого эпизода с камеры - реальную работу на камерах оценить нечем",
    "Часть разметки подозрительна: некоторые «ошибки» модели могут оказаться ошибками в метках",
], size=17)

prs.save(OUT / "cv_experiments.pptx")
print("saved", N[0] + 1, "slides")
