import math, random
from collections import deque, Counter
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import json, datetime as dt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from damp_light import *


def _grid_for_bits(total_bits: int) -> Tuple[int, int]:
    """Подбираем почти квадратную решётку (rows, cols) под total_bits."""
    rows = int(np.floor(np.sqrt(total_bits)))
    while rows > 1 and (total_bits % rows) != 0:
        rows -= 1
    cols = total_bits // rows
    return rows, cols

def _draw_detector_outline(ax, d, color="tab:red", lw=1.5):
    cy, cx = d["center"]
    if d.get("shape") == "ellipse":
        u1, u2 = d["u1"], d["u2"]; r1, r2 = d["r1"], d["r2"]
        ts = np.linspace(0, 2*np.pi, 200)
        ys = cy + r1*np.cos(ts)*u1[0] + r2*np.sin(ts)*u2[0]
        xs = cx + r1*np.cos(ts)*u1[1] + r2*np.sin(ts)*u2[1]
        ax.plot(xs, ys, color=color, lw=lw)
        ax.plot([cx], [cy], marker="o", ms=3, color=color)
    else:
        r = d["radius"]
        ts = np.linspace(0, 2*np.pi, 200)
        ys = cy + r*np.sin(ts)
        xs = cx + r*np.cos(ts)
        ax.plot(xs, ys, color=color, lw=lw)
        ax.plot([cx], [cy], marker="o", ms=3, color=color)


def visualize_pipeline(img: np.ndarray,
                       enc: "PrimaryEncoder",
                       lay: "Layout2D",
                       det: "DetectorSpace",
                       clf: Optional["KNNJaccard"]=None,
                       true_label: Optional[int]=None,
                       title: str = "От стимула к смыслу",
                       show: bool = False,
                       overlay_weight: str = "activation",   # "activation" | "uniform"
                       overlay_only_fired: bool = True,
                       overlay_cmap: str = "inferno"):
    """
    Показывает весь конвейер:
      1) исходное изображение
      2) активные биты первичного кода
      3) карта активации на раскладке
      4) сработавшие детекторы (контуры)
      5) финальный эмбеддинг (битовое полотно)
      6) (если clf) предсказание kNN
    """
    # 1) первичный код
    code = enc.encode(img)

    # 2) карта активации на раскладке
    #    используем внутренний адаптивный порог детекторов
    act = det._activation_map_adaptive(code)  # bool [H,W]
    H, W = act.shape

    # карта-накопитель детекторов для этого стимула
    ov = detector_overlay_matrix(code, det, only_fired=True, weight=overlay_weight, normalize=True)

    # 3) эмбеддинг и сработавшие детекторы
    ones = det.embed(code)  # множество активных битов
    fired = [d for d in det.detectors if d["bit"] in ones]

    # 4) битовое полотно эмбеддинга
    rows, cols = _grid_for_bits(det.emb_bits)
    emb_img = np.zeros((rows, cols), dtype=np.uint8)
    for b in ones:
        r, c = divmod(b, cols)
        emb_img[r, c] = 1

    # 5) предсказание (если есть clf)
    pred_txt = ""
    if clf is not None:
        yhat = clf.predict([ones])[0]
        if true_label is not None:
            ok = "✓" if int(yhat) == int(true_label) else "✗"
            pred_txt = f"Предсказание kNN: {yhat}  (истина: {true_label}) {ok}"
        else:
            pred_txt = f"Предсказание kNN: {yhat}"

    # --- РИСУЕМ ---
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # A) исходное изображение
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax1.set_title("Стимул (изображение)")
    ax1.axis("off")

    # B) первичный код — «штрих-код» активных позиций
    ax2 = plt.subplot(2, 3, 2)
    if len(code) > 0:
        xs = np.fromiter(code, dtype=np.int64)
        ax2.scatter(xs, np.zeros_like(xs), s=8, marker="|")
    ax2.set_xlim(-enc.B*0.02, enc.B*1.02)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Индексы активных битов (первичный код)")
    ax2.set_title(f"Активных битов: {len(code)} из {enc.B}")

    # C) карта активации
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(act, cmap="Greys", origin="upper")
    ax3.set_title("Активация на раскладке (True=сходно)")
    ax3.set_xticks([]); ax3.set_yticks([])

    # D) сработавшие детекторы (контуры поверх активации)
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(act, cmap="Greys", origin="upper")
    for d in fired:
        _draw_detector_outline(ax4, d, color="tab:red", lw=1.5)
    ax4.set_title(f"Сработавшие детекторы: {len(fired)}")
    ax4.set_xticks([]); ax4.set_yticks([])

    # E) эмбеддинг — штрихкод активных битов
    ax5 = plt.subplot(2, 3, 5)
    if len(ones) > 0:
        xs = np.fromiter(sorted(ones), dtype=np.int64)
        ax5.scatter(xs, np.zeros_like(xs), s=8, marker="|")
    ax5.set_xlim(-det.emb_bits * 0.02, det.emb_bits * 1.02)
    ax5.set_ylim(-1, 1)
    ax5.set_yticks([])
    ax5.set_xlabel("Индексы активных битов (эмбеддинг)")
    ax5.set_title(f"Эмбеддинг: {len(ones)} активн. битов из {det.emb_bits}")

    # F) карта-накопитель сработавших детекторов
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(ov, cmap=overlay_cmap, origin="upper")
    ax6.set_title(
        f"Накопительная карта детекторов\n"
        f"({'только сработавшие' if overlay_only_fired else 'все детекторы'}, weight={overlay_weight})"
    )
    ax6.set_xticks([])
    ax6.set_yticks([])

    # небольшая сводка в углу
    info = []
    if clf is not None:
        yhat = clf.predict([ones])[0]
        if true_label is not None:
            ok = "✓" if int(yhat) == int(true_label) else "✗"
            info.append(f"kNN: {yhat} (истина: {true_label}) {ok}")
        else:
            info.append(f"kNN: {yhat}")
    info.append(f"Раскладка: {H}×{W}")
    info.append(f"Детекторов: {len(det.detectors)}")
    ax6.text(
        0.02, 0.98, "\n".join(info),
        transform=ax6.transAxes, va="top", ha="left",
        fontsize=9, color="w",
        bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none")
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


def detector_overlay_matrix(code: Set[int],
                            det: "DetectorSpace",
                            only_fired: bool = True,
                            weight: str = "activation",   # "uniform" | "activation"
                            normalize: bool = True) -> np.ndarray:
    """
    Строит H×W карту-накопитель поверх раскладки:
      - Берём все детекторы (или только сработавшие, если only_fired=True),
      - Для их областей (круг/эллипс) увеличиваем яркость.
    Аргументы:
      code        — разрежённый код стимула (enc.encode(img))
      det         — DetectorSpace
      only_fired  — True: учитывать только реально сработавшие детекторы; False: все
      weight      — "uniform": +1 за каждую область;
                     "activation": +1 только в тех клетках, где карта схожести True
      normalize   — True: нормировать карту в [0,1] (если max>0)
    Возвращает:
      overlay: np.ndarray float32 формы (H, W)
    """
    H, W = det.H, det.W
    overlay = np.zeros((H, W), dtype=np.float32)

    # булева карта схожести (адаптивный порог), пригодится для weight="activation"
    act = det._activation_map_adaptive(code)

    def det_fired(d) -> bool:
        # тот же критерий, что в det.embed()
        cy, cx = d["center"]
        if d.get("shape") == "ellipse":
            u1, u2 = d["u1"], d["u2"]; r1, r2 = d["r1"], d["r2"]
            y0, y1 = max(0, int(cy - r1 - r2)), min(H - 1, int(cy + r1 + r2) + 1)
            x0, x1 = max(0, int(cx - r1 - r2)), min(W - 1, int(cx + r1 + r2) + 1)
            tot = hit = 0
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    vy, vx = (y - cy), (x - cx)
                    a = (vy * u1[0] + vx * u1[1]) / (r1 + 1e-9)
                    b = (vy * u2[0] + vx * u2[1]) / (r2 + 1e-9)
                    if (a*a + b*b) <= 1.0:
                        tot += 1
                        if act[y, x]: hit += 1
            return (tot > 0) and ((hit / tot) >= det.mu)
        else:
            r = d["radius"]
            y0, y1 = max(0, int(cy - r)), min(H - 1, int(cy + r) + 1)
            x0, x1 = max(0, int(cx - r)), min(W - 1, int(cx + r) + 1)
            tot = hit = 0
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2 + 1e-9:
                        tot += 1
                        if act[y, x]: hit += 1
            return (tot > 0) and ((hit / tot) >= det.mu)

    def add_region(d):
        cy, cx = d["center"]
        if d.get("shape") == "ellipse":
            u1, u2 = d["u1"], d["u2"]; r1, r2 = d["r1"], d["r2"]
            y0, y1 = max(0, int(cy - r1 - r2)), min(H - 1, int(cy + r1 + r2) + 1)
            x0, x1 = max(0, int(cx - r1 - r2)), min(W - 1, int(cx + r1 + r2) + 1)
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    vy, vx = (y - cy), (x - cx)
                    a = (vy * u1[0] + vx * u1[1]) / (r1 + 1e-9)
                    b = (vy * u2[0] + vx * u2[1]) / (r2 + 1e-9)
                    if (a*a + b*b) <= 1.0:
                        if weight == "activation":
                            overlay[y, x] += 1.0 if act[y, x] else 0.0
                        else:
                            overlay[y, x] += 1.0
        else:
            r = d["radius"]
            y0, y1 = max(0, int(cy - r)), min(H - 1, int(cy + r) + 1)
            x0, x1 = max(0, int(cx - r)), min(W - 1, int(cx + r) + 1)
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2 + 1e-9:
                        if weight == "activation":
                            overlay[y, x] += 1.0 if act[y, x] else 0.0
                        else:
                            overlay[y, x] += 1.0

    # если хотим именно «сработавшие биты», вычислим по текущему коду
    if only_fired:
        for d in det.detectors:
            if det_fired(d):
                add_region(d)
    else:
        for d in det.detectors:
            add_region(d)

    if normalize and overlay.max() > 0:
        overlay = overlay / float(overlay.max())

    return overlay

def class_overlay_matrix(label: int,
                         X: List[np.ndarray],
                         y: List[int],
                         enc: "PrimaryEncoder",
                         det: "DetectorSpace",
                         limit: int = 100,
                         weight: str = "activation",   # "uniform" | "activation"
                         normalize: bool = True) -> np.ndarray:
    """
    Накопительная карта по многим примерам ОДНОГО класса (например, '9').
    Для каждого примера строит overlay (как для одного изображения) и суммирует.
    """
    # индексы нужного класса
    idxs = [i for i, t in enumerate(y) if int(t) == int(label)]
    if not idxs:
        return np.zeros((det.H, det.W), dtype=np.float32)

    # отберём не более limit примеров (для скорости)
    if limit and len(idxs) > limit:
        rng = np.random.default_rng(0)
        idxs = list(rng.choice(idxs, size=limit, replace=False))

    H, W = det.H, det.W
    acc = np.zeros((H, W), dtype=np.float32)

    for i in idxs:
        code = enc.encode(X[i])
        ov = detector_overlay_matrix(
            code, det,
            only_fired=True,
            weight=weight,
            normalize=False  # важно: суммируем «сырые» карты
        )
        acc += ov

    if normalize and acc.max() > 0:
        acc = acc / float(acc.max())
    return acc


def save_class_overlay_pdf(label: int,
                           X: List[np.ndarray],
                           y: List[int],
                           enc: "PrimaryEncoder",
                           det: "DetectorSpace",
                           limit: int = 100,
                           weight: str = "activation",   # "uniform" | "activation"
                           normalize: bool = True,
                           pdf_path: str = None,
                           cmap: str = "inferno") -> str:
    """
    Строит суммарную карту для класса 'label' и сохраняет в одиноковый PDF (1 страница).
    Возвращает путь к PDF.
    """
    overlay = class_overlay_matrix(label, X, y, enc, det, limit=limit,
                                   weight=weight, normalize=normalize)
    pdf_path = pdf_path or f"class_{label}_overlay_limit{limit}_{weight}.pdf"

    with PdfPages(pdf_path) as pdf:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(overlay, cmap=cmap, origin="upper")
        ax.set_title(
            f"Суммарная карта детекторов — класс '{label}'\n"
            f"limit={limit}, weight={weight}, нормализация={'on' if normalize else 'off'}"
        )
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    return pdf_path


def embedding_core_heatmap(label: int, Z: List[Set[int]], y: List[int], det: DetectorSpace,
                           normalize: bool = True, area_equalize: bool = False) -> np.ndarray:
    """
    Возвращает H×W карту: сумма по детекторам [частота_бита_в_классе] на их фигурах.
    """
    H, W = det.H, det.W
    idxs = [i for i, yy in enumerate(y) if int(yy) == int(label)]
    if not idxs:
        return np.zeros((H, W), dtype=np.float32)
    from collections import Counter
    cnt = Counter()
    for i in idxs:
        for b in Z[i]:
            cnt[b] += 1
    freq = {b: cnt[b] / float(len(idxs)) for b in cnt}

    heat = np.zeros((H, W), dtype=np.float32)

    for d in det.detectors:
        b = d.get("bit")
        p = freq.get(b, 0.0)
        if p <= 0.0:
            continue
        cy, cx = d["center"]

        if d.get("shape") == "ellipse":
            u1, u2 = d["u1"], d["u2"]; r1, r2 = float(d["r1"]), float(d["r2"])
            y0, y1 = max(0, int(cy - r1 - r2)), min(H - 1, int(cy + r1 + r2) + 1)
            x0, x1 = max(0, int(cx - r1 - r2)), min(W - 1, int(cx + r1 + r2) + 1)
            if y1 < y0 or x1 < x0:
                continue
            area = 0
            for yv in range(y0, y1 + 1):
                for xv in range(x0, x1 + 1):
                    vy, vx = (yv - cy), (xv - cx)
                    a = (vy * u1[0] + vx * u1[1]) / (r1 + 1e-9)
                    b2 = (vy * u2[0] + vx * u2[1]) / (r2 + 1e-9)
                    if (a * a + b2 * b2) <= 1.0:
                        area += 1
            if area == 0:
                continue
            add = (p / area) if area_equalize else p
            for yv in range(y0, y1 + 1):
                for xv in range(x0, x1 + 1):
                    vy, vx = (yv - cy), (xv - cx)
                    a = (vy * u1[0] + vx * u1[1]) / (r1 + 1e-9)
                    b2 = (vy * u2[0] + vx * u2[1]) / (r2 + 1e-9)
                    if (a * a + b2 * b2) <= 1.0:
                        heat[yv, xv] += add
        else:
            r = float(d["radius"])
            y0, y1 = max(0, int(cy - r)), min(H - 1, int(cy + r) + 1)
            x0, x1 = max(0, int(cx - r)), min(W - 1, int(cx + r) + 1)
            if y1 < y0 or x1 < x0:
                continue
            area = 0
            for yv in range(y0, y1 + 1):
                for xv in range(x0, x1 + 1):
                    if (yv - cy) ** 2 + (xv - cx) ** 2 <= r ** 2 + 1e-9:
                        area += 1
            if area == 0:
                continue
            add = (p / area) if area_equalize else p
            for yv in range(y0, y1 + 1):
                for xv in range(x0, x1 + 1):
                    if (yv - cy) ** 2 + (xv - cx) ** 2 <= r ** 2 + 1e-9:
                        heat[yv, xv] += add

    if normalize and heat.max() > 0:
        heat = heat / float(heat.max())
    return heat


def save_embedding_core_heatmap(label: int, Z: List[Set[int]], y: List[int], det: DetectorSpace,
                                out_path: str, normalize: bool = True,
                                area_equalize: bool = False) -> str:
    """
    Строит и сохраняет heatmap ядра эмбеддингов класса.
    Возвращает путь к сохранённому PNG.
    """
    core = embedding_core_heatmap(label, Z, y, det, normalize, area_equalize)
    plt.figure(figsize=(5, 5))
    plt.imshow(core, origin="lower")
    plt.title(f"Ядро эмбеддингов класса {label} (частоты детекторов)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
