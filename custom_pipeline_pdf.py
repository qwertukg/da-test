#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_pipeline_pdf.py — делает PDF в стиле mnist_pipeline_* для произвольных картинок.

- Обучает пайплайн на MNIST (как в discrete_digits_final.py)
- Препроцессит изображения в формат 28×28 [0..1]
- Строит по-страничную визуализацию через viz.visualize_pipeline(...)
- Сохраняет в один PDF: custom_pipeline_3.pdf
"""

import os, sys, math, numpy as np
from typing import List
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_digits

# Импортируем классы/функции из твоего основного файла
from damp_light import (
    PrimaryEncoder, Layout2D, DetectorSpace, KNNJaccard,
    load_mnist_28x28, jaccard, cosbin, to01
)
from sklearn.model_selection import train_test_split

# Функции визуализаций из твоего модуля viz
from viz import visualize_pipeline
# ВЕРХ ФАЙЛА: добавь универсальный препроцесс под любую (H,W)
from typing import Tuple

def preprocess_to_hw(path: str, hw: Tuple[int, int]) -> np.ndarray:
    """Готовит картинку к целевому размеру hw=(H,W): float [0..1], белый знак на чёрном, центрирование.
       Масштаб внутри холста берём как ~20/28 от минимального измерения (как у MNIST)."""
    H, W = hw
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Инвертируем, если фон светлый
    if float(arr.mean()) > 127.0:
        arr = 255.0 - arr

    # Обрезка по непустой области
    nz = np.argwhere(arr > 0)
    if nz.size > 0:
        (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
        arr = arr[y0:y1, x0:x1]

    # Вписываем во "внутренний" квадрат (как 20/28 у MNIST)
    inside = max(1, int(round(min(H, W) * 20 / 28)))
    s = inside / max(1, max(arr.shape))
    nh, nw = max(1, int(round(arr.shape[0] * s))), max(1, int(round(arr.shape[1] * s)))
    arr = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.float32)

    # Центрируем на холст H×W
    canvas = np.zeros((H, W), dtype=np.float32)
    y, x = (H - nh) // 2, (W - nw) // 2
    canvas[y:y+nh, x:x+nw] = arr

    # Нормализация
    canvas /= 255.0
    return canvas


def preprocess_to_28x28(path: str) -> np.ndarray:
    """Готовит картинку к формату MNIST: 28×28 float [0..1], белая фигура на чёрном фоне, центрирована."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32)

    # Если фон светлый (средняя яркость > 127), инвертируем (полезно для чёрного знака на белом)
    if float(arr.mean()) > 127.0:
        arr = 255.0 - arr

    # Обрезка по непустой области
    nz = np.argwhere(arr > 0)
    if nz.size > 0:
        (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
        arr = arr[y0:y1, x0:x1]

    # Вписываем в 20×20 с сохранением пропорций
    h, w = arr.shape
    s = 20.0 / max(1, max(h, w))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    arr = np.array(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.float32)

    # Центрируем на холст 28×28
    canvas = np.zeros((28, 28), dtype=np.float32)
    y, x = (28 - nh) // 2, (28 - nw) // 2
    canvas[y:y+nh, x:x+nw] = arr

    # Нормализация [0..1]
    canvas /= 255.0
    return canvas


def build_pipeline(train_limit=8000, seed=0):
    """Строит весь пайплайн (как в main): Encoder -> Layout -> Detectors -> Embeddings -> kNN."""

    # --- данные ---
    data = load_digits()
    X = [to01(img) for img in data.images]
    y = list(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0, stratify=y
    )

    # 1) Данные MNIST
    # X_train, _, y_train, _ = load_mnist_28x28(train_limit=train_limit, test_limit=0, seed=seed)

    # 2) Энкодер
    enc = PrimaryEncoder(
        img_hw=(8, 8), bits=8192,
        grid_g=4, grid_levels=4, grid_bits_per_cell=3,
        coarse_bits_per_cell=4,
        bright_levels=8,
        orient_on=True, orient_bins=8, orient_grid=4,
        orient_bits_per_cell=2, orient_mag_thresh=0.12,
        max_active_bits=260
    )
    codes_train = [enc.encode(img) for img in X_train]

    # 3) Раскладка
    lay = Layout2D(R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123).fit(codes_train)

    # 4) Детекторы
    det = DetectorSpace(
        lay, codes_train, y_train,
        emb_bits=256,
        lam_floor=0.06,
        percentile=0.88,
        min_activated=35,
        mu=0.20,
        seeds=min(1200, len(codes_train)),
        min_comp=5,
        min_center_dist=1.6,
        max_detectors=512,
        seed=7
    )

    # 5) Эмбеддинги и kNN
    Z_train = [det.embed(c) for c in codes_train]
    clf = KNNJaccard(k=5).fit(Z_train, y_train)
    return enc, lay, det, clf


def images_to_pdf(image_paths: List[str], out_pdf: str, enc: PrimaryEncoder, lay: Layout2D, det: DetectorSpace, clf: KNNJaccard):
    """Строит PDF: на каждой странице полная визуализация конвейера для входной картинки."""
    with PdfPages(out_pdf) as pdf:
        for p in image_paths:
            if not os.path.exists(p):
                print(f"[WARN] Файл не найден: {p} — пропускаю")
                continue
            # img = preprocess_to_28x28(p)
            img = preprocess_to_hw(p, (enc.H, enc.W))

            # Предсказание (для подписи)
            code = enc.encode(img)
            z = det.embed(code)
            pred = int(clf.predict([z])[0])

            title = f"{os.path.basename(p)} → предсказано: {pred}"

            # Визуализация полным пайплайном
            try:
                fig = visualize_pipeline(img=img, enc=enc, lay=lay, det=det, clf=clf, true_label=None, title=title)
            except TypeError:
                # если в твоём viz.true_label обязателен другой тип или нет такого параметра
                fig = visualize_pipeline(img=img, enc=enc, lay=lay, det=det, clf=clf, title=title)

            pdf.savefig(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
    print(f"[OK] Сохранено в PDF: {out_pdf}")


def main():
    # Список по умолчанию — как просили
    default_images = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png"]
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else default_images
    out_pdf = "custom_pipeline.pdf"

    enc, lay, det, clf = build_pipeline(train_limit=8000, seed=0)
    images_to_pdf(image_paths, out_pdf, enc, lay, det, clf)


if __name__ == "__main__":
    main()
