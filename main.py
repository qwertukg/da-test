import numpy as np
from typing import List, Set
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from damp_light import (
    PrimaryEncoder, Layout2D, DetectorSpace, KNNJaccard,
    load_mnist_28x28, to01
)
from viz import save_embedding_core_heatmap, show_semantic_closeness


def main(digit: int = 0) -> None:
    """Запускает полный цикл обучения и оценки для одной цифры."""
    dset = "mnist"
    count = 100
    if dset == "mnist":
        X_train, X_test, y_train, y_test = load_mnist_28x28(
            train_limit=8000, test_limit=2000, seed=0
        )
        img_hw = (28, 28)
    else:
        data = load_digits()
        X = [to01(img) for img in data.images]
        y = list(data.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0, stratify=y
        )
        img_hw = (8, 8)

    enc = PrimaryEncoder(
        img_hw=img_hw, bits=8192,
        grid_g=4, grid_levels=4, grid_bits_per_cell=3,
        coarse_bits_per_cell=4,
        bright_levels=8,
        orient_on=True, orient_bins=8, orient_grid=4,
        orient_bits_per_cell=2, orient_mag_thresh=0.12,
        max_active_bits=260
    )

    codes_train = [enc.encode(img) for img in X_train]
    codes_test  = [enc.encode(img) for img in X_test]

    lay = Layout2D(R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123)
    lay.fit(codes_train)

    det = DetectorSpace(
        lay, codes_train, y_train,
        emb_bits=256,
        lam_floor=0.06,
        percentile=0.88,
        min_activated=35,
        mu=0.20,
        seeds=1200,
        min_comp=5,
        min_center_dist=1.6,
        max_detectors=512,
        seed=7
    )

    Z_train = [det.embed(c) for c in codes_train]
    Z_test  = [det.embed(c) for c in codes_test]

    clf = KNNJaccard(k=5).fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nPer-class отчёт:")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    per_digit_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nТочность по классам (recall):")
    for d, a in enumerate(per_digit_acc):
        print(f"Цифра {d}: {a:.3f}")

    def avg_bits(ZZ: List[Set[int]]) -> float:
        return float(np.mean([len(z) for z in ZZ])) if ZZ else 0.0

    print(f"Обучающая выборка: {len(X_train)}, тест: {len(X_test)}")
    print(f"Код: первичный B={enc.B}, целевая плотность ≤ {enc.max_active} бит; эмбеддинг E={det.emb_bits} бит.")
    print(f"Раскладка: сетка {lay.grid_shape()}, детекторов: {len(det.detectors)}")
    print(f"Среднее #битов в эмбеддинге: train={avg_bits(Z_train):.1f}, test={avg_bits(Z_test):.1f}")
    print(f"kNN(k=5) по Жаккару — accuracy: {acc:.4f}")

    try:
        out_png = f"{dset}_embedding_core_class-{digit}.png"
        save_embedding_core_heatmap(
            digit, Z_train, y_train, det, out_png,
            normalize=True, area_equalize=False
        )
        print(f"Сохранено 'ядро' эмбеддингов: {out_png}")
    except Exception as e:
        print(f"[WARN] Heatmap ядра не построен: {e}")

    # dset — строка с названием датасета; если её нет, поставь вручную, например "skdigits" или "mnist"
    try:
        dset_safe = dset if "dset" in locals() else "dataset"
        show_semantic_closeness(Z_train, y_train, det, dset_name=dset_safe, tau=0.25, max_pairs=5000)
    except Exception as e:
        print(f"[WARN] Не удалось построить демонстрации близости: {e}")


if __name__ == "__main__":
    # for i in range(10):
    main(1)
