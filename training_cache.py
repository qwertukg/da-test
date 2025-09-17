"""Кэш обучения для пайплайна DAMP-light."""

import os
import pickle
from typing import List, Sequence, Set, Tuple

import numpy as np

from damp_light import DetectorSpace, Layout2D, PrimaryEncoder

CACHE_FILE = "training_cache.pkl"


def _encode_many(encoder: PrimaryEncoder, images: Sequence[np.ndarray]) -> List[Set[int]]:
    return [encoder.encode(img) for img in images]


def load_or_train(
    X_train: Sequence[np.ndarray],
    X_test: Sequence[np.ndarray],
    y_train: Sequence[int],
    img_hw: Tuple[int, int],
    cache_path: str = CACHE_FILE,
    dset: str = "mnist",
):
    """Возвращает обученные модели и эмбеддинги, используя кэш при наличии."""
    cache_path = f"{dset}_{cache_path}"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    encoder = PrimaryEncoder(
        img_hw=img_hw,
        bits=8192,
        grid_g=4,
        grid_levels=4,
        grid_bits_per_cell=3,
        coarse_bits_per_cell=4,
        bright_levels=8,
        orient_on=True,
        orient_bins=8,
        orient_grid=4,
        orient_bits_per_cell=2,
        orient_mag_thresh=0.10,
        max_active_bits=260,
    )

    codes_train = _encode_many(encoder, X_train)
    codes_test = _encode_many(encoder, X_test)

    layout = Layout2D(R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123)
    layout.fit(codes_train)

    detector = DetectorSpace(
        layout,
        codes_train,
        list(y_train),
        emb_bits=256,
        lam_floor=0.06,
        percentile=0.88,
        min_activated=35,
        mu=0.15,
        seeds=min(1200, len(codes_train)),
        min_comp=4,
        min_center_dist=1.6,
        max_detectors=260,
        seed=7,
    )

    Z_train = [detector.embed(code) for code in codes_train]
    Z_test = [detector.embed(code) for code in codes_test]

    with open(cache_path, "wb") as f:
        pickle.dump((encoder, layout, detector, Z_train, Z_test), f)

    return encoder, layout, detector, Z_train, Z_test
