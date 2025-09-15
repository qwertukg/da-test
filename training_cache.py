import os
import pickle
from typing import List, Set, Tuple
from damp_light import PrimaryEncoder, Layout2D, DetectorSpace

CACHE_FILE = "training_cache.pkl"

def load_or_train(X_train, X_test, y_train, img_hw, cache_path: str = CACHE_FILE):
    """Возвращает модели и эмбеддинги, используя сохранённое состояние при наличии."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            enc, lay, det, Z_train, Z_test = pickle.load(f)
    else:
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
        codes_test = [enc.encode(img) for img in X_test]
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
        Z_test = [det.embed(c) for c in codes_test]
        with open(cache_path, "wb") as f:
            pickle.dump((enc, lay, det, Z_train, Z_test), f)
    return enc, lay, det, Z_train, Z_test
