import os
import pickle
from typing import List, Set, Tuple
import numpy as np

from matplotlib.colors import hsv_to_rgb

from damp_light import PrimaryEncoder, Layout2D, DetectorSpace, PrimaryEncoderKeyholeSobel

from layout_rerun import rr_init, rr_log_layout_snapshot, rr_log_swap, Layout2DRerun, rr_log_pinwheel, \
    rr_log_layout_orientation
from rkse_sobel_layout import SobelKeyholeEncoderMinimal, Layout2DNew, rr_log_layout

rr_init("digits-layout", spawn=True, class_labels={i: str(i) for i in range(10)})

CACHE_FILE = "training_cache.pkl"



def load_or_train(X_train, X_test, y_train, img_hw, cache_path: str = CACHE_FILE, dset: str = "mnist"):
    """Возвращает модели и эмбеддинги, используя сохранённое состояние при наличии."""
    cache_path = f"{dset}_{cache_path}"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            enc, lay, det, Z_train, Z_test = pickle.load(f)
    else:

        # enc = PrimaryEncoder(
        #     img_hw=img_hw, bits=8192,
        #     grid_g=4, grid_levels=4, grid_bits_per_cell=3,
        #     coarse_bits_per_cell=4,
        #     bright_levels=8,
        #     orient_on=True, orient_bins=8, orient_grid=4,
        #     orient_bits_per_cell=2, orient_mag_thresh=0.10,
        #     max_active_bits=260
        # )

        # def on_epoch_cb(phase, epoch_idx, lay):
        #     rr_log_layout_snapshot(lay, codes_train, tag=f"layout_hash/{phase}", step=epoch_idx)
        #
        _swap_counter = {"k": 0}

        def on_swap_cb(a_pos, b_pos, phase, epoch_idx, lay):
            _swap_counter["k"] += 1
            if _swap_counter["k"] % 25 == 0:
                rr_log_swap(a_pos, b_pos, phase, epoch_idx, step=_swap_counter["k"])

        enc = PrimaryEncoderKeyholeSobel(
            img_hw=(28, 28),
            bits=4096,
            keyholes_per_img=25,
            keyhole_size=5,
            orient_bins=12,
            bits_per_keyhole=6,
            mag_thresh=0.25,
            max_active_bits=None,
            deterministic=False,
            centers_mode="grid",
            unique_bits=False,
            grid_shape=(5, 5),
            seed=42
        )

        # enc = SobelKeyholeEncoderMinimal(
        #     img_hw=(28, 28),
        #     bits=2048,  # можно 1–4 тыс.
        #     keyholes_per_img=16,
        #     keyhole_size=5,
        #     orient_bins=8,
        #     mag_thresh=0.10,
        #     deterministic=True,
        #     seed=42
        # )

        codes_train = [enc.encode(img) for img in X_train]
        codes_test = [enc.encode(img) for img in X_test]

        rr_init("rkse+layout", spawn=True)
        def on_epoch(phase, ep, lay):
            rr_log_layout(lay, codes_train, tag=f"layout/{phase}", step=ep)

        lay = Layout2DNew(R_far=8, R_near=3, epochs_far=50, epochs_near=50, seed=123)
        lay.fit(codes_train, on_epoch=on_epoch)  # (on_swap можно добавить при желании)

        print("RKS done!!!")
        # exit(0)

        det = DetectorSpace(
            lay, codes_train, y_train,
            emb_bits=256,
            lam_floor=0.06,
            percentile=0.88,
            min_activated=35,
            mu=0.15,
            seeds=1200,
            min_comp=4,
            min_center_dist=1.6,
            max_detectors=260,
            seed=7
        )
        Z_train = [det.embed(c) for c in codes_train]
        Z_test = [det.embed(c) for c in codes_test]
        with open(cache_path, "wb") as f:
            pickle.dump((enc, lay, det, Z_train, Z_test), f)
    return enc, lay, det, Z_train, Z_test
