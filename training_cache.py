import os
import pickle

from damp_light import DetectorSpace, PrimaryEncoderKeyholeSobel

from layout_rerun import rr_init
from rkse_sobel_layout import Layout2DNew, rr_log_layout

rr_init("digits-layout", spawn=True, class_labels={i: str(i) for i in range(10)})

CACHE_FILE = "training_cache.pkl"



def load_or_train(X_train, X_test, y_train, img_hw, cache_path: str = CACHE_FILE, dset: str = "mnist"):
    """Возвращает энкодер, раскладку, детекторы и кэш эмбеддингов.

    Если на диске уже есть подготовленный кэш, то модели подгружаются без
    повторного обучения. В противном случае запускается полный цикл обучения
    и результат сериализуется для последующих запусков.
    """

    cache_path = f"{dset}_{cache_path}"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    enc = PrimaryEncoderKeyholeSobel(
        img_hw=img_hw,
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
        seed=42,
    )

    codes_train = [enc.encode(img) for img in X_train]
    codes_test = [enc.encode(img) for img in X_test]

    rr_init("rkse+layout", spawn=True)

    def on_epoch(phase, ep, lay):
        rr_log_layout(lay, codes_train, tag=f"layout/{phase}", step=ep)

    lay = Layout2DNew(R_far=8, R_near=3, epochs_far=50, epochs_near=50, seed=123)
    lay.fit(codes_train, on_epoch=on_epoch)

    det = DetectorSpace(
        lay,
        codes_train,
        y_train,
        emb_bits=256,
        lam_floor=0.06,
        percentile=0.88,
        min_activated=35,
        mu=0.15,
        seeds=1200,
        min_comp=4,
        min_center_dist=1.6,
        max_detectors=260,
        seed=7,
    )

    Z_train = [det.embed(c) for c in codes_train]
    Z_test = [det.embed(c) for c in codes_test]

    with open(cache_path, "wb") as f:
        pickle.dump((enc, lay, det, Z_train, Z_test), f)

    return enc, lay, det, Z_train, Z_test
