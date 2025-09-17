import os
import pickle
import numpy as np
import rerun as rr

from damp_light import DetectorSpace, PrimaryEncoderKeyholeSobel
from layout_rerun import rr_init
from rkse_sobel_layout import Layout2DNew, rr_log_layout  # rr_log_layout больше не нужен

CACHE_FILE = "training_cache.pkl"


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПРОТОТИПОВ ----------

def _color_from_code(code) -> np.ndarray:
    """Детерминированный цветовой хеш набора битов."""
    h = hash(tuple(sorted(int(b) for b in code))) & 0xFFFFFF
    return np.array([(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF, 255], dtype=np.uint8)

def _extract_prototype_patch(img: np.ndarray, code: set[int], enc: PrimaryEncoderKeyholeSobel):
    """
    Находит «доминирующую» скважину по битам кода и возвращает:
      patch: окно S×S вокруг её центра (в тех же координатах, что кодировались),
      (cy, cx): координаты центра этой скважины.
    """
    counts = {}
    for b in code:
        info = enc.bit2info.get(int(b))
        if info is None:
            continue
        key = (info["y"], info["x"], info["bin"])
        counts[key] = counts.get(key, 0) + 1

    if counts:
        (cy, cx, _bin) = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        cy, cx = enc.H // 2, enc.W // 2  # fallback

    y0, y1, x0, x1 = _window_bounds(enc, cy, cx, enc.H, enc.W)
    patch = img[y0:y1, x0:x1]
    return patch, (cy, cx)

def _window_bounds(enc: PrimaryEncoderKeyholeSobel, cy: int, cx: int, H: int, W: int):
    r = enc.S // 2
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    return y0, y1, x0, x1

def rr_log_layout_with_keyhole_prototypes(lay: Layout2DNew,
                                          enc: PrimaryEncoderKeyholeSobel,
                                          codes, images,
                                          tag: str,
                                          max_prototypes: int = 600):
    """
    Логирует:
      - scatter точек раскладки;
      - к части точек прикрепляет patch S×S (прототип скважины) и подпись.
    """
    n = len(codes)
    # позиции (x,y) — так в Rerun Points2D привычнее читать
    positions_xy = np.array([[lay.position_of(i)[1], lay.position_of(i)[0]] for i in range(n)], dtype=np.float32)
    colors = np.stack([_color_from_code(c) for c in codes], axis=0)
    labels = [f"id={i}" for i in range(n)]

    # 1) сами точки
    rr.log(f"{tag}/points", rr.Points2D(positions=positions_xy, colors=colors, labels=labels, radii=0.5))

    # 2) прикладываем патчи (чтобы не «убить» вьюер — делаем подвыборку)
    if n == 0:
        return
    stride = max(1, n // max(1, max_prototypes))
    for i in range(0, n, stride):
        patch, (cy, cx) = _extract_prototype_patch(images[i], codes[i], enc)
        # Сам прототип
        rr.log(f"{tag}/prototypes/{i}", rr.Image(patch))
        # Небольшая подпись
        rr.log(f"{tag}/prototypes/{i}/info", rr.TextLog(f"keyhole center=({cy},{cx}), S={enc.S}"))
    # опционально можно добавить маленькие маркеры центров скважин,
    # но это уже избыточно для базовой версии.


# ---------- ОСНОВНОЙ PIPELINE С КЭШЕМ ----------

def load_or_train(X_train, X_test, y_train, img_hw, cache_path: str = CACHE_FILE, dset: str = "mnist"):
    """Возвращает энкодер, раскладку, детекторы и кэш эмбеддингов.
    Если есть кэш — грузим, иначе обучаем и сохраняем.
    """
    cache_path = f"{dset}_{cache_path}"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # --- Энкодер: RKSE (Собель + keyholes) ---
    enc = PrimaryEncoderKeyholeSobel(
        img_hw=(28, 28),
        bits=2048,
        keyholes_per_img=25,  # 5×5
        keyhole_size=9,
        orient_bins=6,
        bits_per_keyhole=12,
        mag_thresh=0.30,
        max_active_bits=None,
        deterministic=False,
        centers_mode="grid",
        unique_bits=False,
        grid_shape=(5, 5),
        seed=42
    )

    codes_train = [enc.encode(img) for img in X_train]
    codes_test  = [enc.encode(img) for img in X_test]

    rr_init("rkse+layout", spawn=True)

    # --- Раскладка с логированием прототипов после каждой эпохи ---
    def on_epoch(phase, ep, lay):
        # один тег на фазу (чтобы не было наложения разных сущностей)
        tag = f"layout/{phase}/epoch_{ep}"
        rr_log_layout_with_keyhole_prototypes(lay, enc, codes_train, X_train, tag=tag, max_prototypes=600)

    def on_epoch_dots(phase, ep, lay):
        rr_log_layout(lay, codes_train, tag=f"layout/{phase}", step=ep)

    lay = Layout2DNew(
        R_far=12, epochs_far=100,
        R_near=3, epochs_near=100,
        seed=123
    )
    lay.fit(codes_train, on_epoch=on_epoch_dots)

    print("RKS layout complete!")

    # --- Детекторы / эмбеддинги ---
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
    Z_test  = [det.embed(c) for c in codes_test]

    with open(cache_path, "wb") as f:
        pickle.dump((enc, lay, det, Z_train, Z_test), f)

    return enc, lay, det, Z_train, Z_test