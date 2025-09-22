import hashlib
import math
from typing import List, Tuple, Set, Optional

import numpy as np
import rerun as rr
from matplotlib.colors import hsv_to_rgb
from torchvision import transforms
from torchvision.datasets import MNIST

from Layout2D import Layout2D
from RandomKeyholeSamplingEncoder import RandomKeyholeSamplingEncoder


def load_mnist_28x28(train_limit=1000, test_limit=200, seed=0):
    tfm = transforms.ToTensor()
    train_ds = MNIST(root="../data", train=True, download=True, transform=tfm)
    test_ds = MNIST(root="../data", train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(train_ds))
    test_idx = np.arange(len(test_ds))

    if train_limit and len(train_idx) > train_limit:
        train_idx = rng.choice(train_idx, size=train_limit, replace=False)
    if test_limit and len(test_idx) > test_limit:
        test_idx = rng.choice(test_idx, size=test_limit, replace=False)

    X_train = [train_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in train_idx]
    y_train = [int(train_ds[i][1]) for i in train_idx]
    X_test = [test_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in test_idx]
    y_test = [int(test_ds[i][1]) for i in test_idx]
    return X_train, X_test, y_train, y_test


def rr_init(app_name: str = "digits-layout", spawn: bool = True, class_labels=None):
    rr.init(app_name, spawn=spawn)
    # rr.save("../out/rks.rrd")


def rgb_from_bits(bits: Set[int]) -> Tuple[int, int, int]:
    key = ",".join(map(str, sorted(bits)))
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (h[0], h[1], h[2])


def rgb_from_angle(angle_rad: float):
    h = (angle_rad / (np.pi*2)) % 1.0
    r, g, b = (hsv_to_rgb([[h, 1.0, 1.0]])[0] * 255).astype(np.uint8)
    return int(r), int(g), int(b)

def rr_log_layout_ang(
    lay: Layout2D,
    codes: List[Set[int]],
    enc,
    tag: str = "layout",
    step: int = 0,
    angles: Optional[List[float]] = None,
    offset_ids: Optional[List[int]] = None,
):
    N = len(codes)
    if angles is not None and len(angles) != N:
        raise ValueError("длина angles должна совпадать с числом кодов")
    if offset_ids is not None and len(offset_ids) != N:
        raise ValueError("длина offset_ids должна совпадать с числом кодов")
    pos = np.zeros((N, 2), dtype=np.float32)
    col = np.zeros((N, 3), dtype=np.uint8)
    labels: List[str] = []
    for i, code in enumerate(codes):
        y, x = lay.position_of(i)
        pos[i] = (x, y)
        if angles is not None:
            angle = angles[i]
        else:
            angle, _ = enc.code_dominant_orientation(code)
        col[i] = np.array(rgb_from_angle(angle), dtype=np.uint8)
        angle_deg = np.degrees(angle)
        suffix = ""
        if offset_ids is not None:
            suffix = f" · копия {offset_ids[i]}"
        labels.append(f"{angle_deg:.2f}°{suffix}")
    timeline_step = step
    if isinstance(tag, str):
        phase_name = tag.rsplit("/", 1)[-1]
        if phase_name == "far":
            timeline_step = step
        elif phase_name == "near":
            timeline_step = getattr(lay, "E_far", 0) + step
    rr.set_time("step", sequence=timeline_step)
    rr.log(
        f"{tag}",
        rr.Points2D(
            positions=pos,
            colors=col,
            radii=0.6,
            labels=labels,
        ),
    )




def run() -> None:
    rr_init("rkse+layout", spawn=True)

    keyhole_codes_train: List[Set[int]] = []
    keyhole_angles_train: List[float] = []
    keyhole_offset_ids_train: List[int] = []
    keyhole_meta_train: List[Tuple[int, int, int, float]] = []

    def on_epoch_dots(phase, ep, lay):
        rr_log_layout_ang(
            lay,
            keyhole_codes_train,
            enc,
            tag=f"layout/{phase}",
            step=ep,
            angles=keyhole_angles_train,
            offset_ids=keyhole_offset_ids_train,
        )


    X_train, X_test, y_train, y_test = load_mnist_28x28(train_limit=10, test_limit=0, seed=11)

    enc = RandomKeyholeSamplingEncoder(
        img_hw=(28, 28),
        bits=256,
        keyholes_per_img=20,
        keyhole_size=5,
        seed=42,
        angle_layers=[                 # (число детекторов, полуширина дуги)
            (256, np.pi/96),          # узкий слой (точность)
            (128, np.pi/48),          # средний слой
            (32,  np.pi/12),          # широкий слой
            (8,   np.pi/4),           # очень широкий (стабильность)
        ],
        detectors_per_layer=[6, 4, 4, 2],  # суммарно 6+4+4+2 = 16 детекторов/скважину
        bits_per_detector=4,               # 16*4 = 64 бита → 64/256 = 0.25 плотность
        mag_eps=0.03, min_active_frac=0.05,
        adaptive_fill=True, adaptive_decay=0.5,
    )

    for img_idx, img in enumerate(X_train):
        enc_codes = enc.encode(img, label=y_train[img_idx])
        if len(enc_codes) != len(enc.keyhole_records):
            raise RuntimeError(
                "encode() должен заполнять keyhole_records для каждой скважины"
            )
        for record in enc.keyhole_records:
            keyhole_codes_train.append(set(record.code))
            keyhole_angles_train.append(record.angle)
            keyhole_offset_ids_train.append(record.offset_id)
            keyhole_meta_train.append((img_idx, record.keyhole_idx, record.offset_id, record.angle))

    enc.print_keyhole_records(True)

    lay = Layout2D(
        R_far=64, epochs_far=10,
        R_near=3, epochs_near=0,
        seed=123
    )

    angle_vectors = [(math.cos(angle), math.sin(angle)) for angle in keyhole_angles_train]

    lay.fit(
        keyhole_codes_train,
        aux_vectors=angle_vectors,
        aux_weight=0.2,
        on_epoch=on_epoch_dots,
    )



    print("RKS layout complete!")
    rr.log("log/summary/encoder", rr.TextLog(f"bits: {enc.B}", level=rr.TextLogLevel.INFO))
    rr.log("log/summary/encoder", rr.TextLog(f"keyhole_size: {enc.S}", level=rr.TextLogLevel.INFO))
    rr.log("log/summary/encoder", rr.TextLog(f"keyholes_per_img: {enc.K}", level=rr.TextLogLevel.INFO))
    rr.log("log/summary/layout", rr.TextLog(f"R_far: {lay.R_far}, epochs_far: {lay.E_far}", level=rr.TextLogLevel.INFO))
    rr.log("log/summary/layout", rr.TextLog(f"R_near: {lay.R_near}, epochs_near: {lay.E_near}", level=rr.TextLogLevel.INFO))


if __name__ == "__main__":
    run()
