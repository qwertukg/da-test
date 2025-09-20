import hashlib
from typing import List, Tuple, Set, Optional

import numpy as np
import rerun as rr
from matplotlib.colors import hsv_to_rgb
from rerun.datatypes import AnnotationInfo
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
    rr.save("../out/rks.rrd")
    if class_labels is not None:
        ann = [AnnotationInfo(id=int(i), label=str(lbl)) for i, lbl in class_labels.items()]
        rr.log("layout", rr.AnnotationContext(ann), static=True)


def rgb_from_bits(bits: Set[int]) -> Tuple[int, int, int]:
    key = ",".join(map(str, sorted(bits)))
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (h[0], h[1], h[2])


def rr_log_layout_bits(lay: Layout2D, codes: List[Set[int]], enc, tag="layout", step=0):
    N = len(codes)
    pos = np.zeros((N, 2), dtype=np.float32)
    col = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        y, x = lay.position_of(i)
        pos[i] = (x, y)
        col[i] = np.array(rgb_from_bits(codes[i]), dtype=np.uint8)
    rr.set_time("step", sequence=step)
    rr.log(f"{tag}", rr.Points2D(positions=pos, colors=col, radii=0.6))

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
):
    N = len(codes)
    if angles is not None and len(angles) != N:
        raise ValueError("длина angles должна совпадать с числом кодов")
    pos = np.zeros((N, 2), dtype=np.float32)
    col = np.zeros((N, 3), dtype=np.uint8)
    for i, code in enumerate(codes):
        y, x = lay.position_of(i)
        pos[i] = (x, y)
        if angles is not None:
            angle = angles[i]
        else:
            angle, _ = enc.code_dominant_orientation(code)
        col[i] = np.array(rgb_from_angle(angle), dtype=np.uint8)
    rr.set_time("step", sequence=step)
    rr.log(f"{tag}", rr.Points2D(positions=pos, colors=col, radii=0.6))




def run() -> None:
    rr_init("rkse+layout", spawn=True)

    def on_epoch_dots(phase, ep, lay):
        rr_log_layout_ang(
            lay,
            keyhole_codes_train,
            enc,
            tag=f"layout/{phase}",
            step=ep,
            angles=keyhole_angles_train,
        )


    X_train, X_test, y_train, y_test = load_mnist_28x28(train_limit=100, test_limit=20, seed=0)

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

    keyhole_codes_train: List[Set[int]] = []
    keyhole_meta_train: List[Tuple[int, int, float]] = []

    for img_idx, img in enumerate(X_train):
        enc_codes = enc.encode(img)
        if len(enc_codes) != len(enc.keyhole_records):
            raise RuntimeError(
                "encode() должен заполнять keyhole_records для каждой скважины"
            )
        for keyhole_idx, (angle, code) in enumerate(enc.keyhole_records):
            keyhole_codes_train.append(code)
            keyhole_meta_train.append((img_idx, keyhole_idx, angle))

    keyhole_angles_train: List[float] = [meta[2] for meta in keyhole_meta_train]

    enc.print_keyhole_records(True)

    lay = Layout2D(
        R_far=64, epochs_far=500,
        R_near=3, epochs_near=0,
        seed=123
    )

    lay.fit(keyhole_codes_train, on_epoch=on_epoch_dots)



    print("RKS layout complete!")


if __name__ == "__main__":
    run()
