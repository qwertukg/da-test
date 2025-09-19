from src.Layout2D import Layout2D
from src.RandomKeyholeSamplingEncoder import RandomKeyholeSamplingEncoder
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
import rerun as rr
import math, random, hashlib
from typing import List, Tuple, Dict, Set, Optional

def load_mnist_28x28(train_limit=8000, test_limit=2000, seed=0):
    """Грузит и подсэмплирует MNIST (28×28, 0..1)."""
    tfm = transforms.ToTensor()
    train_ds = MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = MNIST(root="./data", train=False, download=True, transform=tfm)

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
    """Инициализирует Rerun Viewer (опционально)."""
    rr.init(app_name, spawn=spawn)
    if class_labels is not None:
        # class_labels: {id:int -> label:str}
        from rerun.datatypes import AnnotationInfo
        ann = [AnnotationInfo(id=int(i), label=str(lbl)) for i, lbl in class_labels.items()]
        rr.log("layout", rr.AnnotationContext(ann), static=True)


def rgb_from_bits(bits: Set[int]) -> Tuple[int, int, int]:
    key = ",".join(map(str, sorted(bits)))
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (h[0], h[1], h[2])  # 0..255


def rr_log_layout(lay: Layout2D, codes: List[Set[int]], tag="layout", step=0):
    N = len(codes)
    pos = np.zeros((N, 2), dtype=np.float32)
    col = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        y, x = lay.position_of(i)
        pos[i] = (x, y)
        col[i] = np.array(rgb_from_bits(codes[i]), dtype=np.uint8)
    rr.set_time("step", sequence=step)
    rr.log(f"{tag}", rr.Points2D(positions=pos, colors=col, radii=0.6))


def main() -> None:
    X_train, X_test, y_train, y_test = load_mnist_28x28(train_limit=8000, test_limit=2000, seed=0)

    enc = RandomKeyholeSamplingEncoder(
        img_hw=(28, 28),
        bits=256,
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

    def on_epoch_dots(phase, ep, lay):
        rr_log_layout(lay, codes_train, tag=f"layout/{phase}", step=ep)

    lay = Layout2D(
        R_far=12, epochs_far=1,
        R_near=3, epochs_near=1,
        seed=123
    )
    lay.fit(codes_train, on_epoch=on_epoch_dots)

    print("RKS layout complete!")


if __name__ == "__main__":
    main()