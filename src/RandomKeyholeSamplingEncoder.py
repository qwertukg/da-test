import random
from typing import Dict, List, Set, Tuple

import numpy as np


class RandomKeyholeSamplingEncoder:
    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 8192,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 bits_per_keyhole: int = 1,
                 seed: int = 42,
                 ):

        self.H, self.W = img_hw
        self.B = bits
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.bits_per_keyhole = int(bits_per_keyhole)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.bit2info = {}

        self._codebook: Dict[Tuple[int, int, float], List[int]] = {}

    def encode(self, img: np.ndarray) -> Set[int]:

        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)

        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        centers = self._pick_keyholes()
        active: Set[int] = set()

        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            angle = self._dominant_angle(tile_a, tile_m)

            for bit in self._bits_for(cy, cx, angle):
                active.add(bit)

        return active

    def _pick_keyholes(self) -> List[Tuple[int, int]]:

        pad = self.S // 2

        ys = list(range(pad, self.H - pad))
        xs = list(range(pad, self.W - pad))
        all_centers = [(y, x) for y in ys for x in xs]
        if not all_centers:
            return []

        k = min(self.K, len(all_centers))
        return self.rng.sample(all_centers, k)

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        r = self.S // 2
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

    def _bits_for(self, y: int, x: int, angle: float) -> List[int]:
        key = (int(y), int(x), float(angle))
        if key not in self._codebook:
            self._codebook[key] = self._rand_bits(self.bits_per_keyhole)
            for bit in self._codebook[key]:
                self.bit2info.setdefault(bit, {"y": int(y), "x": int(x), "angle": float(angle)})
        return self._codebook[key]

    def code_dominant_orientation(self, code: set[int]) -> tuple[float, float]:

        if not code:
            return 0.0, 0.0

        angles: List[float] = []
        for bit in code:
            info = self.bit2info.get(int(bit))
            if info is None:
                continue
            angles.append(float(info.get("angle", 0.0)))

        if not angles:
            return 0.0, 0.0

        vectors = np.exp(1j * np.array(angles, dtype=np.float64))
        vec_sum = vectors.sum()
        angle = float(np.angle(vec_sum) % (2 * np.pi))
        selectivity = float(abs(vec_sum) / len(vectors))
        return angle, selectivity

    def _rand_bits(self, k: int) -> List[int]:

        return self.rng.sample(range(self.B), k)

    @staticmethod
    def _sobel(img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)
        gx = RandomKeyholeSamplingEncoder._conv2_same(img01, kx)
        gy = RandomKeyholeSamplingEncoder._conv2_same(img01, ky)
        return gx, gy

    @staticmethod
    def _conv2_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
        H, W = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                patch = pad[y:y + k.shape[0], x:x + k.shape[1]]
                out[y, x] = float(np.sum(patch * k))
        return out

    @staticmethod
    def _dominant_angle(tile_a: np.ndarray, tile_m: np.ndarray) -> float:
        weights = tile_m.astype(np.float64)
        total = float(weights.sum())
        if total <= 1e-8:
            return 0.0
        vectors = np.exp(1j * tile_a.astype(np.float64)) * weights
        vec_sum = vectors.sum()
        if vec_sum == 0:
            return 0.0
        angle = float(np.angle(vec_sum) % (2 * np.pi))
        return angle
