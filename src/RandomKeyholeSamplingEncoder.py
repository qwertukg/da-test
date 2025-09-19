import hashlib
import random
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


class RandomKeyholeSamplingEncoder:
    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 8192,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 orient_bins: int = 12,
                 bits_per_keyhole: int = 1,
                 mag_thresh: float = 0.10,
                 max_active_bits: Optional[int] = None,
                 deterministic: bool = False,
                 seed: int = 42,
                 unique_bits=True,
                 centers_mode: str = "grid",
                 grid_shape: Optional[Tuple[int, int]] = None):

        self.H, self.W = img_hw
        self.B = bits
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.orient_bins = int(orient_bins)
        self.bits_per_keyhole = int(bits_per_keyhole)
        self.mag_thresh = float(mag_thresh)
        self.max_active_bits = max_active_bits if max_active_bits is None else int(max_active_bits)

        self.deterministic = bool(deterministic)
        self.centers_mode = centers_mode
        self.grid_shape = grid_shape
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.unique_bits = unique_bits
        self.bit2info = {}
        self._next_bit = 0

        self._codebook: Dict[Tuple[int, int, int], List[int]] = {}
        self._used_bits: Set[int] = set()
        self.bit2yxb: Dict[int, Tuple[int, int, int]] = {}

    def encode(self, img: np.ndarray) -> Set[int]:

        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)

        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        centers = self._pick_keyholes(img)
        active: Set[int] = set()
        kept_any = False

        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            if float(np.mean(tile_m)) < self.mag_thresh:
                continue

            bidx = np.floor((tile_a / (2 * np.pi)) * self.orient_bins).astype(int) % self.orient_bins

            hist = np.zeros(self.orient_bins, dtype=np.float32)
            for b in range(self.orient_bins):
                hist[b] = float(tile_m[bidx == b].sum())
            b_max = int(np.argmax(hist))

            for bit in self._bits_for(cy, cx, b_max):
                active.add(bit)

            kept_any = True

        if not kept_any and centers:
            strengths = []
            for (cy, cx) in centers:
                y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
                strengths.append((float(np.mean(mnorm[y0:y1, x0:x1])), (cy, cx)))
            strengths.sort(reverse=True)
            (cy, cx) = strengths[0][1]
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]
            bidx = np.floor((tile_a / (2 * np.pi)) * self.orient_bins).astype(int) % self.orient_bins
            hist = np.zeros(self.orient_bins, dtype=np.float32)
            for b in range(self.orient_bins):
                hist[b] = float(tile_m[bidx == b].sum())
            b_max = int(np.argmax(hist))
            for bit in self._bits_for(cy, cx, b_max):
                active.add(bit)

        if self.max_active_bits is not None and len(active) > self.max_active_bits:
            active = set(sorted(active)[: self.max_active_bits])

        return active

    def _pick_keyholes(self, img: np.ndarray) -> List[Tuple[int, int]]:

        pad = self.S // 2

        if self.centers_mode == "grid":

            if self.grid_shape is None:

                side = int(round(np.sqrt(self.K)))
                gh, gw = max(1, side), max(1, side)
            else:
                gh, gw = self.grid_shape
                gh = max(1, int(gh));
                gw = max(1, int(gw))

            ys = np.linspace(pad, self.H - 1 - pad, gh).round().astype(int)
            xs = np.linspace(pad, self.W - 1 - pad, gw).round().astype(int)
            centers = [(int(y), int(x)) for y in ys for x in xs]

            if len(centers) > self.K:
                centers = centers[: self.K]
            return centers

        ys = list(range(pad, self.H - pad))
        xs = list(range(pad, self.W - pad))
        all_centers = [(y, x) for y in ys for x in xs]
        if not all_centers:
            return []

        if self.deterministic:
            h = hashlib.sha256(img.astype(np.float32).tobytes()).digest()
            seed_img = int.from_bytes(h[:8], "little") ^ self.seed
            rng = random.Random(seed_img)
        else:
            rng = self.rng

        k = min(self.K, len(all_centers))
        return rng.sample(all_centers, k)

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        r = self.S // 2
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

    def _bits_for(self, y: int, x: int, b: int) -> List[int]:
        key = (int(y), int(x), int(b))
        if key not in self._codebook:
            if self.unique_bits:
                ids = []
                for _ in range(self.bits_per_keyhole):
                    if self._next_bit >= self.B:
                        break
                    bit = self._next_bit
                    self._next_bit += 1
                    ids.append(bit)
                    self.bit2info[bit] = {"y": int(y), "x": int(x), "bin": int(b)}
                self._codebook[key] = ids
            else:
                self._codebook[key] = self._rand_bits(self.bits_per_keyhole)
                for bit in self._codebook[key]:
                    self.bit2info.setdefault(bit, {"y": int(y), "x": int(x), "bin": int(b)})
        return self._codebook[key]

    def code_dominant_orientation(self, code: set[int]) -> tuple[float, float]:

        if not code:
            return 0.0, 0.0

        hist = np.zeros(self.orient_bins, dtype=np.float32)
        for bit in code:
            info = self.bit2info.get(int(bit))
            if info is None:
                continue
            hist[info["bin"]] += 1.0
        if hist.sum() <= 0:
            return 0.0, 0.0
        b_max = int(hist.argmax())

        angle = (b_max + 0.5) * (np.pi / self.orient_bins)
        selectivity = float(hist[b_max] / (hist.sum() + 1e-9))
        return angle, selectivity

    def _alloc_unique_bits(self, k: int) -> List[int]:

        out: List[int] = []
        tries = 0
        while len(out) < k:
            tries += 1

            if tries > 10000 and (self.B - len(self._used_bits)) < (k - len(out)):
                raise RuntimeError(
                    "Недостаточно свободных битов при unique_bits=True: "
                    f"нужно ещё {k - len(out)}, свободно {self.B - len(self._used_bits)}."
                )
            cand = self.rng.randrange(self.B)
            if cand in self._used_bits:
                continue
            self._used_bits.add(cand)
            out.append(cand)
        return out

    def _rand_bits(self, k: int) -> List[int]:

        return self.rng.sample(range(self.B), k)

    # ---------- Собель 3×3 ----------

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

    # ---------- Вспомогательные методы (необязательные) ----------

    def get_keyhole_centers_grid(self) -> List[Tuple[int, int]]:
        """Вернёт центры скважин для текущей конфигурации в режиме 'grid' (полезно для отладки)."""
        pad = self.S // 2
        if self.grid_shape is None:
            side = int(round(np.sqrt(self.K)))
            gh, gw = max(1, side), max(1, side)
        else:
            gh, gw = self.grid_shape
            gh = max(1, int(gh));
            gw = max(1, int(gw))
        ys = np.linspace(pad, self.H - 1 - pad, gh).round().astype(int)
        xs = np.linspace(pad, self.W - 1 - pad, gw).round().astype(int)
        centers = [(int(y), int(x)) for y in ys for x in xs]
        return centers[: self.K]
