# rkse_sobel_layout.py
import math, random, hashlib
from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from dl_utils import cosbin


# ========== вспомогательные ==========
def rgb_from_bits(bits: Set[int]) -> Tuple[int, int, int]:
    key = ",".join(map(str, sorted(bits)))
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return (h[0], h[1], h[2])  # 0..255

# ========== 1) Минимальный Sobel-keyhole энкодер ==========
class SobelKeyholeEncoderMinimal:
    """
    K случайных «скважин» S×S; в каждой — Собель (gx, gy), гистограмма углов,
    берём доминирующий bin -> один бит. Бит = hash(y,x,bin,seed) % bits.
    """
    def __init__(self,
                 img_hw=(28, 28),
                 bits=1024,
                 keyholes_per_img=16,
                 keyhole_size=5,
                 orient_bins=8,
                 mag_thresh=0.10,
                 deterministic=True,
                 seed=42):
        assert keyhole_size % 2 == 1, "keyhole_size должен быть нечётным"
        self.H, self.W = img_hw
        self.B = bits
        self.K = keyholes_per_img
        self.S = keyhole_size
        self.BINS = orient_bins
        self.TH = mag_thresh
        self.det = deterministic
        self.seed = seed
        self.rng = random.Random(seed)
        r = self.S // 2
        self.centers = [(y, x) for y in range(r, self.H - r) for x in range(r, self.W - r)]

    # --- публичное ---
    def encode(self, img: np.ndarray) -> Set[int]:
        H, W = img.shape
        assert (H, W) == (self.H, self.W), f"ожидался {self.H}x{self.W}, получен {H}x{W}"
        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag
        ang = (np.arctan2(gy, gx) + np.pi)  # [0, 2π)

        rng = self._rng_for_img(img)
        centers = rng.sample(self.centers, k=min(self.K, len(self.centers)))

        active: Set[int] = set()
        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._win(cy, cx)
            tile_m = mnorm[y0:y1, x0:x1]
            if float(np.mean(tile_m)) < self.TH:
                continue
            tile_a = ang[y0:y1, x0:x1]
            bidx = np.floor((tile_a / (2*np.pi)) * self.BINS).astype(int) % self.BINS
            # весим по tile_m
            hist = np.zeros(self.BINS, dtype=np.float32)
            for b in range(self.BINS):
                hist[b] = float(tile_m[bidx == b].sum())
            bmax = int(np.argmax(hist))
            bit = self._bit_for(cy, cx, bmax)
            active.add(bit)

        # fallback: если ничего не прошло порог, самый «сильный» центр всё же активируем
        if not active and centers:
            strengths = [(float(np.mean(mnorm[self._win(*c)[0]:self._win(*c)[1],
                                                       self._win(*c)[2]:self._win(*c)[3]])), c)
                         for c in centers]
            strengths.sort(reverse=True)
            cy, cx = strengths[0][1]
            y0, y1, x0, x1 = self._win(cy, cx)
            tile_a = ang[y0:y1, x0:x1]
            tile_m = mnorm[y0:y1, x0:x1]
            bidx = np.floor((tile_a / (2*np.pi)) * self.BINS).astype(int) % self.BINS
            hist = np.zeros(self.BINS, dtype=np.float32)
            for b in range(self.BINS):
                hist[b] = float(tile_m[bidx == b].sum())
            bmax = int(np.argmax(hist))
            active.add(self._bit_for(cy, cx, bmax))

        return active

    # --- внутренние ---
    def _rng_for_img(self, img: np.ndarray) -> random.Random:
        if not self.det:
            return self.rng
        h = hashlib.sha256(img.astype(np.float32).tobytes()).digest()
        seed_img = int.from_bytes(h[:8], "little") ^ self.seed
        return random.Random(seed_img)

    def _win(self, cy: int, cx: int):
        r = self.S // 2
        return cy - r, cy + r + 1, cx - r, cx + r + 1

    def _bit_for(self, y: int, x: int, b: int) -> int:
        h = hashlib.sha256(f"{y},{x},{b},{self.seed}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % self.B

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

    @classmethod
    def _sobel(cls, img01: np.ndarray):
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float32)
        gx = cls._conv2_same(img01, kx)
        gy = cls._conv2_same(img01, ky)
        return gx, gy

# ========== 2) Layout2D (FAR → NEAR) ==========
class Layout2DNew:
    """
    Два этапа:
      - FAR (большой радиус): принимаем свап, если энергия ПОСЛЕ меньше (минимизация).
      - NEAR (малый радиус): принимаем свап, если энергия ПОСЛЕ больше (максимизация).
    """
    def __init__(self, R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123):
        self.R_far = R_far
        self.R_near = R_near
        self.E_far = epochs_far
        self.E_near = epochs_near
        self.rng = random.Random(seed)
        self.shape: Tuple[int, int] = (0, 0)
        self.idx2cell: Dict[int, Tuple[int, int]] = {}
        self._codes: List[Set[int]] = []
        self._cell_owner: Dict[Tuple[int, int], Optional[int]] = {}

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        s = math.ceil(math.sqrt(n)); return (s, s)

    def _neighbors(self, y: int, x: int, R: int) -> List[Tuple[int, int]]:
        H, W = self.shape; out = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W: out.append((ny, nx))
        return out

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _local_energy(self, yx, center_idx, R, sim_cache, override=None) -> float:
        y, x = yx; e = 0.0
        ci = override.get((y, x), center_idx) if override else center_idx
        code_c = self._codes[ci]
        for ny, nx in self._neighbors(y, x, R):
            jdx = (override.get((ny, nx), self._cell_owner.get((ny, nx)))
                   if override else self._cell_owner.get((ny, nx)))
            if jdx is None: continue
            a, b = (ci, jdx) if ci <= jdx else (jdx, ci)
            s = sim_cache.get((a, b))
            if s is None:
                s = cosbin(code_c, self._codes[jdx])
                sim_cache[(a, b)] = s
            e += s * self._dist((y, x), (ny, nx))
        return e

    def fit(self, codes: List[Set[int]], on_epoch=None, on_swap=None):
        self._codes = codes
        n = len(codes)
        H, W = self._grid_shape(n); self.shape = (H, W)

        # начальная укладка — построчно
        cells = [(y, x) for y in range(H) for x in range(W)]
        self.idx2cell = {}; self._cell_owner = {yx: None for yx in cells}
        for i in range(n):
            yx = cells[i]; self.idx2cell[i] = yx; self._cell_owner[yx] = i

        def pass_epoch(R: int, iters: int, phase: str):
            for ep in range(iters):
                occupied = list(self.idx2cell.items()); self.rng.shuffle(occupied)
                pairs = []
                for i in range(0, len(occupied) - 1, 2):
                    (ia, yxa), (ib, yxb) = occupied[i], occupied[i + 1]
                    pairs.append((ia, yxa, ib, yxb))
                sim_cache: Dict[Tuple[int, int], float] = {}
                for ia, yxa, ib, yxb in pairs:
                    e_cur = self._local_energy(yxa, ia, R, sim_cache) + \
                            self._local_energy(yxb, ib, R, sim_cache)
                    override = {yxa: ib, yxb: ia}
                    e_swp = self._local_energy(yxa, ib, R, sim_cache, override=override) + \
                            self._local_energy(yxb, ia, R, sim_cache, override=override)

                    if phase == "far":
                        # минимизация энергии
                        if e_swp + 1e-9 < e_cur:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:  # phase == "near"
                        # максимизация энергии
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)

                if on_epoch: on_epoch(phase, ep, self)

        pass_epoch(self.R_far,  self.E_far,  phase="far")   # FAR: минимизация
        pass_epoch(self.R_near, self.E_near, phase="near")  # NEAR: максимизация
        return self

    def grid_shape(self) -> Tuple[int, int]: return self.shape
    def position_of(self, idx: int) -> Tuple[int, int]: return self.idx2cell[idx]

# ========== 3) Rerun-хелперы (минимум) ==========
def rr_init(app_name="rkse-layout", spawn=True):
    import rerun as rr
    try: rr.init(app_name)
    except Exception: pass
    if spawn:
        try: rr.spawn()
        except Exception:
            try: rr.connect()
            except Exception: pass

def rr_log_layout(lay: Layout2DNew, codes: List[Set[int]], tag="layout", step=0):
    import rerun as rr
    H, W = lay.grid_shape(); N = len(codes)
    pos = np.zeros((N, 2), dtype=np.float32)
    col = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        y, x = lay.position_of(i)
        pos[i] = (x, y)
        col[i] = np.array(rgb_from_bits(codes[i]), dtype=np.uint8)
    rr.set_time_sequence("step", step)
    rr.log(f"{tag}", rr.Points2D(positions=pos, colors=col, radii=0.6))