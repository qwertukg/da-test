import math
import random
from typing import List, Tuple, Dict, Set, Optional


class Layout2D:

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
        s = math.ceil(math.sqrt(n));
        return (s, s)

    def _neighbors(self, y: int, x: int, R: int) -> List[Tuple[int, int]]:
        H, W = self.shape;
        out = []
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
        y, x = yx;
        e = 0.0
        ci = override.get((y, x), center_idx) if override else center_idx
        code_c = self._codes[ci]
        for ny, nx in self._neighbors(y, x, R):
            jdx = (override.get((ny, nx), self._cell_owner.get((ny, nx)))
                   if override else self._cell_owner.get((ny, nx)))
            if jdx is None: continue
            a, b = (ci, jdx) if ci <= jdx else (jdx, ci)
            s = sim_cache.get((a, b))
            if s is None:
                s = self.cosbin(code_c, self._codes[jdx])
                sim_cache[(a, b)] = s
            e += s * self._dist((y, x), (ny, nx))
        return e

    def fit(self, codes: List[Set[int]], on_epoch=None, on_swap=None):
        self._codes = codes
        n = len(codes)
        H, W = self._grid_shape(n);
        self.shape = (H, W)

        cells = [(y, x) for y in range(H) for x in range(W)]
        self.idx2cell = {};
        self._cell_owner = {yx: None for yx in cells}
        for i in range(n):
            yx = cells[i];
            self.idx2cell[i] = yx;
            self._cell_owner[yx] = i

        def pass_epoch(R: int, iters: int, phase: str):
            for ep in range(iters):
                occupied = list(self.idx2cell.items());
                self.rng.shuffle(occupied)
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
                        if e_swp + 1e-9 < e_cur:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)

                if on_epoch: on_epoch(phase, ep, self)

        pass_epoch(self.R_far, self.E_far, phase="far")
        pass_epoch(self.R_near, self.E_near, phase="near")
        return self

    def grid_shape(self) -> Tuple[int, int]:
        return self.shape

    def position_of(self, idx: int) -> Tuple[int, int]:
        return self.idx2cell[idx]

    def cosbin(self, a: Set[int], b: Set[int]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / math.sqrt(len(a) * len(b))
