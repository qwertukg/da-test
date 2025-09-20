import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
        self._code_norms: List[float] = []
        self._cell_owner_grid: List[List[Optional[int]]] = []
        self._neighbor_cache: Dict[int, Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]]] = {}

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        s = math.ceil(math.sqrt(n));
        return (s, s)

    def _neighbors(self, y: int, x: int, R: int) -> Sequence[Tuple[Tuple[int, int], float]]:
        return self._neighbor_cache[R][(y, x)]

    @staticmethod
    def _resolve_override(cell: Tuple[int, int], default_idx: Optional[int], override) -> Optional[int]:
        if override:
            for oyx, idx in override:
                if oyx == cell:
                    return idx
        return default_idx

    def _similarity(self, ia: int, ib: int, cache: Dict[Tuple[int, int], float]) -> float:
        a, b = (ia, ib) if ia <= ib else (ib, ia)
        cached = cache.get((a, b))
        if cached is not None:
            return cached
        denom = self._code_norms[a] * self._code_norms[b]
        if denom == 0.0:
            sim = 0.0
        else:
            sim = len(self._codes[a] & self._codes[b]) / denom
        cache[(a, b)] = sim
        return sim

    def _local_energy(self, yx, center_idx, R, sim_cache, override=None) -> float:
        ci = self._resolve_override(yx, center_idx, override)
        if ci is None:
            return 0.0
        energy = 0.0
        for (ny, nx), dist in self._neighbors(yx[0], yx[1], R):
            jdx = self._resolve_override((ny, nx), self._cell_owner_grid[ny][nx], override)
            if jdx is None:
                continue
            energy += self._similarity(ci, jdx, sim_cache) * dist
        return energy

    def _prepare_neighbors(self, radii: Iterable[int]) -> None:
        H, W = self.shape
        all_cells = [(y, x) for y in range(H) for x in range(W)]
        for R in set(radii):
            cache_R: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
            if R <= 0:
                for cell in all_cells:
                    cache_R[cell] = []
                self._neighbor_cache[R] = cache_R
                continue
            for y, x in all_cells:
                neighbors: List[Tuple[Tuple[int, int], float]] = []
                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        if dy == 0 and dx == 0:
                            continue
                        dist = math.hypot(dy, dx)
                        if dist > R:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            neighbors.append(((ny, nx), dist))
                cache_R[(y, x)] = neighbors
            self._neighbor_cache[R] = cache_R

    def fit(self, codes: List[Set[int]], on_epoch=None, on_swap=None):
        self._codes = codes
        n = len(codes)
        H, W = self._grid_shape(n);
        self.shape = (H, W)

        cells = [(y, x) for y in range(H) for x in range(W)]
        self.idx2cell = {}
        self._cell_owner_grid = [[None for _ in range(W)] for _ in range(H)]
        self._neighbor_cache.clear()
        self._code_norms = [math.sqrt(len(code)) if code else 0.0 for code in codes]
        self._prepare_neighbors([self.R_far, self.R_near])
        for i in range(n):
            yx = cells[i];
            self.idx2cell[i] = yx;
            self._cell_owner_grid[yx[0]][yx[1]] = i

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
                    override = ((yxa, ib), (yxb, ia))
                    e_swp = self._local_energy(yxa, ib, R, sim_cache, override=override) + \
                            self._local_energy(yxb, ia, R, sim_cache, override=override)

                    if phase == "far":
                        if e_swp + 1e-9 < e_cur:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner_grid[yxa[0]][yxa[1]] = ib
                            self._cell_owner_grid[yxb[0]][yxb[1]] = ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner_grid[yxa[0]][yxa[1]] = ib
                            self._cell_owner_grid[yxb[0]][yxb[1]] = ia
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
