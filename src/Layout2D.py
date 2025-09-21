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
            ca = self._codes[a]
            cb = self._codes[b]
            if len(ca) > len(cb):
                ca, cb = cb, ca
            inter = sum(1 for item in ca if item in cb)
            sim = inter / denom
        cache[(a, b)] = sim
        return sim

    def _local_energy(self, yx, center_idx, R, sim_cache, override=None) -> float:
        override_map = None
        if override:
            if isinstance(override, dict):
                override_map = override
            else:
                override_map = dict(override)
        if override_map is not None:
            ci = override_map.get(yx, center_idx)
        else:
            ci = center_idx
        if ci is None:
            return 0.0
        energy = 0.0
        owner_grid = self._cell_owner_grid
        neighbors = self._neighbor_cache[R][yx]
        if override_map is not None:
            for (ny, nx), dist in neighbors:
                jdx = override_map.get((ny, nx), owner_grid[ny][nx])
                if jdx is None:
                    continue
                energy += self._similarity(ci, jdx, sim_cache) * dist
            return energy

        for (ny, nx), dist in neighbors:
            jdx = owner_grid[ny][nx]
            if jdx is None:
                continue
            energy += self._similarity(ci, jdx, sim_cache) * dist
        return energy

    def _prepare_neighbors(self, radii: Iterable[int]) -> None:
        H, W = self.shape
        if H == 0 or W == 0:
            return
        for R in set(radii):
            cache_R: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
            if R <= 0:
                for y in range(H):
                    for x in range(W):
                        cache_R[(y, x)] = []
                self._neighbor_cache[R] = cache_R
                continue

            offsets: List[Tuple[int, int, float]] = []
            RR = R * R
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if dy == 0 and dx == 0:
                        continue
                    dist_sq = dy * dy + dx * dx
                    if dist_sq > RR:
                        continue
                    offsets.append((dy, dx, math.sqrt(dist_sq)))

            for y in range(H):
                for x in range(W):
                    neighbors: List[Tuple[Tuple[int, int], float]] = []
                    for dy, dx, dist in offsets:
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
            rng = self.rng
            idx2cell_local = self.idx2cell
            grid = self._cell_owner_grid
            local_energy = self._local_energy
            for ep in range(iters):
                indices = list(idx2cell_local)
                rng.shuffle(indices)
                sim_cache: Dict[Tuple[int, int], float] = {}
                for i in range(0, len(indices) - 1, 2):
                    ia = indices[i]
                    ib = indices[i + 1]
                    yxa = idx2cell_local[ia]
                    yxb = idx2cell_local[ib]

                    e_cur = local_energy(yxa, ia, R, sim_cache) + \
                            local_energy(yxb, ib, R, sim_cache)
                    override_map = {yxa: ib, yxb: ia}
                    e_swp = local_energy(yxa, ib, R, sim_cache, override=override_map) + \
                            local_energy(yxb, ia, R, sim_cache, override=override_map)

                    if phase == "far":
                        if e_swp + 1e-9 < e_cur:
                            idx2cell_local[ia], idx2cell_local[ib] = yxb, yxa
                            grid[yxa[0]][yxa[1]] = ib
                            grid[yxb[0]][yxb[1]] = ia
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            idx2cell_local[ia], idx2cell_local[ib] = yxb, yxa
                            grid[yxa[0]][yxa[1]] = ib
                            grid[yxb[0]][yxb[1]] = ia
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
