import math
import multiprocessing as mp
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
        self._code_bitmasks: List[int] = []
        self._cell_owner_grid: List[List[Optional[int]]] = []
        self._neighbor_cache: Dict[int, Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]]] = {}
        self._aux_vecs: Optional[List[Optional[Tuple[float, ...]]]] = None
        self._aux_weight: float = 0.0

    def _make_worker_snapshot(self, R: int) -> Dict[str, object]:
        return {
            "neighbors": self._neighbor_cache[R],
            "cell_owner_grid": tuple(tuple(row) for row in self._cell_owner_grid),
            "code_bitmasks": self._code_bitmasks,
            "code_norms": self._code_norms,
            "aux_vecs": self._aux_vecs,
            "aux_weight": self._aux_weight,
        }

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        s = math.ceil(math.sqrt(n));
        return (s, s)

    @staticmethod
    def _code_to_bitmask(code: Set[int]) -> int:
        bitmask = 0
        for bit in code:
            bitmask |= 1 << bit
        return bitmask

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
            inter = (self._code_bitmasks[a] & self._code_bitmasks[b]).bit_count()
            sim = inter / denom
        if self._aux_vecs is not None and self._aux_weight > 0.0:
            va = self._aux_vecs[a]
            vb = self._aux_vecs[b]
            if va is not None and vb is not None:
                dot = sum(ax * bx for ax, bx in zip(va, vb))
                sim += self._aux_weight * ((dot + 1.0) * 0.5)
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

    def fit(self,
            codes: List[Set[int]],
            *,
            aux_vectors: Optional[Sequence[Sequence[float]]] = None,
            aux_weight: float = 0.0,
            on_epoch=None,
            on_swap=None):
        self._codes = codes
        self._code_bitmasks = [self._code_to_bitmask(code) for code in codes]
        n = len(codes)
        H, W = self._grid_shape(n);
        self.shape = (H, W)

        cells = [(y, x) for y in range(H) for x in range(W)]
        self.idx2cell = {}
        self._cell_owner_grid = [[None for _ in range(W)] for _ in range(H)]
        self._neighbor_cache.clear()
        self._code_norms = [math.sqrt(len(code)) if code else 0.0 for code in codes]
        self._aux_vecs = None
        self._aux_weight = 0.0
        if aux_vectors is not None:
            if len(aux_vectors) != n:
                raise ValueError("длина aux_vectors должна совпадать с числом кодов")
            normed: List[Optional[Tuple[float, ...]]] = []
            for vec in aux_vectors:
                arr = tuple(float(v) for v in vec)
                norm = math.sqrt(sum(v * v for v in arr))
                if norm == 0.0:
                    normed.append(None)
                else:
                    normed.append(tuple(v / norm for v in arr))
            self._aux_vecs = normed
            self._aux_weight = float(aux_weight)
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
                energies: List[Tuple[float, float]] = []
                worker_count = 0
                if pairs:
                    try:
                        worker_count = mp.cpu_count() or 1
                    except NotImplementedError:
                        worker_count = 1
                    worker_count = min(worker_count, len(pairs))
                if worker_count > 1:
                    snapshot = self._make_worker_snapshot(R)
                    with mp.Pool(processes=worker_count, initializer=_init_energy_worker, initargs=(snapshot,)) as pool:
                        energies = pool.map(_energy_for_pair, pairs)
                else:
                    sim_cache: Dict[Tuple[int, int], float] = {}
                    for ia, yxa, ib, yxb in pairs:
                        e_cur = self._local_energy(yxa, ia, R, sim_cache) + \
                                self._local_energy(yxb, ib, R, sim_cache)
                        override = ((yxa, ib), (yxb, ia))
                        e_swp = self._local_energy(yxa, ib, R, sim_cache, override=override) + \
                                self._local_energy(yxb, ia, R, sim_cache, override=override)
                        energies.append((e_cur, e_swp))

                for (ia, yxa, ib, yxb), (e_cur, e_swp) in zip(pairs, energies):
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
        mask_a = self._code_to_bitmask(a)
        mask_b = self._code_to_bitmask(b)
        inter = (mask_a & mask_b).bit_count()
        return inter / math.sqrt(len(a) * len(b))


_ENERGY_WORKER_STATE: Dict[str, object] = {}


def _init_energy_worker(snapshot: Dict[str, object]) -> None:
    global _ENERGY_WORKER_STATE
    _ENERGY_WORKER_STATE = {
        "neighbors": snapshot["neighbors"],
        "cell_owner_grid": snapshot["cell_owner_grid"],
        "code_bitmasks": snapshot["code_bitmasks"],
        "code_norms": snapshot["code_norms"],
        "aux_vecs": snapshot["aux_vecs"],
        "aux_weight": snapshot["aux_weight"],
        "sim_cache": {},
    }


def _similarity_from_snapshot(ia: int, ib: int) -> float:
    state = _ENERGY_WORKER_STATE
    cache: Dict[Tuple[int, int], float] = state["sim_cache"]  # type: ignore[assignment]
    a, b = (ia, ib) if ia <= ib else (ib, ia)
    cached = cache.get((a, b))
    if cached is not None:
        return cached
    code_bitmasks: Sequence[int] = state["code_bitmasks"]  # type: ignore[assignment]
    code_norms: Sequence[float] = state["code_norms"]  # type: ignore[assignment]
    denom = code_norms[a] * code_norms[b]
    if denom == 0.0:
        sim = 0.0
    else:
        inter = (code_bitmasks[a] & code_bitmasks[b]).bit_count()
        sim = inter / denom
    aux_vecs = state["aux_vecs"]  # type: ignore[assignment]
    aux_weight: float = state["aux_weight"]  # type: ignore[assignment]
    if aux_vecs is not None and aux_weight > 0.0:
        va = aux_vecs[a]
        vb = aux_vecs[b]
        if va is not None and vb is not None:
            dot = sum(ax * bx for ax, bx in zip(va, vb))
            sim += aux_weight * ((dot + 1.0) * 0.5)
    cache[(a, b)] = sim
    return sim


def _local_energy_from_snapshot(yx: Tuple[int, int], center_idx: Optional[int], override: Optional[Dict[Tuple[int, int], Optional[int]]] = None) -> float:
    state = _ENERGY_WORKER_STATE
    if override:
        ci = override.get(yx, center_idx)
    else:
        ci = center_idx
    if ci is None:
        return 0.0
    neighbors: Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]] = state["neighbors"]  # type: ignore[assignment]
    cell_owner_grid: Sequence[Sequence[Optional[int]]] = state["cell_owner_grid"]  # type: ignore[assignment]
    energy = 0.0
    for (ny, nx), dist in neighbors[yx]:
        jdx = cell_owner_grid[ny][nx]
        if override and (ny, nx) in override:
            jdx = override[(ny, nx)]
        if jdx is None:
            continue
        energy += _similarity_from_snapshot(ci, jdx) * dist
    return energy


def _energy_for_pair(pair: Tuple[int, Tuple[int, int], int, Tuple[int, int]]) -> Tuple[float, float]:
    ia, yxa, ib, yxb = pair
    override = {yxa: ib, yxb: ia}
    e_cur = _local_energy_from_snapshot(yxa, ia) + _local_energy_from_snapshot(yxb, ib)
    e_swp = _local_energy_from_snapshot(yxa, ib, override=override) + _local_energy_from_snapshot(yxb, ia, override=override)
    return e_cur, e_swp
