import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

NeighborEntry = Tuple[Tuple[int, int], float]
NeighborMap = Dict[Tuple[int, int], Sequence[NeighborEntry]]
NeighborCache = Dict[int, NeighborMap]
NeighborOwners = Sequence[int]

WorkerState = Tuple[
    Sequence[int],
    Sequence[float],
    Optional[Sequence[Optional[Tuple[float, ...]]]],
    float,
    NeighborCache,
]

CacheToken = Tuple[str, int, int]

_WORKER_STATE: WorkerState = ((), (), None, 0.0, {})
_WORKER_SIM_CACHE: Dict[Tuple[int, int], float] = {}
_WORKER_CACHE_TOKEN: Optional[CacheToken] = None


def _init_worker_state(
    code_bitmasks: Sequence[int],
    code_norms: Sequence[float],
    aux_vecs: Optional[Sequence[Optional[Tuple[float, ...]]]],
    aux_weight: float,
    neighbor_cache: NeighborCache,
) -> None:
    global _WORKER_STATE, _WORKER_SIM_CACHE, _WORKER_CACHE_TOKEN
    _WORKER_STATE = (code_bitmasks, code_norms, aux_vecs, aux_weight, neighbor_cache)
    _WORKER_SIM_CACHE = {}
    _WORKER_CACHE_TOKEN = None


def _compute_similarity(idx_a: int, idx_b: int, state: WorkerState) -> float:
    code_bitmasks, code_norms, aux_vecs, aux_weight, _ = state
    a, b = (idx_a, idx_b) if idx_a <= idx_b else (idx_b, idx_a)
    denom = code_norms[a] * code_norms[b]
    if denom == 0.0:
        sim = 0.0
    else:
        inter = (code_bitmasks[a] & code_bitmasks[b]).bit_count()
        sim = inter / denom
    if aux_vecs is not None and aux_weight > 0.0:
        va = aux_vecs[a]
        vb = aux_vecs[b]
        if va is not None and vb is not None:
            dot = sum(ax * bx for ax, bx in zip(va, vb))
            sim += aux_weight * ((dot + 1.0) * 0.5)
    return sim


def _evaluate_pair_core(
    ia: int,
    yxa: Tuple[int, int],
    ib: int,
    yxb: Tuple[int, int],
    owners_a: NeighborOwners,
    owners_b: NeighborOwners,
    radius: int,
    state: Optional[WorkerState] = None,
    sim_cache: Optional[Dict[Tuple[int, int], float]] = None,
) -> Tuple[int, Tuple[int, int], int, Tuple[int, int], float, float]:
    if state is None:
        state = _WORKER_STATE

    neighbor_cache = state[4]
    neighbors_a_entries = neighbor_cache[radius][yxa]
    neighbors_b_entries = neighbor_cache[radius][yxb]
    assert len(neighbors_a_entries) == len(owners_a)
    assert len(neighbors_b_entries) == len(owners_b)

    def similarity(idx1: int, idx2: int) -> float:
        key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
        if sim_cache is not None:
            cached = sim_cache.get(key)
            if cached is not None:
                return cached
        value = _compute_similarity(idx1, idx2, state)
        if sim_cache is not None:
            sim_cache[key] = value
        return value

    def local_energy(
        center_idx: int,
        neighbor_entries: Sequence[NeighborEntry],
        neighbor_owners: NeighborOwners,
        overrides: Dict[Tuple[int, int], int],
    ) -> float:
        energy = 0.0
        for (cell_coord, dist), owner in zip(neighbor_entries, neighbor_owners):
            override_owner = overrides.get(cell_coord)
            idx = owner if override_owner is None else override_owner
            if idx < 0:
                continue
            energy += similarity(center_idx, idx) * dist
        return energy

    overrides_left: Dict[Tuple[int, int], int] = {}
    overrides_right: Dict[Tuple[int, int], int] = {}

    current = (
        local_energy(ia, neighbors_a_entries, owners_a, overrides_left)
        + local_energy(ib, neighbors_b_entries, owners_b, overrides_right)
    )

    overrides_left[yxb] = ia
    overrides_right[yxa] = ib

    swapped = (
        local_energy(ib, neighbors_a_entries, owners_a, overrides_left)
        + local_energy(ia, neighbors_b_entries, owners_b, overrides_right)
    )

    return ia, yxa, ib, yxb, current, swapped


def evaluate_pair_worker(
    task: Tuple[
        int,
        Tuple[int, int],
        int,
        Tuple[int, int],
        NeighborOwners,
        NeighborOwners,
        CacheToken,
        int,
    ]
) -> Tuple[int, Tuple[int, int], int, Tuple[int, int], float, float]:
    global _WORKER_CACHE_TOKEN
    ia, yxa, ib, yxb, owners_a, owners_b, cache_token, radius = task
    if cache_token != _WORKER_CACHE_TOKEN:
        _WORKER_SIM_CACHE.clear()
        _WORKER_CACHE_TOKEN = cache_token
    return _evaluate_pair_core(
        ia,
        yxa,
        ib,
        yxb,
        owners_a,
        owners_b,
        radius,
        state=None,
        sim_cache=_WORKER_SIM_CACHE,
    )


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
        self._neighbor_cache: NeighborCache = {}
        self._aux_vecs: Optional[List[Optional[Tuple[float, ...]]]] = None
        self._aux_weight: float = 0.0

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

    def _neighbors(self, y: int, x: int, R: int) -> Sequence[NeighborEntry]:
        return self._neighbor_cache[R][(y, x)]

    @staticmethod
    def _resolve_override(cell: Tuple[int, int], default_idx: Optional[int], override) -> Optional[int]:
        if override:
            for oyx, idx in override:
                if oyx == cell:
                    return idx
        return default_idx

    def _similarity(
        self,
        ia: int,
        ib: int,
        cache: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> float:
        state: WorkerState = (
            self._code_bitmasks,
            self._code_norms,
            self._aux_vecs,
            self._aux_weight,
            self._neighbor_cache,
        )
        key = (ia, ib) if ia <= ib else (ib, ia)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached
        value = _compute_similarity(ia, ib, state)
        if cache is not None:
            cache[key] = value
        return value

    def _local_energy(
        self,
        yx: Tuple[int, int],
        center_idx: Optional[int],
        R: int,
        sim_cache: Optional[Dict[Tuple[int, int], float]],
        override=None,
    ) -> float:
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

    def _pair_energy(self,
                     left_yx: Tuple[int, int],
                     left_idx: Optional[int],
                     right_yx: Tuple[int, int],
                     right_idx: Optional[int],
                     R: int,
                     sim_cache: Optional[Dict[Tuple[int, int], float]],
                     override=None) -> float:
        return (
            self._local_energy(left_yx, left_idx, R, sim_cache, override=override)
            + self._local_energy(right_yx, right_idx, R, sim_cache, override=override)
        )

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

        executor: Optional[ProcessPoolExecutor] = None
        worker_count = 0
        if len(cells) > 1:
            worker_count = min(os.cpu_count() or 1, len(cells))
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker_state,
                initargs=(
                    self._code_bitmasks,
                    self._code_norms,
                    self._aux_vecs,
                    self._aux_weight,
                    self._neighbor_cache,
                ),
            )

        state_for_local: WorkerState = (
            self._code_bitmasks,
            self._code_norms,
            self._aux_vecs,
            self._aux_weight,
            self._neighbor_cache,
        )

        def pass_epoch(R: int, iters: int, phase: str):

            def gather_owners(yx: Tuple[int, int]) -> Tuple[int, ...]:
                owners: List[int] = []
                for (ny, nx), _ in self._neighbors(yx[0], yx[1], R):
                    holder = self._cell_owner_grid[ny][nx]
                    owners.append(holder if holder is not None else -1)
                return tuple(owners)

            for ep in range(iters):
                occupied = list(self.idx2cell.items());
                self.rng.shuffle(occupied)
                pairs = []
                for i in range(0, len(occupied) - 1, 2):
                    (ia, yxa), (ib, yxb) = occupied[i], occupied[i + 1]
                    pairs.append((ia, yxa, ib, yxb))
                use_parallel = executor is not None and len(pairs) > 1
                if not pairs:
                    if on_epoch: on_epoch(phase, ep, self)
                    continue
                if not use_parallel:
                    sim_cache: Dict[Tuple[int, int], float] = {}
                    for ia, yxa, ib, yxb in pairs:
                        owners_a = gather_owners(yxa)
                        owners_b = gather_owners(yxb)
                        _, _, _, _, e_cur, e_swp = _evaluate_pair_core(
                            ia,
                            yxa,
                            ib,
                            yxb,
                            owners_a,
                            owners_b,
                            R,
                            state=state_for_local,
                            sim_cache=sim_cache,
                        )
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
                else:
                    cache_token: CacheToken = (phase, ep, R)
                    tasks = []
                    for ia, yxa, ib, yxb in pairs:
                        owners_a = gather_owners(yxa)
                        owners_b = gather_owners(yxb)
                        tasks.append(
                            (
                                ia,
                                yxa,
                                ib,
                                yxb,
                                owners_a,
                                owners_b,
                                cache_token,
                                R,
                            )
                        )
                    if worker_count:
                        chunk = max(1, len(tasks) // (worker_count * 4))
                    else:
                        chunk = 1
                    for ia, yxa, ib, yxb, e_cur, e_swp in executor.map(
                        evaluate_pair_worker,
                        tasks,
                        chunksize=chunk,
                    ):
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

        try:
            pass_epoch(self.R_far, self.E_far, phase="far")
            pass_epoch(self.R_near, self.E_near, phase="near")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
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
