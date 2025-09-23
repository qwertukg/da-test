import math
import multiprocessing as mp
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
        self._owner_array: Optional[Any] = None

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
        self._ensure_owner_array()
        for i in range(n):
            yx = cells[i]
            self.idx2cell[i] = yx
            self._assign_owner(yx[0], yx[1], i)

        self.pass_epoch(self.R_far, self.E_far, phase="far", on_epoch=on_epoch, on_swap=on_swap)
        self.pass_epoch(self.R_near, self.E_near, phase="near", on_epoch=on_epoch, on_swap=on_swap)
        return self

    def _ensure_owner_array(self) -> None:
        H, W = self.shape
        total = H * W
        if total == 0:
            self._owner_array = None
            return
        ctx = mp.get_context()
        owner_array = self._owner_array
        if owner_array is None or len(owner_array) != total:
            self._owner_array = ctx.Array('i', total, lock=False)
            owner_array = self._owner_array
        assert owner_array is not None
        for y in range(H):
            base = y * W
            row = self._cell_owner_grid[y]
            for x in range(W):
                idx = row[x]
                owner_array[base + x] = -1 if idx is None else idx

    def _assign_owner(self, y: int, x: int, idx: Optional[int]) -> None:
        self._cell_owner_grid[y][x] = idx
        owner_array = self._owner_array
        if owner_array is not None:
            owner_array[y * self.shape[1] + x] = -1 if idx is None else idx

    def _make_worker_payload(self) -> Dict[str, Any]:
        neighbor_payload: Dict[int, Dict[Tuple[int, int], Tuple[Tuple[Tuple[int, int], float], ...]]] = {}
        for radius, mapping in self._neighbor_cache.items():
            neighbor_payload[radius] = {cell: tuple(neigh) for cell, neigh in mapping.items()}
        aux_vecs = None
        if self._aux_vecs is not None:
            aux_vecs = tuple(self._aux_vecs)
        return {
            "neighbor_cache": neighbor_payload,
            "code_norms": tuple(self._code_norms),
            "code_bitmasks": tuple(self._code_bitmasks),
            "aux_vecs": aux_vecs,
            "aux_weight": self._aux_weight,
            "grid_shape": self.shape,
            "owner_array": self._owner_array,
        }

    def pass_epoch(self,
                   R: int,
                   iters: int,
                   *,
                   phase: str,
                   on_epoch=None,
                   on_swap=None,
                   processes: Optional[int] = None) -> None:
        if iters <= 0:
            return
        total = self.shape[0] * self.shape[1]
        if total == 0:
            for ep in range(iters):
                if on_epoch:
                    on_epoch(phase, ep, self)
            return
        if self._owner_array is None:
            self._ensure_owner_array()
        payload = self._make_worker_payload()
        if payload["owner_array"] is None:
            return
        ctx = mp.get_context()
        proc_count = processes if processes is not None else mp.cpu_count()
        proc_count = max(1, proc_count)
        manager = ctx.Manager()
        try:
            with ctx.Pool(processes=proc_count, initializer=_worker_init, initargs=(payload,)) as pool:
                pool_size = pool._processes
                for ep in range(iters):
                    sim_cache_proxy = manager.dict()
                    setup_payload = (sim_cache_proxy, R, phase)
                    pool.map(_worker_setup_epoch, [setup_payload] * pool_size)

                    occupied = list(self.idx2cell.items())
                    self.rng.shuffle(occupied)
                    pairs = []
                    for i in range(0, len(occupied) - 1, 2):
                        (ia, yxa), (ib, yxb) = occupied[i], occupied[i + 1]
                        pairs.append((ia, yxa, ib, yxb))

                    if pairs:
                        for ia, ib, yxa, yxb, do_swap in pool.imap(_worker_process_pair, pairs, chunksize=1):
                            if not do_swap:
                                continue
                            if self.idx2cell.get(ia) != yxa or self.idx2cell.get(ib) != yxb:
                                continue
                            if self._cell_owner_grid[yxa[0]][yxa[1]] != ia or \
                                    self._cell_owner_grid[yxb[0]][yxb[1]] != ib:
                                continue
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._assign_owner(yxa[0], yxa[1], ib)
                            self._assign_owner(yxb[0], yxb[1], ia)
                            if on_swap:
                                on_swap(yxa, yxb, phase, ep, self)
                    if on_epoch:
                        on_epoch(phase, ep, self)
        finally:
            manager.shutdown()

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


_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(payload: Dict[str, Any]) -> None:
    neighbor_cache = {}
    for radius, mapping in payload["neighbor_cache"].items():
        neighbor_cache[radius] = {cell: tuple(neigh) for cell, neigh in mapping.items()}
    _WORKER_STATE.update({
        "neighbor_cache": neighbor_cache,
        "code_norms": payload["code_norms"],
        "code_bitmasks": payload["code_bitmasks"],
        "aux_vecs": payload["aux_vecs"],
        "aux_weight": payload["aux_weight"],
        "grid_shape": payload["grid_shape"],
        "owner_array": payload["owner_array"],
        "sim_cache": None,
        "R": 0,
        "phase": "",
    })


def _worker_setup_epoch(args: Tuple[Any, int, str]) -> int:
    sim_cache, radius, phase = args
    _WORKER_STATE["sim_cache"] = sim_cache
    _WORKER_STATE["R"] = radius
    _WORKER_STATE["phase"] = phase
    return 0


def _worker_process_pair(
        pair: Tuple[int, Tuple[int, int], int, Tuple[int, int]]) -> Tuple[int, int, Tuple[int, int], Tuple[int, int], bool]:
    ia, yxa, ib, yxb = pair
    radius = _WORKER_STATE["R"]
    sim_cache = _WORKER_STATE["sim_cache"]
    if sim_cache is None:
        raise RuntimeError("sim_cache не инициализирован")
    override = ((yxa, ib), (yxb, ia))
    e_cur = (_worker_local_energy(yxa, ia, radius, sim_cache, None) +
             _worker_local_energy(yxb, ib, radius, sim_cache, None))
    e_swp = (_worker_local_energy(yxa, ib, radius, sim_cache, override) +
             _worker_local_energy(yxb, ia, radius, sim_cache, override))
    if _WORKER_STATE["phase"] == "far":
        do_swap = e_swp + 1e-9 < e_cur
    else:
        do_swap = e_swp > e_cur + 1e-9
    return ia, ib, yxa, yxb, do_swap


def _worker_local_energy(yx: Tuple[int, int], center_idx: Optional[int], radius: int, sim_cache, override) -> float:
    ci = Layout2D._resolve_override(yx, center_idx, override)
    if ci is None:
        return 0.0
    energy = 0.0
    neighbors = _WORKER_STATE["neighbor_cache"][radius][yx]
    for (ny, nx), dist in neighbors:
        owner = _worker_owner_at(ny, nx)
        owner = Layout2D._resolve_override((ny, nx), owner, override)
        if owner is None:
            continue
        energy += _worker_similarity(ci, owner, sim_cache) * dist
    return energy


def _worker_owner_at(y: int, x: int) -> Optional[int]:
    owner_array = _WORKER_STATE["owner_array"]
    if owner_array is None:
        return None
    _, width = _WORKER_STATE["grid_shape"]
    val = owner_array[y * width + x]
    val_int = int(val)
    return None if val_int < 0 else val_int


def _worker_similarity(ia: int, ib: int, cache) -> float:
    if ia > ib:
        ia, ib = ib, ia
    key = (ia, ib)
    cached = cache.get(key)
    if cached is not None:
        return cached
    code_norms = _WORKER_STATE["code_norms"]
    denom = code_norms[ia] * code_norms[ib]
    if denom == 0.0:
        sim = 0.0
    else:
        bitmasks = _WORKER_STATE["code_bitmasks"]
        sim = (bitmasks[ia] & bitmasks[ib]).bit_count() / denom
    aux_vecs = _WORKER_STATE["aux_vecs"]
    aux_weight = _WORKER_STATE["aux_weight"]
    if aux_vecs is not None and aux_weight > 0.0:
        va = aux_vecs[ia]
        vb = aux_vecs[ib]
        if va is not None and vb is not None:
            dot = sum(ax * bx for ax, bx in zip(va, vb))
            sim += aux_weight * ((dot + 1.0) * 0.5)
    cache[key] = sim
    return sim
