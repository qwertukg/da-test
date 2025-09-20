import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class Layout2D:

    def __init__(
        self,
        R_far: int = 7,
        R_near: int = 3,
        epochs_far: int = 8,
        epochs_near: int = 6,
        seed: int = 123,
        energy_batch_size: Optional[int] = 64,
    ):
        self.R_far = R_far
        self.R_near = R_near
        self.E_far = epochs_far
        self.E_near = epochs_near
        self.rng = random.Random(seed)
        self.energy_batch_size = energy_batch_size
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

    def _cell_energy_pair(
        self,
        center_cell: Tuple[int, int],
        idx_before: int,
        idx_after: int,
        other_cell: Tuple[int, int],
        other_idx_before: int,
        other_idx_after: int,
        R: int,
        sim_cache: Dict[Tuple[int, int], float],
    ) -> Tuple[float, float]:
        """Считает вклад одной точки пары до и после свапа.

        Реализация следует разделу 5.5 статьи DAML: в рамках одного прохода
        одновременно накапливаются энергии φ_c и φ_s, чтобы не перечитывать
        данные из пространства кода. Локальная аппроксимация учитывает только
        окрестность радиуса *R* (аналог раздела 5.5.2).
        """

        energy_before = 0.0
        energy_after = 0.0
        cy, cx = center_cell
        neighbors = self._neighbors(cy, cx, R)
        for (ny, nx), dist in neighbors:
            neighbor_idx = self._cell_owner_grid[ny][nx]
            if neighbor_idx is None:
                continue
            neighbor_idx_before = (
                other_idx_before if (ny, nx) == other_cell else neighbor_idx
            )
            neighbor_idx_after = (
                other_idx_after if (ny, nx) == other_cell else neighbor_idx
            )
            energy_before += self._similarity(idx_before, neighbor_idx_before, sim_cache) * dist
            energy_after += self._similarity(idx_after, neighbor_idx_after, sim_cache) * dist
        return energy_before, energy_after

    def _pair_energy_batch(
        self,
        batch: Sequence[Tuple[int, Tuple[int, int], int, Tuple[int, int]]],
        R: int,
        sim_cache: Dict[Tuple[int, int], float],
    ) -> Tuple[List[float], List[float]]:
        """Возвращает энергии пар до и после обмена для батча.

        Подход копирует схему «батчевого» расчёта из раздела 5.12 статьи DAML:
        сначала вычисляется вклад всех пар при неизменном состоянии, затем
        суммарные энергии используются при принятии решения о свапе.
        """

        cur_vals: List[float] = []
        swap_vals: List[float] = []
        for ia, yxa, ib, yxb in batch:
            cur_a, swap_a = self._cell_energy_pair(
                yxa,
                ia,
                ib,
                yxb,
                ib,
                ia,
                R,
                sim_cache,
            )
            cur_b, swap_b = self._cell_energy_pair(
                yxb,
                ib,
                ia,
                yxa,
                ia,
                ib,
                R,
                sim_cache,
            )
            cur_vals.append(cur_a + cur_b)
            swap_vals.append(swap_a + swap_b)
        return cur_vals, swap_vals

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
                if not pairs:
                    continue
                sim_cache: Dict[Tuple[int, int], float] = {}
                batch_size = self.energy_batch_size if self.energy_batch_size and self.energy_batch_size > 0 else len(pairs)
                for start in range(0, len(pairs), batch_size):
                    batch = pairs[start:start + batch_size]
                    cur_vals, swap_vals = self._pair_energy_batch(batch, R, sim_cache)
                    for (ia, yxa, ib, yxb), e_cur, e_swp in zip(batch, cur_vals, swap_vals):
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
