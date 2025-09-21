from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


class Layout2D:

    def __init__(self, R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123,
                 *, use_torch_energy: bool = False, torch_device: Optional[str] = None):
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
        self._aux_vecs: Optional[List[Optional[Tuple[float, ...]]]] = None
        self._aux_weight: float = 0.0
        self._use_torch_energy = bool(use_torch_energy)
        self._torch_device = torch_device
        self._torch_calc: Optional[_TorchEnergyCalculator] = None

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
        if self._torch_calc is not None:
            neighbor_indices: List[int] = []
            distances: List[float] = []
            for (ny, nx), dist in self._neighbors(yx[0], yx[1], R):
                jdx = self._resolve_override((ny, nx), self._cell_owner_grid[ny][nx], override)
                if jdx is None:
                    continue
                neighbor_indices.append(jdx)
                distances.append(dist)
            if not neighbor_indices:
                return 0.0
            return self._torch_calc.local_energy(ci, neighbor_indices, distances)
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
        if self._use_torch_energy:
            self._torch_calc = _TorchEnergyCalculator(
                codes,
                self._code_norms,
                self._aux_vecs,
                self._aux_weight,
                device=self._torch_device,
            )
        else:
            self._torch_calc = None
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


class _TorchEnergyCalculator:

    def __init__(
            self,
            codes: Sequence[Set[int]],
            code_norms: Sequence[float],
            aux_vectors: Optional[Sequence[Optional[Tuple[float, ...]]]],
            aux_weight: float,
            *,
            device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.word_size = 64
        self.code_norms = torch.tensor(code_norms, dtype=torch.float32, device=self.device)
        self.aux_weight = float(aux_weight)
        self._bit_masks = (1 << torch.arange(self.word_size, dtype=torch.int64, device=self.device))
        self._codes = self._encode_codes(codes)
        self._aux_vecs, self._aux_mask = self._encode_aux(aux_vectors)

    def _encode_codes(self, codes: Sequence[Set[int]]) -> torch.Tensor:
        if not codes:
            return torch.zeros((0, 0), dtype=torch.int64, device=self.device)
        max_bit = -1
        for code in codes:
            if code:
                local_max = max(code)
                if local_max > max_bit:
                    max_bit = local_max
        if max_bit < 0:
            return torch.zeros((len(codes), 0), dtype=torch.int64, device=self.device)
        words = (max_bit // self.word_size) + 1
        data = torch.zeros((len(codes), words), dtype=torch.int64, device=self.device)
        for idx, code in enumerate(codes):
            if not code:
                continue
            row = data[idx]
            for bit in code:
                word = bit // self.word_size
                offset = bit % self.word_size
                row[word] |= (1 << offset)
        return data

    def _encode_aux(
            self,
            aux_vectors: Optional[Sequence[Optional[Tuple[float, ...]]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if aux_vectors is None:
            return None, None
        dim = None
        for vec in aux_vectors:
            if vec is not None:
                dim = len(vec)
                break
        if dim is None:
            return None, None
        aux = torch.zeros((len(aux_vectors), dim), dtype=torch.float32, device=self.device)
        mask = torch.zeros(len(aux_vectors), dtype=torch.bool, device=self.device)
        for idx, vec in enumerate(aux_vectors):
            if vec is None:
                continue
            aux[idx] = torch.tensor(vec, dtype=torch.float32, device=self.device)
            mask[idx] = True
        return aux, mask

    def local_energy(
            self,
            center_idx: int,
            neighbor_indices: Sequence[int],
            distances: Sequence[float],
    ) -> float:
        if not neighbor_indices:
            return 0.0
        neighbor_idx = torch.tensor(neighbor_indices, dtype=torch.long, device=self.device)
        dist = torch.tensor(distances, dtype=torch.float32, device=self.device)
        if self._codes.size(1) == 0:
            overlaps = torch.zeros(neighbor_idx.shape[0], dtype=torch.float32, device=self.device)
        else:
            center_words = self._codes[center_idx]
            neighbor_words = torch.index_select(self._codes, 0, neighbor_idx)
            intersections = torch.bitwise_and(neighbor_words, center_words)
            bit_hits = torch.bitwise_and(intersections.unsqueeze(-1), self._bit_masks)
            overlaps = bit_hits.ne(0).sum(dim=-1).sum(dim=-1).to(torch.float32)
        denom = self.code_norms[center_idx] * torch.index_select(self.code_norms, 0, neighbor_idx)
        sims = torch.zeros_like(overlaps)
        valid = denom > 0
        if valid.any():
            sims[valid] = overlaps[valid] / denom[valid]
        if self._aux_vecs is not None and self.aux_weight > 0.0:
            if self._aux_mask is not None and self._aux_mask[center_idx]:
                neighbor_mask = torch.index_select(self._aux_mask, 0, neighbor_idx)
                if neighbor_mask.any():
                    center_vec = self._aux_vecs[center_idx]
                    vecs = torch.index_select(self._aux_vecs, 0, neighbor_idx[neighbor_mask])
                    dots = torch.matmul(vecs, center_vec)
                    sims[neighbor_mask] += self.aux_weight * ((dots + 1.0) * 0.5)
        energy = torch.dot(sims, dist)
        return float(energy.item())
