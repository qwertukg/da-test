import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # GPU поддержка добавляется опционально, см. §5.10–5.12 DAML
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # torch может отсутствовать в окружении
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


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
        self._similarity_matrix: Optional[np.ndarray] = None
        self._similarity_tensor: Optional["torch.Tensor"] = None
        self._torch_device: Optional["torch.device"] = None
        self._point_energy_cache: Dict[Tuple[int, str], np.ndarray] = {}

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

    def _similarity_slow(self, ia: int, ib: int, cache: Optional[Dict[Tuple[int, int], float]]) -> float:
        a, b = (ia, ib) if ia <= ib else (ib, ia)
        if cache is not None:
            cached = cache.get((a, b))
            if cached is not None:
                return cached
        denom = self._code_norms[a] * self._code_norms[b]
        if denom == 0.0:
            sim = 0.0
        else:
            sim = len(self._codes[a] & self._codes[b]) / denom
        if cache is not None:
            cache[(a, b)] = sim
        return sim

    def _similarity_lookup(self, ia: int, ib: int, cache: Optional[Dict[Tuple[int, int], float]] = None) -> float:
        if self._similarity_matrix is not None:
            return float(self._similarity_matrix[ia, ib])
        return self._similarity_slow(ia, ib, cache)

    def _local_energy(self, yx, center_idx, R, sim_cache=None, override=None) -> float:
        ci = self._resolve_override(yx, center_idx, override)
        if ci is None:
            return 0.0
        energy = 0.0
        for (ny, nx), dist2 in self._neighbors(yx[0], yx[1], R):
            jdx = self._resolve_override((ny, nx), self._cell_owner_grid[ny][nx], override)
            if jdx is None:
                continue
            energy += self._similarity_lookup(ci, jdx, sim_cache) * dist2
        return energy

    def _prepare_similarity_tensor(self) -> None:
        if not _TORCH_AVAILABLE or self._similarity_matrix is None:
            self._similarity_tensor = None
            self._torch_device = None
            return
        if self._similarity_tensor is not None and self._similarity_tensor.numel() == self._similarity_matrix.size:
            return
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # GPU ускорение доступно только при наличии CUDA (см. §5.12 DAML)
        if device.type == "cuda":
            self._similarity_tensor = torch.tensor(self._similarity_matrix, device=device)
            self._torch_device = device
        else:
            self._similarity_tensor = None
            self._torch_device = None

    def _maybe_prepare_similarity_matrix(self) -> None:
        n = len(self._codes)
        if n == 0:
            self._similarity_matrix = None
            self._similarity_tensor = None
            return
        avg_len = sum(len(code) for code in self._codes) / max(1, n)
        approx_bytes = n * n * 4
        # Эвристика из §5.9.5 DAML: предвычисление имеет смысл для длинных кодов при приемлемой памяти
        memory_limit = 512 * 1024 * 1024  # 512 МБ
        if avg_len < 32 and approx_bytes > 128 * 1024 * 1024:
            self._similarity_matrix = None
            self._similarity_tensor = None
            return
        if approx_bytes > memory_limit:
            self._similarity_matrix = None
            self._similarity_tensor = None
            return
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            code_i = self._codes[i]
            norm_i = self._code_norms[i]
            sim[i, i] = 1.0 if norm_i else 0.0
            for j in range(i + 1, n):
                norm_j = self._code_norms[j]
                if norm_i == 0.0 or norm_j == 0.0:
                    value = 0.0
                else:
                    value = len(code_i & self._codes[j]) / (norm_i * norm_j)
                sim[i, j] = sim[j, i] = value
        self._similarity_matrix = sim
        self._prepare_similarity_tensor()

    def _batched_similarity_dot(self, centers: Sequence[int], neighbor_indices: Sequence[Sequence[int]], weights: Sequence[Sequence[float]]) -> np.ndarray:
        if not centers:
            return np.zeros(0, dtype=np.float32)
        if self._similarity_matrix is None:
            # медленный путь без матрицы, но с кешированием (§5.9.2)
            energies = []
            cache: Dict[Tuple[int, int], float] = {}
            for ci, idxs, ws in zip(centers, neighbor_indices, weights):
                total = 0.0
                for nb, w in zip(idxs, ws):
                    total += self._similarity_slow(ci, nb, cache) * w
                energies.append(total)
            return np.asarray(energies, dtype=np.float32)
        if self._torch_device is not None and self._similarity_tensor is not None:
            max_len = max((len(ids) for ids in neighbor_indices), default=0)
            if max_len == 0:
                return np.zeros(len(centers), dtype=np.float32)
            idx_tensor = torch.full((len(centers), max_len), -1, dtype=torch.long, device=self._torch_device)
            weight_tensor = torch.zeros((len(centers), max_len), dtype=torch.float32, device=self._torch_device)
            for row, (idxs, ws) in enumerate(zip(neighbor_indices, weights)):
                if not idxs:
                    continue
                idx_tensor[row, : len(idxs)] = torch.as_tensor(idxs, dtype=torch.long, device=self._torch_device)
                weight_tensor[row, : len(ws)] = torch.as_tensor(ws, dtype=torch.float32, device=self._torch_device)
            center_tensor = torch.as_tensor(centers, dtype=torch.long, device=self._torch_device)
            sim_rows = torch.take_along_dim(self._similarity_tensor[center_tensor], idx_tensor.clamp(min=0), dim=1)
            sim_rows = torch.where(idx_tensor >= 0, sim_rows, torch.zeros_like(sim_rows))
            energies = (sim_rows * weight_tensor).sum(dim=1)
            return energies.detach().cpu().numpy().astype(np.float32)
        # CPU-векторизация на numpy
        sim_matrix = self._similarity_matrix
        assert sim_matrix is not None
        result = np.zeros(len(centers), dtype=np.float32)
        for row, (ci, idxs, ws) in enumerate(zip(centers, neighbor_indices, weights)):
            if not idxs:
                continue
            sim_vals = sim_matrix[ci, np.asarray(list(idxs), dtype=np.int32)]
            result[row] = float(np.dot(sim_vals, np.asarray(list(ws), dtype=np.float32)))
        return result

    def _compute_point_energies(self, R: int) -> np.ndarray:
        cache_key = (R, "point")
        if cache_key in self._point_energy_cache:
            return self._point_energy_cache[cache_key]
        centers: List[int] = []
        neighbor_indices: List[List[int]] = []
        weights: List[List[float]] = []
        for idx, yx in self.idx2cell.items():
            centers.append(idx)
            local_idxs: List[int] = []
            local_weights: List[float] = []
            for (ny, nx), dist2 in self._neighbors(yx[0], yx[1], R):
                nb = self._cell_owner_grid[ny][nx]
                if nb is None:
                    continue
                local_idxs.append(nb)
                local_weights.append(dist2)
            neighbor_indices.append(local_idxs)
            weights.append(local_weights)
        energies = self._batched_similarity_dot(centers, neighbor_indices, weights)
        self._point_energy_cache[cache_key] = energies
        return energies

    def _select_active_indices(self, weights: Sequence[float]) -> List[int]:
        if not weights:
            return list(self.idx2cell.keys())
        arr = np.asarray(weights, dtype=np.float32)
        positive = arr[arr > 0.0]
        if positive.size == 0:
            return list(self.idx2cell.keys())
        cutoff = float(np.quantile(positive, 0.5))
        active = [idx for idx, w in enumerate(arr) if w >= cutoff]
        return active if active else list(self.idx2cell.keys())

    def _weighted_order(self, indices: Sequence[int], weights: Sequence[float]) -> List[int]:
        if not indices:
            return []
        w = np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)
        selected_weights = w[list(indices)] if len(w) else np.ones(len(indices), dtype=np.float64)
        if selected_weights.sum() <= 0:
            order = list(indices)
            self.rng.shuffle(order)
            return order
        # SeedSequence требует целочисленную энтропию, используем 32-битное значение из базового ГПСЧ
        rng = np.random.default_rng(self.rng.randrange(2 ** 32))
        probs = selected_weights / selected_weights.sum()
        order = list(rng.choice(indices, size=len(indices), replace=False, p=probs))
        return order

    @staticmethod
    def _similarity_cutoff(phase: str, progress: float) -> float:
        if phase == "far":
            return 0.02 + 0.08 * progress
        return 0.1 + 0.15 * progress

    def _generate_pairs(self, R: int, phase: str, weights: Sequence[float], progress: float) -> List[Tuple[int, Tuple[int, int], int, Tuple[int, int]]]:
        active_candidates = self._select_active_indices(weights)
        available = set(active_candidates)
        order = self._weighted_order(active_candidates, weights)
        pairs: List[Tuple[int, Tuple[int, int], int, Tuple[int, int]]] = []
        sim_threshold = self._similarity_cutoff(phase, progress)
        for ia in order:
            if ia not in available:
                continue
            yxa = self.idx2cell[ia]
            neighbors_info: List[Tuple[int, float, float, float]] = []
            for (ny, nx), dist2 in self._neighbors(yxa[0], yxa[1], R):
                ib = self._cell_owner_grid[ny][nx]
                if ib is None or ib == ia or ib not in available:
                    continue
                sim = self._similarity_lookup(ia, ib)
                if sim < sim_threshold:
                    continue
                weight_ib = weights[ib] if ib < len(weights) else 0.0
                neighbors_info.append((ib, sim, dist2, weight_ib))
            if not neighbors_info:
                available.remove(ia)
                continue
            scores = [sim * (1.0 + nb_w) / (1.0 + dist2) for (_, sim, dist2, nb_w) in neighbors_info]
            total = sum(scores)
            if total <= 0:
                chosen = neighbors_info[self.rng.randrange(len(neighbors_info))]
            else:
                r = self.rng.random() * total
                chosen = neighbors_info[-1]
                acc = 0.0
                for info, score in zip(neighbors_info, scores):
                    acc += score
                    if r <= acc:
                        chosen = info
                        break
            ib = chosen[0]
            yxb = self.idx2cell[ib]
            pairs.append((ia, yxa, ib, yxb))
            available.discard(ia)
            available.discard(ib)
        return pairs

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
                        dist2 = dy * dy + dx * dx
                        if dist2 > R * R:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            neighbors.append(((ny, nx), float(dist2)))
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
        self._point_energy_cache.clear()
        self._prepare_neighbors([self.R_far, self.R_near])
        for i in range(n):
            yx = cells[i];
            self.idx2cell[i] = yx;
            self._cell_owner_grid[yx[0]][yx[1]] = i
        self._maybe_prepare_similarity_matrix()

        def pass_epoch(R: int, iters: int, phase: str):
            for ep in range(iters):
                self._point_energy_cache.pop((R, "point"), None)
                energies = self._compute_point_energies(R)
                if energies.size:
                    max_energy = float(energies.max())
                    weights = (energies / max(max_energy, 1e-9)).astype(np.float32).tolist()
                else:
                    weights = [1.0] * len(self.idx2cell)
                progress = ep / max(1, iters)
                pairs = self._generate_pairs(R, phase, weights, progress)
                sim_cache: Optional[Dict[Tuple[int, int], float]] = {} if self._similarity_matrix is None else None
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
                            self._point_energy_cache.pop((R, "point"), None)
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner_grid[yxa[0]][yxa[1]] = ib
                            self._cell_owner_grid[yxb[0]][yxb[1]] = ia
                            self._point_energy_cache.pop((R, "point"), None)
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
