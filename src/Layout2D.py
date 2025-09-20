import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


class Layout2D:

    def __init__(self,
                 R_far: int = 7,
                 R_near: int = 3,
                 epochs_far: int = 8,
                 epochs_near: int = 6,
                 seed: int = 123,
                 device: str = "cpu"):
        self.R_far = R_far
        self.R_near = R_near
        self.E_far = epochs_far
        self.E_near = epochs_near
        self.rng = random.Random(seed)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.shape: Tuple[int, int] = (0, 0)
        self.idx2cell: Dict[int, Tuple[int, int]] = {}
        self._codes: List[Set[int]] = []
        self._code_norms: List[float] = []
        self._cell_owner_grid: List[List[Optional[int]]] = []
        self._neighbor_cache: Dict[int, Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]]] = {}
        self._neighbor_indices_gpu: Dict[int, torch.Tensor] = {}
        self._neighbor_dists_gpu: Dict[int, torch.Tensor] = {}
        self._neighbor_mask_gpu: Dict[int, torch.Tensor] = {}
        self._code_tensor: Optional[torch.Tensor] = None
        self._code_norms_t: Optional[torch.Tensor] = None
        self._cell_owner_gpu: Optional[torch.Tensor] = None
        self._cell_owner_flat: Optional[torch.Tensor] = None
        self._cell_id_map: Dict[Tuple[int, int], int] = {}

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        s = math.ceil(math.sqrt(n))
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

    def _local_energy_cpu(self, yx, center_idx, R, sim_cache, override=None) -> float:
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

    def _local_energy_gpu(self, yx, center_idx, R, override=None) -> float:
        if self._code_tensor is None or self._cell_owner_flat is None or self._code_norms_t is None:
            return 0.0
        ci = self._resolve_override(yx, center_idx, override)
        if ci is None:
            return 0.0
        cell_id = self._cell_id_map.get(yx)
        if cell_id is None:
            return 0.0
        neigh_idx = self._neighbor_indices_gpu[R]
        neigh_dist = self._neighbor_dists_gpu[R]
        neigh_mask = self._neighbor_mask_gpu[R]
        mask = neigh_mask[cell_id]
        if not bool(mask.any()):
            return 0.0
        neighbors = neigh_idx[cell_id][mask]
        dists = neigh_dist[cell_id][mask]
        owners = self._cell_owner_flat.index_select(0, neighbors)
        if override:
            owners = owners.clone()
            for oyx, idx in override:
                override_cell = self._cell_id_map.get(oyx)
                if override_cell is None:
                    continue
                eq_mask = neighbors == override_cell
                if bool(eq_mask.any()):
                    owners[eq_mask] = int(idx)
        valid = owners >= 0
        if not bool(valid.any()):
            return 0.0
        owners = owners[valid]
        dists = dists[valid]
        center_vec = self._code_tensor[ci]
        neighbor_vecs = self._code_tensor.index_select(0, owners)
        numerators = torch.mv(neighbor_vecs, center_vec)
        center_norm = self._code_norms_t[ci]
        neighbor_norms = self._code_norms_t.index_select(0, owners)
        denom = center_norm * neighbor_norms
        sims = torch.where(denom > 0.0, numerators / denom, torch.zeros_like(numerators))
        energy = torch.dot(sims, dists)
        return float(energy.item())

    def _local_energy(self, yx, center_idx, R, sim_cache, override=None) -> float:
        if self._code_tensor is not None and self._code_norms_t is not None and self._cell_owner_flat is not None:
            return self._local_energy_gpu(yx, center_idx, R, override=override)
        return self._local_energy_cpu(yx, center_idx, R, sim_cache, override=override)

    def _prepare_neighbors(self, radii: Iterable[int]) -> None:
        H, W = self.shape
        all_cells = [(y, x) for y in range(H) for x in range(W)]
        self._cell_id_map = {cell: idx for idx, cell in enumerate(all_cells)}
        for R in set(radii):
            cache_R: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
            if R <= 0:
                for cell in all_cells:
                    cache_R[cell] = []
                self._neighbor_cache[R] = cache_R
                empty_idx = torch.empty((H * W, 0), dtype=torch.long, device=self.device)
                empty_dist = torch.empty((H * W, 0), dtype=torch.float32, device=self.device)
                empty_mask = torch.empty((H * W, 0), dtype=torch.bool, device=self.device)
                self._neighbor_indices_gpu[R] = empty_idx
                self._neighbor_dists_gpu[R] = empty_dist
                self._neighbor_mask_gpu[R] = empty_mask
                continue
            idx_lists: List[List[int]] = []
            dist_lists: List[List[float]] = []
            max_deg = 0
            for y, x in all_cells:
                neighbors: List[Tuple[Tuple[int, int], float]] = []
                flat_idx: List[int] = []
                flat_dist: List[float] = []
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
                            flat_idx.append(ny * W + nx)
                            flat_dist.append(dist)
                cache_R[(y, x)] = neighbors
                idx_lists.append(flat_idx)
                dist_lists.append(flat_dist)
                if len(flat_idx) > max_deg:
                    max_deg = len(flat_idx)
            if max_deg == 0:
                neigh_idx = torch.empty((H * W, 0), dtype=torch.long, device=self.device)
                neigh_dist = torch.empty((H * W, 0), dtype=torch.float32, device=self.device)
                neigh_mask = torch.empty((H * W, 0), dtype=torch.bool, device=self.device)
            else:
                neigh_idx = torch.full((H * W, max_deg), -1, dtype=torch.long, device=self.device)
                neigh_dist = torch.zeros((H * W, max_deg), dtype=torch.float32, device=self.device)
                neigh_mask = torch.zeros((H * W, max_deg), dtype=torch.bool, device=self.device)
                for cell_id, (inds, dists) in enumerate(zip(idx_lists, dist_lists)):
                    if not inds:
                        continue
                    L = len(inds)
                    neigh_idx[cell_id, :L] = torch.tensor(inds, dtype=torch.long, device=self.device)
                    neigh_dist[cell_id, :L] = torch.tensor(dists, dtype=torch.float32, device=self.device)
                    neigh_mask[cell_id, :L] = True
            self._neighbor_cache[R] = cache_R
            self._neighbor_indices_gpu[R] = neigh_idx
            self._neighbor_dists_gpu[R] = neigh_dist
            self._neighbor_mask_gpu[R] = neigh_mask

    def _build_code_tensor(self, codes: List[Set[int]]) -> None:
        if not codes:
            self._code_tensor = None
            self._code_norms_t = None
            return
        max_bit = max((max(code) for code in codes if code), default=-1)
        if max_bit < 0:
            self._code_tensor = None
            self._code_norms_t = None
            return
        B = max_bit + 1
        code_tensor = torch.zeros((len(codes), B), dtype=torch.float32, device=self.device)
        for i, code in enumerate(codes):
            if not code:
                continue
            idxs = torch.tensor(sorted(code), dtype=torch.long, device=self.device)
            code_tensor[i, idxs] = 1.0
        self._code_tensor = code_tensor
        self._code_norms_t = torch.sqrt(code_tensor.sum(dim=1).clamp_min(0.0))

    def fit(self, codes: List[Set[int]], on_epoch=None, on_swap=None):
        self._codes = codes
        n = len(codes)
        H, W = self._grid_shape(n)
        self.shape = (H, W)

        cells = [(y, x) for y in range(H) for x in range(W)]
        self.idx2cell = {}
        self._cell_owner_grid = [[None for _ in range(W)] for _ in range(H)]
        self._neighbor_cache.clear()
        self._neighbor_indices_gpu.clear()
        self._neighbor_dists_gpu.clear()
        self._neighbor_mask_gpu.clear()
        self._code_norms = [math.sqrt(len(code)) if code else 0.0 for code in codes]
        self._build_code_tensor(codes)
        self._prepare_neighbors([self.R_far, self.R_near])

        self._cell_owner_gpu = torch.full((H, W), -1, dtype=torch.long, device=self.device)
        self._cell_owner_flat = self._cell_owner_gpu.view(-1)
        for i in range(n):
            yx = cells[i]
            self.idx2cell[i] = yx
            self._cell_owner_grid[yx[0]][yx[1]] = i
            self._cell_owner_gpu[yx[0], yx[1]] = int(i)

        def pass_epoch(R: int, iters: int, phase: str):
            for ep in range(iters):
                occupied = list(self.idx2cell.items())
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
                            if self._cell_owner_gpu is not None:
                                self._cell_owner_gpu[yxa[0], yxa[1]] = int(ib)
                                self._cell_owner_gpu[yxb[0], yxb[1]] = int(ia)
                            if on_swap:
                                on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner_grid[yxa[0]][yxa[1]] = ib
                            self._cell_owner_grid[yxb[0]][yxb[1]] = ia
                            if self._cell_owner_gpu is not None:
                                self._cell_owner_gpu[yxa[0], yxa[1]] = int(ib)
                                self._cell_owner_gpu[yxb[0], yxb[1]] = int(ia)
                            if on_swap:
                                on_swap(yxa, yxb, phase, ep, self)

                if on_epoch:
                    on_epoch(phase, ep, self)

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
