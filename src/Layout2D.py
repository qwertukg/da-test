import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import torch
except ImportError:  # pragma: no cover - опциональная зависимость
    torch = None


@dataclass
class _NeighborTensor:
    indices: "torch.Tensor"
    distances: "torch.Tensor"
    mask: "torch.Tensor"


class _TorchEnergyCalculator:
    def __init__(
            self,
            shape: Tuple[int, int],
            codes: List[Set[int]],
            code_norms: List[float],
            neighbor_cache: Dict[int, Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]]],
            cell_owner_grid: List[List[Optional[int]]],
            aux_vectors: Optional[List[Optional[Tuple[float, ...]]]],
            aux_weight: float,
            device: "torch.device"):
        if torch is None:
            raise RuntimeError("Torch недоступен, GPU-ускорение невозможно")
        self.device = device
        self.height, self.width = shape
        self.cell_count = self.height * self.width
        self._aux_weight = float(aux_weight)
        self._build_code_bitsets(codes, code_norms, aux_vectors)
        self._build_neighbors(neighbor_cache)
        self.grid_owner = torch.full((self.cell_count,), -1, dtype=torch.int64, device=self.device)
        self.sync_grid(cell_owner_grid)

    def _cell_index(self, y: int, x: int) -> int:
        return y * self.width + x

    def _build_code_bitsets(
            self,
            codes: List[Set[int]],
            code_norms: List[float],
            aux_vectors: Optional[List[Optional[Tuple[float, ...]]]]):
        max_bit = -1
        for code in codes:
            if code:
                m = max(code)
                if m > max_bit:
                    max_bit = m
        words = max((max_bit + 64) // 64, 1)
        bitsets = torch.zeros((len(codes), words), dtype=torch.int64, device=self.device)
        for idx, code in enumerate(codes):
            for bit in code:
                word = bit // 64
                offset = bit % 64
                bitsets[idx, word] |= (1 << offset)
        self.code_bitsets = bitsets
        self.code_norms = torch.tensor(code_norms, dtype=torch.float32, device=self.device)

        if aux_vectors is None or self._aux_weight <= 0.0:
            self.aux_vectors = None
            self.aux_mask = None
            return
        aux_dim = 0
        for vec in aux_vectors:
            if vec is not None:
                aux_dim = len(vec)
                break
        if aux_dim == 0:
            self.aux_vectors = None
            self.aux_mask = None
            return
        aux_tensor = torch.zeros((len(codes), aux_dim), dtype=torch.float32, device=self.device)
        aux_mask = torch.zeros((len(codes),), dtype=torch.bool, device=self.device)
        for idx, vec in enumerate(aux_vectors):
            if vec is not None:
                aux_tensor[idx] = torch.tensor(vec, dtype=torch.float32, device=self.device)
                aux_mask[idx] = True
        self.aux_vectors = aux_tensor
        self.aux_mask = aux_mask

    def _build_neighbors(self, neighbor_cache):
        self.neighbors: Dict[int, _NeighborTensor] = {}
        for R, table in neighbor_cache.items():
            max_len = 0
            for neighbors in table.values():
                if len(neighbors) > max_len:
                    max_len = len(neighbors)
            max_len = max_len or 1
            indices = torch.full((self.cell_count, max_len), -1, dtype=torch.int64, device=self.device)
            distances = torch.zeros((self.cell_count, max_len), dtype=torch.float32, device=self.device)
            mask = torch.zeros((self.cell_count, max_len), dtype=torch.bool, device=self.device)
            for (y, x), neighbors in table.items():
                cell_id = self._cell_index(y, x)
                for j, ((ny, nx), dist) in enumerate(neighbors):
                    indices[cell_id, j] = self._cell_index(ny, nx)
                    distances[cell_id, j] = float(dist)
                    mask[cell_id, j] = True
            self.neighbors[R] = _NeighborTensor(indices=indices, distances=distances, mask=mask)

    def sync_grid(self, cell_owner_grid: List[List[Optional[int]]]) -> None:
        data = torch.full((self.cell_count,), -1, dtype=torch.int64, device=self.device)
        idx = 0
        for y in range(self.height):
            for x in range(self.width):
                owner = cell_owner_grid[y][x]
                data[idx] = -1 if owner is None else owner
                idx += 1
        self.grid_owner.copy_(data)

    def swap_cells(self, cell_a: Tuple[int, int], cell_b: Tuple[int, int], ia: int, ib: int) -> None:
        idx_a = self._cell_index(cell_a[0], cell_a[1])
        idx_b = self._cell_index(cell_b[0], cell_b[1])
        self.grid_owner[idx_a] = ib
        self.grid_owner[idx_b] = ia

    def local_energy(
            self,
            cell: Tuple[int, int],
            center_idx: Optional[int],
            R: int,
            override=None) -> float:
        if center_idx is None:
            return 0.0
        cell_id = self._cell_index(cell[0], cell[1])
        override_entries: Tuple[Tuple[int, Optional[int]], ...] = tuple(
            (self._cell_index(oy, ox), idx) for (oy, ox), idx in (override or ()))
        ci = center_idx
        for ocell, idx in override_entries:
            if ocell == cell_id:
                ci = idx
                break
        if ci is None:
            return 0.0
        tensor = self.neighbors[R]
        mask = tensor.mask[cell_id]
        if not bool(mask.any()):
            return 0.0
        neighbors = tensor.indices[cell_id][mask]
        distances = tensor.distances[cell_id][mask]
        if neighbors.numel() == 0:
            return 0.0
        owners = self.grid_owner[neighbors]
        if override_entries:
            for ocell, idx in override_entries:
                matches = neighbors == ocell
                if bool(matches.any()):
                    fill_val = -1 if idx is None else idx
                    owners = torch.where(matches, torch.full_like(owners, fill_val), owners)
        valid_mask = owners >= 0
        if not bool(valid_mask.any()):
            return 0.0
        owners = owners[valid_mask].long()
        distances = distances[valid_mask]
        neighbors_bits = self.code_bitsets[owners]
        center_bits = self.code_bitsets[ci].unsqueeze(0)
        intersections = torch.bitwise_and(center_bits, neighbors_bits)
        overlaps = torch.bit_count(intersections).sum(dim=1).to(torch.float32)
        denom = self.code_norms[ci] * self.code_norms[owners]
        sims = torch.zeros_like(denom)
        denom_mask = denom > 0.0
        if bool(denom_mask.any()):
            sims[denom_mask] = overlaps[denom_mask] / denom[denom_mask]
        if self.aux_vectors is not None and self._aux_weight > 0.0 and bool(self.aux_mask[ci]):
            aux_mask = self.aux_mask[owners]
            if bool(aux_mask.any()):
                vec_a = self.aux_vectors[ci]
                vec_b = self.aux_vectors[owners[aux_mask]]
                dots = torch.matmul(vec_b, vec_a)
                contrib = torch.zeros_like(sims)
                contrib[aux_mask] = self._aux_weight * ((dots + 1.0) * 0.5)
                sims = sims + contrib
        return float(torch.dot(sims, distances).item())


class Layout2D:

    def __init__(self, R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123, device: str = "auto"):
        self.R_far = R_far
        self.R_near = R_near
        self.E_far = epochs_far
        self.E_near = epochs_near
        self.rng = random.Random(seed)
        self._device_preference = device
        self.shape: Tuple[int, int] = (0, 0)
        self.idx2cell: Dict[int, Tuple[int, int]] = {}
        self._codes: List[Set[int]] = []
        self._code_norms: List[float] = []
        self._cell_owner_grid: List[List[Optional[int]]] = []
        self._neighbor_cache: Dict[int, Dict[Tuple[int, int], Sequence[Tuple[Tuple[int, int], float]]]] = {}
        self._aux_vecs: Optional[List[Optional[Tuple[float, ...]]]] = None
        self._aux_weight: float = 0.0
        self._torch_energy: Optional[_TorchEnergyCalculator] = None

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
        if self._torch_energy is not None:
            return self._torch_energy.local_energy(yx, center_idx, R, override=override)
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

    def _init_torch_energy(self) -> None:
        self._torch_energy = None
        if self._device_preference == "cpu" or torch is None:
            return
        if self._device_preference == "auto":
            if not torch.cuda.is_available():
                return
            device = torch.device("cuda")
        else:
            device = torch.device(self._device_preference)
            if device.type != "cuda":
                return
            if not torch.cuda.is_available():
                raise RuntimeError("Запрошено устройство CUDA, но оно недоступно")
        self._torch_energy = _TorchEnergyCalculator(
            self.shape,
            self._codes,
            self._code_norms,
            self._neighbor_cache,
            self._cell_owner_grid,
            self._aux_vecs,
            self._aux_weight,
            device)

    def fit(self,
            codes: List[Set[int]],
            *,
            aux_vectors: Optional[Sequence[Sequence[float]]] = None,
            aux_weight: float = 0.0,
            on_epoch=None,
            on_swap=None):
        self._codes = codes
        self._torch_energy = None
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

        self._init_torch_energy()

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
                            if self._torch_energy is not None:
                                self._torch_energy.swap_cells(yxa, yxb, ia, ib)
                            if on_swap: on_swap(yxa, yxb, phase, ep, self)
                    else:
                        if e_swp > e_cur + 1e-9:
                            self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                            self._cell_owner_grid[yxa[0]][yxa[1]] = ib
                            self._cell_owner_grid[yxb[0]][yxb[1]] = ia
                            if self._torch_energy is not None:
                                self._torch_energy.swap_cells(yxa, yxb, ia, ib)
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
