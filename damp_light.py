#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discrete_digits_final.py — дискретное кодирование 0–9 с улучшениями:
1) Первичный код: BRIGHT + COARSE + ORIENT (Sobel) + GRID, "цветовое" объединение (top-K по приоритетам)
2) Раскладка (DAMP-light): локальные свапы с фикс-override владельцев клеток
3) Детекторы: адаптивный порог (перцентиль), эллиптические поля (PCA), дедуп центров
4) Эмбеддинг: срабатывание детекторов -> разрежённый код
5) Классификация: взвешенный kNN по Жаккару

Датасет: sklearn.load_digits (8×8). Работает оффлайн.
"""

import math, random
from collections import deque, Counter
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.manifold import MDS
import json, datetime as dt
from pathlib import Path
from scipy.signal import convolve2d
from collections import Counter

def print_top_common_bits(Z, y, top=10):
    by_cls = {}
    for z, c in zip(Z, y):
        by_cls.setdefault(int(c), []).append(z)
    for c, zs in sorted(by_cls.items()):
        n = len(zs)
        df = Counter(b for z in zs for b in z)  # частота детектора по классу
        top_bits = sorted(df.items(), key=lambda t: t[1], reverse=True)[:top]
        print(f"\nКласс {c}: n={n}")
        for b, f in top_bits:
            print(f"  бит {b}: доля {f/n:.2f}")


# ============================================================
# Утилиты для битовых кодов (коды = set[int] активных позиций)
# ============================================================

def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def cosbin(a: Set[int], b: Set[int]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))

# ============================================================
# Свёртка и ориентиры Собеля
# ============================================================

def _conv2_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    H, W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(H):
        for x in range(W):
            patch = pad[y:y + k.shape[0], x:x + k.shape[1]]
            out[y, x] = float(np.sum(patch * k))
    return out

def sobel_mag_dir(img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float32)
    gx = _conv2_same(img01, kx)
    gy = _conv2_same(img01, ky)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi)  # [0, 2π)
    return mag, ang

# ============================================================
# 1) Первичное кодирование (каналы + "цветовое" объединение)
# ============================================================

class PrimaryEncoder:
    """
    Каналы:
      - BRIGHT: глобальная яркость (квантование на bright_levels)
      - COARSE: 2×2 грубая сетка (наличие штриха)
      - ORIENT: ориентиры Собрела, доминантный угол на coarse-сетке orient_grid × orient_grid
      - GRID: G×G, квантование яркости на grid_levels
    "Цветовое" объединение: упорядочиваем (priority, bit) и берём top-K (max_active_bits).
    """
    def __init__(self,
                 img_hw=(8, 8),
                 bits=8192,
                 seed=42,
                 # GRID
                 grid_g=4, grid_levels=4, grid_bits_per_cell=3,
                 # COARSE
                 coarse_bits_per_cell=4,
                 # BRIGHT
                 bright_levels=8,
                 # ORIENT (Sobel)
                 orient_on=True, orient_bins=8, orient_grid=4,
                 orient_bits_per_cell=2, orient_mag_thresh=0.12,
                 # итоговый лимит
                 max_active_bits=260):
        self.H, self.W = img_hw
        self.B = bits
        self.rng = random.Random(seed)

        self.G = grid_g
        self.grid_levels = grid_levels
        self.grid_bits_per_cell = grid_bits_per_cell

        self.coarse_bits_per_cell = coarse_bits_per_cell
        self.bright_levels = bright_levels

        self.orient_on = orient_on
        self.orient_bins = orient_bins
        self.orient_grid = orient_grid
        self.orient_bits_per_cell = orient_bits_per_cell
        self.orient_mag_thresh = orient_mag_thresh

        self.max_active = max_active_bits

        # Кодбуки (стабильны от seed)
        self.grid_codebook: Dict[Tuple[int, int, int], List[int]] = {}
        for y in range(self.G):
            for x in range(self.G):
                for lvl in range(self.grid_levels):
                    self.grid_codebook[(y, x, lvl)] = self._rand_bits(self.grid_bits_per_cell)

        self.coarse_codebook: Dict[Tuple[int, int], List[int]] = {}
        for y in range(2):
            for x in range(2):
                self.coarse_codebook[(y, x)] = self._rand_bits(self.coarse_bits_per_cell)

        self.bright_codebook: Dict[int, List[int]] = {}
        for lvl in range(self.bright_levels):
            self.bright_codebook[lvl] = self._rand_bits(4)  # 4 бита на BRIGHT

        self.orient_codebook: Dict[Tuple[int, int, int], List[int]] = {}
        if self.orient_on:
            for y in range(self.orient_grid):
                for x in range(self.orient_grid):
                    for b in range(self.orient_bins):
                        self.orient_codebook[(y, x, b)] = self._rand_bits(self.orient_bits_per_cell)

    def _rand_bits(self, k: int) -> List[int]:
        return self.rng.sample(range(self.B), k)

    @staticmethod
    def _quantize01(v: float, levels: int) -> int:
        x = max(0.0, min(1.0, v))
        return min(levels - 1, int(x * levels))

    def encode(self, img: np.ndarray) -> Set[int]:
        """
        img: 2D numpy (H×W) с яркостями [0..1]
        -> set[int] активных битов
        """
        H, W, G = self.H, self.W, self.G
        ph: List[Tuple[float, int]] = []  # (priority, bit)

        # --- BRIGHT (priority 0.0) ---
        mean_b = float(np.mean(img))
        lvl_b = self._quantize01(mean_b, self.bright_levels)
        for b in self.bright_codebook[lvl_b]:
            ph.append((0.0, b))

        # --- COARSE 2×2 (priority 1.0) ---
        h2, w2 = H // 2, W // 2
        blocks = [
            img[0:h2, 0:w2], img[0:h2, w2:W],
            img[h2:H, 0:w2], img[h2:H, w2:W]
        ]
        for i, blk in enumerate(blocks):
            y, x = divmod(i, 2)
            if float(np.mean(blk)) > 0.15:
                for b in self.coarse_codebook[(y, x)]:
                    ph.append((1.0, b))

        # --- ORIENT (priority 1.5) ---
        if self.orient_on:
            mag, ang = sobel_mag_dir(img)
            gy, gx = self.H // self.orient_grid, self.W // self.orient_grid
            mmax = float(np.max(mag))
            mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag
            bins = np.linspace(0.0, 2 * np.pi, self.orient_bins + 1)
            for y in range(self.orient_grid):
                for x in range(self.orient_grid):
                    tile_m = mnorm[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
                    tile_a = ang[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
                    m = float(np.mean(tile_m))
                    if m < self.orient_mag_thresh:
                        continue
                    hist = np.zeros(self.orient_bins, dtype=np.float32)
                    h, w = tile_m.shape
                    for iy in range(h):
                        for ix in range(w):
                            b = int(np.clip(np.searchsorted(bins, tile_a[iy, ix], side='right') - 1,
                                            0, self.orient_bins - 1))
                            hist[b] += tile_m[iy, ix]
                    b_max = int(np.argmax(hist))
                    for bit in self.orient_codebook[(y, x, b_max)]:
                        ph.append((1.5, bit))

        # --- GRID G×G (priority 2.0) ---
        gy, gx = H // G, W // G
        for y in range(G):
            for x in range(G):
                tile = img[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
                m = float(np.mean(tile))
                lvl = self._quantize01(m, self.grid_levels)
                lvls = {lvl}
                if lvl > 0: lvls.add(lvl - 1)  # лёгкое перекрытие
                for lv in lvls:
                    for b in self.grid_codebook[(y, x, lv)]:
                        ph.append((2.0, b))

        # --- "цветовое" объединение (top-K по приоритетам) ---
        ph.sort(key=lambda z: z[0])
        out: Set[int] = set()
        for _, bit in ph:
            if len(out) >= self.max_active: break
            out.add(bit)
        return out

# ============================================================
# 2) Раскладка (упрощённый DAMP с локальными свапами и override)
# ============================================================

class Layout2D:
    """
    Укладывает N кодов на дискретную сетку h×w (h*w>=N).
    Оптимизация: многократно перебираем случайные непересекающиеся пары занятых ячеек
    и меняем местами их элементы, если это снижает локальную "энергию":
       sum_{соседи} sim(code_i, code_nbr) * dist(cell_i, cell_nbr)
    Два режима: FAR (радиус R_far) и NEAR (радиус R_near).
    """
    def __init__(self, R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123):
        self.R_far = R_far
        self.R_near = R_near
        self.E_far = epochs_far
        self.E_near = epochs_near
        self.rng = random.Random(seed)
        self.shape: Tuple[int, int] = (0, 0)
        self.cell2idx: List[Optional[int]] = []
        self.idx2cell: Dict[int, Tuple[int, int]] = {}

    @staticmethod
    def _grid_shape(n: int) -> Tuple[int, int]:
        s = math.ceil(math.sqrt(n))
        return (s, s)

    def _neighbors(self, y: int, x: int, R: int) -> List[Tuple[int, int]]:
        H, W = self.shape
        out = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    out.append((ny, nx))
        return out

    @staticmethod
    def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _local_energy(self,
                      yx: Tuple[int, int],
                      center_idx: int,
                      R: int,
                      sim_cache: Dict[Tuple[int, int], float],
                      override: Optional[Dict[Tuple[int, int], int]] = None) -> float:
        """
        Энергия клетки yx с "центральным" кодом center_idx как сумма sim*dist
        по соседям в радиусе R. Коды берём по индексам владельцев self._cell_owner,
        для "виртуального свапа" можно передать override: {(y,x)->idx}.
        """
        y, x = yx
        e = 0.0
        ci = override.get((y, x), center_idx) if override else center_idx
        code_c = self._codes[ci]
        for ny, nx in self._neighbors(y, x, R):
            jdx = (override.get((ny, nx), self._cell_owner.get((ny, nx)))
                   if override else self._cell_owner.get((ny, nx)))
            if jdx is None:
                continue
            a, b = (ci, jdx) if ci <= jdx else (jdx, ci)
            s = sim_cache.get((a, b))
            if s is None:
                s = cosbin(code_c, self._codes[jdx])
                sim_cache[(a, b)] = s
            e += s * self._dist((y, x), (ny, nx))
        return e

    def fit(self, codes: List[Set[int]]):
        n = len(codes)
        H, W = self._grid_shape(n)
        self.shape = (H, W)
        self._codes = codes  # держим ссылку на коды

        # начальная укладка — построчно
        cells = [(y, x) for y in range(H) for x in range(W)]
        self.cell2idx = [None] * (H * W)
        self.idx2cell = {}
        for i in range(n):
            y, x = cells[i]
            self.cell2idx[i] = i
            self.idx2cell[i] = (y, x)

        # владельцы ячеек
        self._cell_owner: Dict[Tuple[int, int], Optional[int]] = {}
        for yx in cells:
            self._cell_owner[yx] = None
        for i in range(n):
            y, x = self.idx2cell[i]
            self._cell_owner[(y, x)] = i

        def pass_epoch(R: int, iters: int):
            for _ in range(iters):
                occupied = list(self.idx2cell.items())  # (idx -> (y,x)), длина = n
                self.rng.shuffle(occupied)
                pairs = []
                for i in range(0, len(occupied) - 1, 2):
                    (ia, yxa), (ib, yxb) = occupied[i], occupied[i + 1]
                    pairs.append((ia, yxa, ib, yxb))
                sim_cache: Dict[Tuple[int, int], float] = {}
                for ia, yxa, ib, yxb in pairs:
                    e_cur = (
                        self._local_energy(yxa, ia, R, sim_cache) +
                        self._local_energy(yxb, ib, R, sim_cache)
                    )
                    override = {yxa: ib, yxb: ia}
                    e_swp = (
                        self._local_energy(yxa, ib, R, sim_cache, override=override) +
                        self._local_energy(yxb, ia, R, sim_cache, override=override)
                    )
                    if e_swp + 1e-9 < e_cur:
                        # применяем свап
                        self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
                        self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia

        # FAR порядок (несколько проходов большим радиусом)
        pass_epoch(self.R_far, self.E_far)
        # NEAR порядок (полировка)
        pass_epoch(self.R_near, self.E_near)
        return self

    def grid_shape(self) -> Tuple[int, int]:
        return self.shape

    def position_of(self, idx: int) -> Tuple[int, int]:
        return self.idx2cell[idx]

# ============================================================
# 3) Служебные функции для детекторов
# ============================================================

def connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or seen[y, x]: continue
            q = deque([(y, x)]); seen[y, x] = True
            comp = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0: continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                            comp.append((ny, nx))
            comps.append(comp)
    return comps

def comp_center_radius(comp: List[Tuple[int, int]]) -> Tuple[Tuple[float, float], float]:
    ys = [p[0] for p in comp]; xs = [p[1] for p in comp]
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    r = 0.0
    for (y, x) in comp:
        r = max(r, math.hypot(y - cy, x - cx))
    return (cy, cx), (r * 1.15 + 0.5)

def pca_ellipse(points: List[Tuple[int, int]], pad_scale: float = 1.6):
    ys = np.array([p[0] for p in points], dtype=np.float32)
    xs = np.array([p[1] for p in points], dtype=np.float32)
    cy, cx = float(ys.mean()), float(xs.mean())
    Y = np.stack([ys - cy, xs - cx], axis=0)  # 2×N
    if Y.shape[1] < 3:
        # слишком мало точек — вернём маленький круг
        return (cy, cx), (np.array([1.0, 0.0]), np.array([0.0, 1.0])), (1.0, 1.0)
    C = (Y @ Y.T) / max(1, Y.shape[1] - 1)    # ковариация 2×2
    vals, vecs = np.linalg.eigh(C)            # по возрастанию
    vals = np.clip(vals, 1e-6, None)
    r1, r2 = pad_scale * float(np.sqrt(vals[1])), pad_scale * float(np.sqrt(vals[0]))
    u1 = vecs[:, 1] / np.linalg.norm(vecs[:, 1])
    u2 = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
    return (cy, cx), (u1, u2), (r1, r2)

# ============================================================
# 4) Детекторы: адаптивный порог + эллипсы
# ============================================================

class DetectorSpace:
    """
    Строим детекторы поверх раскладки:
      - для "якорей" считаем похожесть со всеми клетками;
      - порог = max(lam_floor, квантиль(percentile)); страховка top-N;
      - связные компоненты -> круги/эллипсы; дедуп центров; бит на детектор.
    Эмбеддинг: детектор "срабатывает", если доля активных клеток в его фигуре ≥ μ.
    """
    def __init__(self,
                 layout: Layout2D,
                 codes_train: List[Set[int]],
                 y_train: Optional[List[int]] = None,
                 emb_bits=2048,
                 lam_floor=0.06,
                 percentile=0.88,
                 min_activated=35,
                 mu=0.20,
                 seeds=1200,
                 min_comp=5,
                 min_center_dist=1.6,
                 max_detectors=512,
                 seed=7):
        self.layout = layout
        self.codes_train = codes_train
        self.y_train = y_train
        self.H, self.W = layout.grid_shape()
        self.emb_bits = emb_bits
        self.lam_floor = lam_floor
        self.percentile = percentile
        self.min_activated = min_activated
        self.mu = mu
        self.seeds = min(seeds, len(codes_train))
        self.min_comp = min_comp
        self.min_center_dist = min_center_dist
        self.max_detectors = max_detectors
        self.rng = random.Random(seed)

        self.cell_owner: Dict[Tuple[int, int], int] = {}
        for idx, yx in layout.idx2cell.items():
            self.cell_owner[yx] = idx

        self.detectors: List[Dict] = []
        self._build()

    def _activation_map_adaptive(self, code: Set[int]) -> np.ndarray:
        sims = np.zeros((self.H, self.W), dtype=np.float32)
        has = np.zeros((self.H, self.W), dtype=bool)
        vals, coords = [], []
        for (y, x), idx in self.cell_owner.items():
            s = cosbin(code, self.codes_train[idx])
            sims[y, x] = s; has[y, x] = True
            vals.append(s); coords.append((y, x))
        if not vals:
            return np.zeros((self.H, self.W), dtype=bool)

        v = np.array(vals, dtype=np.float32)
        thr = max(self.lam_floor, float(np.quantile(v, self.percentile)))
        act = (sims >= thr) & has
        if act.sum() < self.min_activated:
            order = np.argsort(-v)[:self.min_activated]
            act[:] = False
            for k in order:
                y, x = coords[k]
                act[y, x] = True
        return act

    def _too_close(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> bool:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < self.min_center_dist

    def _sample_seed_indices(self) -> List[int]:
        if not self.y_train:
            return self.rng.sample(range(len(self.codes_train)), self.seeds)
        by_cls: Dict[int, List[int]] = {}
        for i, y in enumerate(self.y_train):
            by_cls.setdefault(int(y), []).append(i)
        per = max(1, self.seeds // max(1, len(by_cls)))
        out = []
        for _, idxs in by_cls.items():
            self.rng.shuffle(idxs)
            out.extend(idxs[:per])
        self.rng.shuffle(out)
        return out[:self.seeds]

    def _build(self):
        used_bits: Set[int] = set()
        def alloc_bit() -> int:
            for b in range(self.emb_bits):
                if b not in used_bits:
                    used_bits.add(b); return b
            return self.rng.randrange(self.emb_bits)

        candidates: List[Dict] = []
        seed_idxs = self._sample_seed_indices()
        for idx in seed_idxs:
            code = self.codes_train[idx]
            act = self._activation_map_adaptive(code)
            comps = connected_components(act)
            for comp in comps:
                if len(comp) < self.min_comp: continue
                center_c, radius_c = comp_center_radius(comp)
                (cent_e, (u1, u2), (r1, r2)) = pca_ellipse(comp, pad_scale=1.6)
                elong = (max(r1, r2) / max(1e-6, min(r1, r2)))
                if elong >= 1.4 and max(r1, r2) > 0.8:
                    candidates.append({
                        "shape": "ellipse",
                        "center": cent_e,
                        "u1": u1, "u2": u2,
                        "r1": r1, "r2": r2,
                        "score": len(comp),
                    })
                else:
                    candidates.append({
                        "shape": "circle",
                        "center": center_c,
                        "radius": radius_c,
                        "score": len(comp),
                    })

        candidates.sort(key=lambda z: -z["score"])
        kept: List[Dict] = []
        for c in candidates:
            if all(not self._too_close(c["center"], k["center"]) for k in kept):
                kept.append(c)
            if len(kept) >= self.max_detectors:
                break

        for c in kept:
            c["bit"] = alloc_bit()

        self.detectors = kept

    def embed(self, code: Set[int]) -> Set[int]:
        act = self._activation_map_adaptive(code)
        ones = set()
        for d in self.detectors:
            cy, cx = d["center"]
            if d.get("shape") == "ellipse":
                u1, u2 = d["u1"], d["u2"]; r1, r2 = d["r1"], d["r2"]
                y0, y1 = max(0, int(cy - r1 - r2)), min(self.H - 1, int(cy + r1 + r2) + 1)
                x0, x1 = max(0, int(cx - r1 - r2)), min(self.W - 1, int(cx + r1 + r2) + 1)
                tot = hit = 0
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        vy, vx = (y - cy), (x - cx)
                        a = (vy * u1[0] + vx * u1[1]) / (r1 + 1e-9)
                        b = (vy * u2[0] + vx * u2[1]) / (r2 + 1e-9)
                        if (a * a + b * b) <= 1.0:
                            tot += 1
                            if act[y, x]: hit += 1
                if tot and (hit / tot) >= self.mu:
                    ones.add(d["bit"])
            else:
                r = d["radius"]
                y0, y1 = max(0, int(cy - r)), min(self.H - 1, int(cy + r) + 1)
                x0, x1 = max(0, int(cx - r)), min(self.W - 1, int(cx + r) + 1)
                tot = hit = 0
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2 + 1e-9:
                            tot += 1
                            if act[y, x]: hit += 1
                if tot and (hit / tot) >= self.mu:
                    ones.add(d["bit"])
        return ones

# ============================================================
# 5) Взвешенный kNN по Жаккару
# ============================================================

class KNNJaccard:
    def __init__(self, k=5, eps=1e-6):
        self.k = k; self.eps = eps
        self.X: List[Set[int]] = []; self.y: List[int] = []

    def fit(self, X: List[Set[int]], y: List[int]):
        self.X = X; self.y = y
        return self

    def predict(self, X: List[Set[int]]) -> List[int]:
        out = []
        for q in X:
            sims = [(jaccard(q, x), yi) for x, yi in zip(self.X, self.y)]
            sims.sort(key=lambda z: -z[0])
            top = sims[:self.k]
            score: Dict[int, float] = {}
            for s, yi in top:
                score[yi] = score.get(yi, 0.0) + (s + self.eps)
            out.append(max(score.items(), key=lambda z: z[1])[0])
        return out

# ============================================================
# 6) Всё вместе
# ============================================================

def to01(img8x8: np.ndarray) -> np.ndarray:
    # load_digits даёт 0..16 -> в 0..1
    return (img8x8.astype(np.float32) / 16.0).reshape(8, 8)


def load_mnist_28x28(train_limit=8000, test_limit=2000, seed=0):
    """
    Грузим MNIST (28×28, 0..1), слегка подсэмплируем для разумного времени работы.
    Вернёт списки 2D numpy-массивов (H×W) и метки.
    """
    tfm = transforms.ToTensor()  # -> tensor [1,28,28] в [0,1]
    train_ds = MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = MNIST(root="./data", train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(train_ds))
    test_idx  = np.arange(len(test_ds))

    if train_limit and len(train_idx) > train_limit:
        train_idx = rng.choice(train_idx, size=train_limit, replace=False)
    if test_limit and len(test_idx) > test_limit:
        test_idx = rng.choice(test_idx, size=test_limit, replace=False)

    X_train = [train_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in train_idx]
    y_train = [int(train_ds[i][1]) for i in train_idx]
    X_test  = [test_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in test_idx]
    y_test  = [int(test_ds[i][1]) for i in test_idx]
    return X_train, X_test, y_train, y_test


