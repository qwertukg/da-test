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

import math, random, hashlib
from collections import deque
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from dl_utils import cosbin
from layout_rerun import Layout2DRerun


import random
from typing import Dict, List, Set, Tuple, Optional

from rkse_sobel_layout import Layout2DNew


class PrimaryEncoderKeyholeSobel:
    """
    Первичный энкодер на одном канале ORIENT (Собель) через «замочную скважину».

    Идея:
      - Берём K скважин (малые окна S×S) вблизи центров на изображении.
      - Для каждой скважины считаем направления градиента (Собель), строим
        взвешенную гистограмму по orient_bins бинам, выбираем доминирующий бин.
      - Для тройки (y, x, bin) активируем заранее выделенные биты (bits_per_keyhole шт.).
      - Объединяем биты по всем скважинам → получаем разрежённый код (set[int]).

    Параметры:
      img_hw: (H, W) размер изображения (например, (28, 28) для MNIST).
      bits: размер «вселенной» битов (общее число возможных индексов битов).
      keyholes_per_img: количество скважин K на изображение.
      keyhole_size: размер окна скважины (нечётный), обычно 5.
      orient_bins: число ориентационных корзин (направлений), напр. 8 или 12.
      bits_per_keyhole: сколько бит выдаёт одна тройка (y, x, bin).
      mag_thresh: порог «средней силы градиента» для принятия скважины.
      max_active_bits: опциональное ограничение на итоговое число активных битов (top-K).
      deterministic: детерминированный выбор скважин в режиме random (по хэшу изображения).
      seed: общий сид ГСЧ.
      unique_bits: если True — каждому (y,x,bin) назначаются уникальные (глобально) биты.
      centers_mode: "grid" (равномерная решётка центров, одинаковая для всех картинок)
                    или "random" (случайный выбор центров).
      grid_shape: (Gh, Gw) если centers_mode="grid"; например (14,14) для 196 скважин.

    Атрибуты, полезные для визуализаций:
      bit2yxb: Dict[int, (y, x, bin)] — обратная карта для каждого выделенного бита.
               С её помощью можно окрасить «пинвил» по ориентациям и позициям.
    """

    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 8192,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 orient_bins: int = 12,
                 bits_per_keyhole: int = 1,
                 mag_thresh: float = 0.10,
                 max_active_bits: Optional[int] = None,
                 deterministic: bool = False,
                 seed: int = 42,
                 unique_bits = True,
                 centers_mode: str = "grid",           # "grid" или "random"
                 grid_shape: Optional[Tuple[int, int]] = None):
        # --- геометрия/пул битов ---
        self.H, self.W = img_hw
        self.B = bits
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.orient_bins = int(orient_bins)
        self.bits_per_keyhole = int(bits_per_keyhole)
        self.mag_thresh = float(mag_thresh)
        self.max_active_bits = max_active_bits if max_active_bits is None else int(max_active_bits)

        # --- выбор центров скважин ---
        self.deterministic = bool(deterministic)
        self.centers_mode = centers_mode
        self.grid_shape = grid_shape  # (Gh, Gw) или None
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        self.unique_bits = unique_bits
        self.bit2info = {}  # bit -> {"y":int,"x":int,"bin":int}
        self._next_bit = 0

        # --- кодбук и обратная карта ---
        self._codebook: Dict[Tuple[int, int, int], List[int]] = {}
        self._used_bits: Set[int] = set()
        self.bit2yxb: Dict[int, Tuple[int, int, int]] = {}

    # ======================================================================
    # Публичный API
    # ======================================================================

    def encode(self, img: np.ndarray) -> Set[int]:
        """
        img: 2D numpy, float32/float64, значения в [0,1], размер (H, W).
        Возвращает: множество активных битов (set[int]).
        """
        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)  # диапазон [0, 2π)

        # Нормируем magnitude, чтобы порог был сопоставим на разных картинках.
        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        centers = self._pick_keyholes(img)  # список (cy, cx)
        active: Set[int] = set()
        kept_any = False

        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            # средняя «сила» скважины — фильтруем слабые
            if float(np.mean(tile_m)) < self.mag_thresh:
                continue

            # индексы бинов направлений (веса = tile_m)
            bidx = np.floor((tile_a / (2 * np.pi)) * self.orient_bins).astype(int) % self.orient_bins
            # быстрая гистограмма с весами
            hist = np.zeros(self.orient_bins, dtype=np.float32)
            for b in range(self.orient_bins):
                hist[b] = float(tile_m[bidx == b].sum())
            b_max = int(np.argmax(hist))

            # назначаем биты под (y, x, b_max)
            for bit in self._bits_for(cy, cx, b_max):
                active.add(bit)

            kept_any = True

        # fallback: если все окна были «слабыми», возьмём самую сильную скважину
        if not kept_any and centers:
            strengths = []
            for (cy, cx) in centers:
                y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
                strengths.append((float(np.mean(mnorm[y0:y1, x0:x1])), (cy, cx)))
            strengths.sort(reverse=True)
            (cy, cx) = strengths[0][1]
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]
            bidx = np.floor((tile_a / (2 * np.pi)) * self.orient_bins).astype(int) % self.orient_bins
            hist = np.zeros(self.orient_bins, dtype=np.float32)
            for b in range(self.orient_bins):
                hist[b] = float(tile_m[bidx == b].sum())
            b_max = int(np.argmax(hist))
            for bit in self._bits_for(cy, cx, b_max):
                active.add(bit)

        # (опционально) жёстко ограничим число активных битов
        if self.max_active_bits is not None and len(active) > self.max_active_bits:
            active = set(sorted(active)[: self.max_active_bits])

        return active

    # ======================================================================
    # Служебные методы
    # ======================================================================

    def _pick_keyholes(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Вернёт список центров скважин (cy, cx)."""
        pad = self.S // 2

        if self.centers_mode == "grid":
            # равномерная решётка (одинакова для всех изображений)
            if self.grid_shape is None:
                # находим квадратную решётку примерно под K
                side = int(round(np.sqrt(self.K)))
                gh, gw = max(1, side), max(1, side)
            else:
                gh, gw = self.grid_shape
                gh = max(1, int(gh)); gw = max(1, int(gw))

            ys = np.linspace(pad, self.H - 1 - pad, gh).round().astype(int)
            xs = np.linspace(pad, self.W - 1 - pad, gw).round().astype(int)
            centers = [(int(y), int(x)) for y in ys for x in xs]
            # обрежем до K при необходимости
            if len(centers) > self.K:
                centers = centers[: self.K]
            return centers

        # режим "random"
        ys = list(range(pad, self.H - pad))
        xs = list(range(pad, self.W - pad))
        all_centers = [(y, x) for y in ys for x in xs]
        if not all_centers:
            return []

        if self.deterministic:
            # сид от содержимого изображения (стабильно между запусками)
            import hashlib
            h = hashlib.sha256(img.astype(np.float32).tobytes()).digest()
            seed_img = int.from_bytes(h[:8], "little") ^ self.seed
            rng = random.Random(seed_img)
        else:
            rng = self.rng

        k = min(self.K, len(all_centers))
        return rng.sample(all_centers, k)

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        r = self.S // 2
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

    # замена _bits_for (гарантируем уникальность и регистрируем метаданные)
    def _bits_for(self, y: int, x: int, b: int) -> List[int]:
        key = (int(y), int(x), int(b))
        if key not in self._codebook:
            if self.unique_bits:
                ids = []
                for _ in range(self.bits_per_keyhole):
                    if self._next_bit >= self.B:
                        break
                    bit = self._next_bit
                    self._next_bit += 1
                    ids.append(bit)
                    self.bit2info[bit] = {"y": int(y), "x": int(x), "bin": int(b)}
                self._codebook[key] = ids
            else:
                self._codebook[key] = self._rand_bits(self.bits_per_keyhole)
                for bit in self._codebook[key]:
                    self.bit2info.setdefault(bit, {"y": int(y), "x": int(x), "bin": int(b)})
        return self._codebook[key]

    # вспомогательное: угол/селективность из кода
    def code_dominant_orientation(self, code: set[int]) -> tuple[float, float]:
        """Возвращает (угол в [0, π), селективность в [0,1])."""
        if not code:
            return 0.0, 0.0
        # считаем глобальную гистограмму по ориентационным корзинам
        hist = np.zeros(self.orient_bins, dtype=np.float32)
        for bit in code:
            info = self.bit2info.get(int(bit))
            if info is None:
                continue
            hist[info["bin"]] += 1.0
        if hist.sum() <= 0:
            return 0.0, 0.0
        b_max = int(hist.argmax())
        # ориентация — без направленности (π-периодическая): центр корзины
        angle = (b_max + 0.5) * (np.pi / self.orient_bins)  # ∈ [0, π)
        selectivity = float(hist[b_max] / (hist.sum() + 1e-9))  # чем ближе к 1, тем «уже» тюнинг
        return angle, selectivity

    def _alloc_unique_bits(self, k: int) -> List[int]:
        """Выделяет k ранее неиспользованных битов из диапазона [0, B)."""
        out: List[int] = []
        tries = 0
        while len(out) < k:
            tries += 1
            # быстрый выход, если битов не хватает
            if tries > 10000 and (self.B - len(self._used_bits)) < (k - len(out)):
                raise RuntimeError(
                    "Недостаточно свободных битов при unique_bits=True: "
                    f"нужно ещё {k - len(out)}, свободно {self.B - len(self._used_bits)}."
                )
            cand = self.rng.randrange(self.B)
            if cand in self._used_bits:
                continue
            self._used_bits.add(cand)
            out.append(cand)
        return out

    def _rand_bits(self, k: int) -> List[int]:
        """Выдаёт k случайных битов (могут повторяться для разных (y,x,bin))."""
        # Без уникальности: допускаем переиспользование битов на разных ключах
        return self.rng.sample(range(self.B), k)

    # ---------- Собель 3×3 ----------

    @staticmethod
    def _sobel(img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[ 1, 2, 1],
                       [ 0, 0, 0],
                       [-1,-2,-1]], dtype=np.float32)
        gx = PrimaryEncoderKeyholeSobel._conv2_same(img01, kx)
        gy = PrimaryEncoderKeyholeSobel._conv2_same(img01, ky)
        return gx, gy

    @staticmethod
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

    # ---------- Вспомогательные методы (необязательные) ----------

    def get_keyhole_centers_grid(self) -> List[Tuple[int, int]]:
        """Вернёт центры скважин для текущей конфигурации в режиме 'grid' (полезно для отладки)."""
        pad = self.S // 2
        if self.grid_shape is None:
            side = int(round(np.sqrt(self.K)))
            gh, gw = max(1, side), max(1, side)
        else:
            gh, gw = self.grid_shape
            gh = max(1, int(gh)); gw = max(1, int(gw))
        ys = np.linspace(pad, self.H - 1 - pad, gh).round().astype(int)
        xs = np.linspace(pad, self.W - 1 - pad, gw).round().astype(int)
        centers = [(int(y), int(x)) for y in ys for x in xs]
        return centers[: self.K]


# ============================================================
# 1) Первичное кодирование (каналы + "цветовое" объединение)
# ============================================================

# class PrimaryEncoder:
#     """
#     Каналы:
#       - BRIGHT: глобальная яркость (квантование на bright_levels)
#       - COARSE: 2×2 грубая сетка (наличие штриха)
#       - ORIENT: ориентиры Собрела, доминантный угол на coarse-сетке orient_grid × orient_grid
#       - GRID: G×G, квантование яркости на grid_levels
#     "Цветовое" объединение: упорядочиваем (priority, bit) и берём top-K (max_active_bits).
#     """
#     def __init__(self,
#                  img_hw=(8, 8),
#                  bits=8192,
#                  seed=42,
#                  # GRID
#                  grid_g=4, grid_levels=4, grid_bits_per_cell=3,
#                  # COARSE
#                  coarse_bits_per_cell=4,
#                  # BRIGHT
#                  bright_levels=8,
#                  # ORIENT (Sobel)
#                  orient_on=True, orient_bins=8, orient_grid=4,
#                  orient_bits_per_cell=2, orient_mag_thresh=0.12,
#                  # итоговый лимит
#                  max_active_bits=260):
#         self.H, self.W = img_hw
#         self.B = bits
#         self.rng = random.Random(seed)
#
#         self.G = grid_g
#         self.grid_levels = grid_levels
#         self.grid_bits_per_cell = grid_bits_per_cell
#
#         self.coarse_bits_per_cell = coarse_bits_per_cell
#         self.bright_levels = bright_levels
#
#         self.orient_on = orient_on
#         self.orient_bins = orient_bins
#         self.orient_grid = orient_grid
#         self.orient_bits_per_cell = orient_bits_per_cell
#         self.orient_mag_thresh = orient_mag_thresh
#
#         self.max_active_bits = max_active_bits
#
#         # Кодбуки (стабильны от seed)
#         self.grid_codebook: Dict[Tuple[int, int, int], List[int]] = {}
#         for y in range(self.G):
#             for x in range(self.G):
#                 for lvl in range(self.grid_levels):
#                     self.grid_codebook[(y, x, lvl)] = self._rand_bits(self.grid_bits_per_cell)
#
#         self.coarse_codebook: Dict[Tuple[int, int], List[int]] = {}
#         for y in range(2):
#             for x in range(2):
#                 self.coarse_codebook[(y, x)] = self._rand_bits(self.coarse_bits_per_cell)
#
#         self.bright_codebook: Dict[int, List[int]] = {}
#         for lvl in range(self.bright_levels):
#             self.bright_codebook[lvl] = self._rand_bits(4)  # 4 бита на BRIGHT
#
#         self.orient_codebook: Dict[Tuple[int, int, int], List[int]] = {}
#         if self.orient_on:
#             for y in range(self.orient_grid):
#                 for x in range(self.orient_grid):
#                     for b in range(self.orient_bins):
#                         self.orient_codebook[(y, x, b)] = self._rand_bits(self.orient_bits_per_cell)
#
#     def _rand_bits(self, k: int) -> List[int]:
#         return self.rng.sample(range(self.B), k)
#
#     @staticmethod
#     def _quantize01(v: float, levels: int) -> int:
#         x = max(0.0, min(1.0, v))
#         return min(levels - 1, int(x * levels))
#
#     def encode(self, img: np.ndarray) -> Set[int]:
#         """
#         img: 2D numpy (H×W) с яркостями [0..1]
#         -> set[int] активных битов
#         """
#         H, W, G = self.H, self.W, self.G
#         ph: List[Tuple[float, int]] = []  # (priority, bit)
#
#         # --- BRIGHT (priority 0.0) ---
#         mean_b = float(np.mean(img))
#         lvl_b = self._quantize01(mean_b, self.bright_levels)
#         for b in self.bright_codebook[lvl_b]:
#             ph.append((0.0, b))
#
#         # --- COARSE 2×2 (priority 1.0) ---
#         h2, w2 = H // 2, W // 2
#         blocks = [
#             img[0:h2, 0:w2], img[0:h2, w2:W],
#             img[h2:H, 0:w2], img[h2:H, w2:W]
#         ]
#         for i, blk in enumerate(blocks):
#             y, x = divmod(i, 2)
#             if float(np.mean(blk)) > 0.15:
#                 for b in self.coarse_codebook[(y, x)]:
#                     ph.append((1.0, b))
#
#         # --- ORIENT (priority 1.5) ---
#         if self.orient_on:
#             mag, ang = self._sobel_mag_dir(img)
#             gy, gx = self.H // self.orient_grid, self.W // self.orient_grid
#             mmax = float(np.max(mag))
#             mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag
#             bins = np.linspace(0.0, 2 * np.pi, self.orient_bins + 1)
#             for y in range(self.orient_grid):
#                 for x in range(self.orient_grid):
#                     tile_m = mnorm[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
#                     tile_a = ang[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
#                     m = float(np.mean(tile_m))
#                     if m < self.orient_mag_thresh:
#                         continue
#                     hist = np.zeros(self.orient_bins, dtype=np.float32)
#                     h, w = tile_m.shape
#                     for iy in range(h):
#                         for ix in range(w):
#                             b = int(np.clip(np.searchsorted(bins, tile_a[iy, ix], side='right') - 1,
#                                             0, self.orient_bins - 1))
#                             hist[b] += tile_m[iy, ix]
#                     b_max = int(np.argmax(hist))
#                     for bit in self.orient_codebook[(y, x, b_max)]:
#                         ph.append((1.5, bit))
#
#         # --- GRID G×G (priority 2.0) ---
#         gy, gx = H // G, W // G
#         for y in range(G):
#             for x in range(G):
#                 tile = img[y * gy:(y + 1) * gy, x * gx:(x + 1) * gx]
#                 m = float(np.mean(tile))
#                 lvl = self._quantize01(m, self.grid_levels)
#                 lvls = {lvl}
#                 if lvl > 0: lvls.add(lvl - 1)  # лёгкое перекрытие
#                 for lv in lvls:
#                     for b in self.grid_codebook[(y, x, lv)]:
#                         ph.append((2.0, b))
#
#         # --- "цветовое" объединение (top-K по приоритетам) ---
#         ph.sort(key=lambda z: z[0])
#         out: Set[int] = set()
#         for _, bit in ph:
#             if len(out) >= self.max_active_bits: break
#             out.add(bit)
#         return out
#
#     # --- служебные методы ---
#     @staticmethod
#     def _conv2_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
#         ph, pw = k.shape[0] // 2, k.shape[1] // 2
#         pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
#         H, W = img.shape
#         out = np.zeros_like(img, dtype=np.float32)
#         for y in range(H):
#             for x in range(W):
#                 patch = pad[y:y + k.shape[0], x:x + k.shape[1]]
#                 out[y, x] = float(np.sum(patch * k))
#         return out
#
#     @classmethod
#     def _sobel_mag_dir(cls, img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#         ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float32)
#         gx = cls._conv2_same(img01, kx)
#         gy = cls._conv2_same(img01, ky)
#         mag = np.hypot(gx, gy)
#         ang = (np.arctan2(gy, gx) + np.pi)
#         return mag, ang

# ============================================================
# 2) Раскладка (упрощённый DAMP с локальными свапами и override)
# ============================================================

# class Layout2D:
#     """
#     Укладывает N кодов на дискретную сетку h×w (h*w>=N).
#     Оптимизация: многократно перебираем случайные непересекающиеся пары занятых ячеек
#     и меняем местами их элементы, если это снижает локальную "энергию":
#        sum_{соседи} sim(code_i, code_nbr) * dist(cell_i, cell_nbr)
#     Два режима: FAR (радиус R_far) и NEAR (радиус R_near).
#     """
#     def __init__(self, R_far=7, R_near=3, epochs_far=8, epochs_near=6, seed=123):
#         self.R_far = R_far
#         self.R_near = R_near
#         self.E_far = epochs_far
#         self.E_near = epochs_near
#         self.rng = random.Random(seed)
#         self.shape: Tuple[int, int] = (0, 0)
#         self.cell2idx: List[Optional[int]] = []
#         self.idx2cell: Dict[int, Tuple[int, int]] = {}
#
#     @staticmethod
#     def _grid_shape(n: int) -> Tuple[int, int]:
#         s = math.ceil(math.sqrt(n))
#         return (s, s)
#
#     def _neighbors(self, y: int, x: int, R: int) -> List[Tuple[int, int]]:
#         H, W = self.shape
#         out = []
#         for dy in range(-R, R + 1):
#             for dx in range(-R, R + 1):
#                 if dy == 0 and dx == 0: continue
#                 ny, nx = y + dy, x + dx
#                 if 0 <= ny < H and 0 <= nx < W:
#                     out.append((ny, nx))
#         return out
#
#     @staticmethod
#     def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
#         return math.hypot(a[0] - b[0], a[1] - b[1])
#
#     def _local_energy(self,
#                       yx: Tuple[int, int],
#                       center_idx: int,
#                       R: int,
#                       sim_cache: Dict[Tuple[int, int], float],
#                       override: Optional[Dict[Tuple[int, int], int]] = None) -> float:
#         """
#         Энергия клетки yx с "центральным" кодом center_idx как сумма sim*dist
#         по соседям в радиусе R. Коды берём по индексам владельцев self._cell_owner,
#         для "виртуального свапа" можно передать override: {(y,x)->idx}.
#         """
#         y, x = yx
#         e = 0.0
#         ci = override.get((y, x), center_idx) if override else center_idx
#         code_c = self._codes[ci]
#         for ny, nx in self._neighbors(y, x, R):
#             jdx = (override.get((ny, nx), self._cell_owner.get((ny, nx)))
#                    if override else self._cell_owner.get((ny, nx)))
#             if jdx is None:
#                 continue
#             a, b = (ci, jdx) if ci <= jdx else (jdx, ci)
#             s = sim_cache.get((a, b))
#             if s is None:
#                 s = cosbin(code_c, self._codes[jdx])
#                 sim_cache[(a, b)] = s
#             e += s * self._dist((y, x), (ny, nx))
#         return e
#
#     def fit(self, codes: List[Set[int]]):
#         n = len(codes)
#         H, W = self._grid_shape(n)
#         self.shape = (H, W)
#         self._codes = codes  # держим ссылку на коды
#
#         # начальная укладка — построчно
#         cells = [(y, x) for y in range(H) for x in range(W)]
#         self.cell2idx = [None] * (H * W)
#         self.idx2cell = {}
#         for i in range(n):
#             y, x = cells[i]
#             self.cell2idx[i] = i
#             self.idx2cell[i] = (y, x)
#
#         # владельцы ячеек
#         self._cell_owner: Dict[Tuple[int, int], Optional[int]] = {}
#         for yx in cells:
#             self._cell_owner[yx] = None
#         for i in range(n):
#             y, x = self.idx2cell[i]
#             self._cell_owner[(y, x)] = i
#
#         def pass_epoch(R: int, iters: int):
#             for _ in range(iters):
#                 occupied = list(self.idx2cell.items())  # (idx -> (y,x)), длина = n
#                 self.rng.shuffle(occupied)
#                 pairs = []
#                 for i in range(0, len(occupied) - 1, 2):
#                     (ia, yxa), (ib, yxb) = occupied[i], occupied[i + 1]
#                     pairs.append((ia, yxa, ib, yxb))
#                 sim_cache: Dict[Tuple[int, int], float] = {}
#                 for ia, yxa, ib, yxb in pairs:
#                     e_cur = (
#                         self._local_energy(yxa, ia, R, sim_cache) +
#                         self._local_energy(yxb, ib, R, sim_cache)
#                     )
#                     override = {yxa: ib, yxb: ia}
#                     e_swp = (
#                         self._local_energy(yxa, ib, R, sim_cache, override=override) +
#                         self._local_energy(yxb, ia, R, sim_cache, override=override)
#                     )
#                     if e_swp + 1e-9 < e_cur:
#                         # применяем свап
#                         self.idx2cell[ia], self.idx2cell[ib] = yxb, yxa
#                         self._cell_owner[yxa], self._cell_owner[yxb] = ib, ia
#
#         # FAR порядок (несколько проходов большим радиусом)
#         pass_epoch(self.R_far, self.E_far)
#         # NEAR порядок (полировка)
#         pass_epoch(self.R_near, self.E_near)
#         return self
#
#     def grid_shape(self) -> Tuple[int, int]:
#         return self.shape
#
#     def position_of(self, idx: int) -> Tuple[int, int]:
#         return self.idx2cell[idx]

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
                 layout: Layout2DNew,
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
            comps = self._connected_components(act)
            for comp in comps:
                if len(comp) < self.min_comp: continue
                center_c, radius_c = self._comp_center_radius(comp)
                (cent_e, (u1, u2), (r1, r2)) = self._pca_ellipse(comp, pad_scale=1.6)
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

    # --- служебные методы ---
    @staticmethod
    def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        H, W = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        for y in range(H):
            for x in range(W):
                if not mask[y, x] or seen[y, x]:
                    continue
                q = deque([(y, x)])
                seen[y, x] = True
                comp = [(y, x)]
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                                comp.append((ny, nx))
                comps.append(comp)
        return comps

    @staticmethod
    def _comp_center_radius(comp: List[Tuple[int, int]]) -> Tuple[Tuple[float, float], float]:
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        r = 0.0
        for (y, x) in comp:
            r = max(r, math.hypot(y - cy, x - cx))
        return (cy, cx), (r * 1.15 + 0.5)

    @staticmethod
    def _pca_ellipse(points: List[Tuple[int, int]], pad_scale: float = 1.6):
        ys = np.array([p[0] for p in points], dtype=np.float32)
        xs = np.array([p[1] for p in points], dtype=np.float32)
        cy, cx = float(ys.mean()), float(xs.mean())
        Y = np.stack([ys - cy, xs - cx], axis=0)
        if Y.shape[1] < 3:
            return (cy, cx), (np.array([1.0, 0.0]), np.array([0.0, 1.0])), (1.0, 1.0)
        C = (Y @ Y.T) / max(1, Y.shape[1] - 1)
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 1e-6, None)
        r1 = pad_scale * float(np.sqrt(vals[1]))
        r2 = pad_scale * float(np.sqrt(vals[0]))
        u1 = vecs[:, 1] / np.linalg.norm(vecs[:, 1])
        u2 = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
        return (cy, cx), (u1, u2), (r1, r2)

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
            sims = [(self.jaccard(q, x), yi) for x, yi in zip(self.X, self.y)]
            sims.sort(key=lambda z: -z[0])
            top = sims[:self.k]
            score: Dict[int, float] = {}
            for s, yi in top:
                score[yi] = score.get(yi, 0.0) + (s + self.eps)
            out.append(max(score.items(), key=lambda z: z[1])[0])
        return out

    @staticmethod
    def jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0


