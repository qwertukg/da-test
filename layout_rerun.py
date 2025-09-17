# layout_rerun.py
# -*- coding: utf-8 -*-
"""
Раскладка (Layout2D) с колбэками для логирования в Rerun:
- on_epoch(phase: str, epoch_idx: int, lay: Layout2D) -> None
- on_swap(a_pos: (y,x), b_pos: (y,x), phase: str, epoch_idx: int, lay: Layout2D) -> None

Если Rerun не установлен, хелперы rr_* тихо отключаются.
"""

from __future__ import annotations

import colorsys
import hashlib
import math, random
from typing import List, Tuple, Dict, Set, Optional, Callable

import numpy as np
from matplotlib.colors import hsv_to_rgb

from dl_utils import cosbin

# === Опциональные хелперы для Rerun =========================================
try:
    import rerun as rr
    _RR_OK = True
except Exception:
    _RR_OK = False

def _hsv_to_rgb_u8(h, s, v):
    # h∈[0,1), s,v∈[0,1] -> uint8 RGB
    i = int(h*6.0) % 6
    f = h*6.0 - i
    p = v*(1.0 - s)
    q = v*(1.0 - f*s)
    t = v*(1.0 - (1.0 - f)*s)
    if   i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else:        r,g,b = v,p,q
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)

def rr_log_layout_orientation(lay, codes_train, enc, tag="layout", step=None):
    if step is not None:
        rr.set_time_sequence("epoch", int(step))
    positions, colors = [], []
    for idx, (y, x) in lay.idx2cell.items():
        angle, sel = enc.code_dominant_orientation(codes_train[idx])
        hue = (angle / np.pi) % 1.0           # 0..1
        sat = float(np.clip(sel, 0.2, 1.0))   # не даём полностью серых точек
        val = 1.0
        rgb = _hsv_to_rgb_u8(hue, sat, val)
        positions.append([float(x), float(y)])   # (x, y)
        colors.append(rgb)
    positions = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    rr.log(f"layout", rr.Points2D(positions=positions, colors=colors))

# 1) Получаем «любимую» ориентацию образца из его кода (мода по бинам)
def preferred_orientation_from_code(enc, code_bits: set[int]):
    bins = []
    for b in code_bits:
        yxb = enc.bit2yxb.get(b)  # <- нужна обратная карта в энкодере
        if yxb is None:
            continue
        _, _, bb = yxb
        bins.append(int(bb))
    if not bins:
        return None
    b_max = int(np.bincount(np.array(bins)).argmax())
    # центр бина → угол [0..2π)
    theta = 2.0*np.pi*(b_max + 0.5) / enc.orient_bins
    return theta

# 2) Лог «пинвила»: точки = позиции образцов на решётке, цвет = оттенок по ориентации
def rr_log_pinwheel(lay, enc, codes_train, tag: str, step: int):
    pts = []
    cols = []
    for idx, (y, x) in lay.idx2cell.items():
        theta = preferred_orientation_from_code(enc, codes_train[idx])
        if theta is None:
            rgb = (128, 128, 128)
        else:
            h = float(theta / (2*np.pi))     # 0..1
            rgb = tuple((hsv_to_rgb([h, 1.0, 1.0]) * 255).astype(np.uint8))
        pts.append([x, y])
        cols.append(rgb)
    rr.set_time_sequence("epoch", step)
    rr.log("layout", rr.Points2D(positions=np.array(pts), colors=np.array(cols, dtype=np.uint8)))

def rr_init(app_name: str = "digits-layout", spawn: bool = True, class_labels=None):
    """Инициализирует Rerun Viewer (опционально)."""
    if not _RR_OK: return
    rr.init(app_name, spawn=spawn)
    if class_labels is not None:
        # class_labels: {id:int -> label:str}
        from rerun.datatypes import AnnotationInfo
        ann = [AnnotationInfo(id=int(i), label=str(lbl)) for i, lbl in class_labels.items()]
        rr.log("layout", rr.AnnotationContext(ann), static=True)

def code_to_rgb_from_hash(code_bits) -> np.ndarray:
    """
    code_bits: set[int] или list[int] — активные биты первичного кода
    → np.uint8[3], стабильный «морфологический» цвет.
    """
    if not code_bits:
        return np.array([180, 180, 180], dtype=np.uint8)  # серый для пустых

    # Стабильный байтовый поток: отсортировали, упаковали
    arr = np.fromiter(sorted(code_bits), dtype=np.uint32)
    bs = arr.tobytes()

    # Короткий, но стойкий хэш. Ключ фиксируем, чтобы цвета были воспроизводимыми.
    digest = hashlib.blake2b(bs, digest_size=3, key=b"pinwheel-morph").digest()
    d0, d1, d2 = digest[0], digest[1], digest[2]

    # HSV: яркие, но не кислотные
    hue = d0 / 255.0                            # 0..1
    sat = 0.55 + 0.35 * (d1 / 255.0)            # ~0.55..0.90
    val = 0.60 + 0.35 * (d2 / 255.0)            # ~0.60..0.95

    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)

def rr_log_layout_snapshot(lay, codes_train, tag="layout", step=None):
    # 1) если хотим анимировать по шагам — выставляем "время"
    if step is not None:
        rr.set_time_sequence("epoch", int(step))   # <-- ключевая замена step=

    # 2) собираем точки и цвета
    positions = []
    colors = []
    for idx, (y, x) in lay.idx2cell.items():
        positions.append([float(x), float(y)])                   # X, Y
        colors.append(code_to_rgb_from_hash(codes_train[idx]))   # RGB uint8

    positions = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    # 3) логируем без step=
    rr.log(f"layout", rr.Points2D(positions=positions, colors=colors))

def rr_log_swap(a_pos, b_pos, phase, epoch_idx, step=None):
    if step is not None:
        rr.set_time_sequence("epoch", int(step))

    # Пример стрелки: из A в B
    origins = np.array([[float(a_pos[1]), float(a_pos[0])]], dtype=np.float32)  # (x,y)
    vecs    = np.array([[float(b_pos[1]-a_pos[1]), float(b_pos[0]-a_pos[0])]], dtype=np.float32)

    rr.log(f"layout", rr.Arrows2D(origins=origins, vectors=vecs))

# === Собственно раскладка ====================================================
OnEpochCB = Optional[Callable[[str, int, "Layout2D"], None]]
OnSwapCB  = Optional[Callable[[Tuple[int,int], Tuple[int,int], str, int, "Layout2D"], None]]

class Layout2DRerun:
    """
    Укладывает N кодов на дискретную сетку h×w (h*w>=N).
    Оптимизация: случайные непересекающиеся пары занятых ячеек; свап, если снижает локальную "энергию":
        sum_{соседи} sim(code_i, code_nbr) * dist(cell_i, cell_nbr)
    Два режима: FAR (крупный радиус) и NEAR (полировка).

    Новое:
    - fit(.., on_epoch, on_swap): колбэки для визуализации/логирования (например, в Rerun).
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
        """Энергия клетки yx с кодом center_idx по соседям в радиусе R."""
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

    def fit(self,
            codes: List[Set[int]],
            on_epoch: OnEpochCB = None,
            on_swap:  OnSwapCB  = None):
        """
        codes: список множеств активных битов (по одному на пример).
        on_epoch: вызывается после каждой «эпохи» FAR/NEAR.
        on_swap:  вызывается для каждого принятого свапа (можно внутри колбэка поредить логирование).
        """
        n = len(codes)
        H, W = self._grid_shape(n)
        self.shape = (H, W)
        self._codes = codes  # ссылка на коды

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

        # счётчик «шагов» для таймлайна (удобно в Viewer)
        step_counter = 0

        def pass_epoch(R: int, iters: int, phase: str):
            nonlocal step_counter
            for ep in range(iters):
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
                        if on_swap is not None:
                            on_swap(yxa, yxb, phase, ep, self)
                # кадр после эпохи
                if on_epoch is not None:
                    on_epoch(phase, ep, self)
                step_counter += 1

        # FAR порядок (несколько проходов большим радиусом)
        pass_epoch(self.R_far, self.E_far, phase="far")
        # NEAR порядок (полировка)
        pass_epoch(self.R_near, self.E_near, phase="near")
        return self

    def grid_shape(self) -> Tuple[int, int]:
        return self.shape

    def position_of(self, idx: int) -> Tuple[int, int]:
        return self.idx2cell[idx]