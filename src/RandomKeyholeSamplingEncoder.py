import random
from typing import Dict, List, Set, Tuple

import numpy as np
import math, hashlib


class RandomKeyholeSamplingEncoder:
    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 256,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 bits_per_keyhole: int = 2,
                 seed: int = 42,
                 ):

        self.H, self.W = img_hw
        self.B = bits
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.bits_per_keyhole = int(bits_per_keyhole)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        # ---- Параметры «широких детекторов» для угла (гл. 4) ----
        # Несколько слоёв (масштабов) ширины дуги: ближний и дальний порядок
        self.angle_layers = [  # (число детекторов на слое, полуширина дуги)
            (64, float(np.pi / 16)),   # высокая «разрешающая» способность
            (32, float(np.pi / 8)),    # средняя
            (16, float(np.pi / 4)),    # дальний порядок (сглаживание)
            (8,  float(np.pi / 2)),    # ДОБАВЛЕНО: широкий «красный» слой
            (4,  float(3 * np.pi / 4)) # ДОБАВЛЕНО: ещё более широкий «красный» слой
        ]
        self.bits_per_detector = 3            # сколько битов выдаёт один детектор
        self.max_detectors_per_center = 8     # максимум активных детекторов на одну скважину
        self._angle_detectors: List[Tuple[float, float, List[int]]] = []
        self._build_angle_detectors()

        # ---- Пороговые параметры для отсеивания «плоских» окон и адаптивного добора ----
        self.mag_eps: float = 0.03          # пиксель «ненулевой», если m > mag_eps
        self.min_active_frac: float = 0.05  # мин. доля таких пикселей в окне (иначе окно «плоское»)
        self.adaptive_fill: bool = True     # ослаблять пороги, чтобы добрать до K
        self.adaptive_decay: float = 0.5    # во сколько раз уменьшать пороги, если не хватает

        self.bit2info: Dict[int, Dict[str, float]] = {}
        self._codebook: Dict[Tuple[int, int, float], List[int]] = {}

    def encode(self, img: np.ndarray) -> Set[int]:
        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)

        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        # ВАЖНО: берём центры только из неплоских окон и адаптивно добиваем до K
        centers = self._pick_keyholes(mnorm)
        active: Set[int] = set()

        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            angle = self._dominant_angle(tile_a, tile_m)
            # глобальные угловые детекторы (независимо от координат)
            for bit in self._angle_bits(angle):
                active.add(bit)
                self.bit2info.setdefault(bit, {"angle": float(angle)})  # опционально для отладки

        self.print_barcode(active, float(angle))

        return active

    # ---------- выбор центров с пропуском плоских и адаптивным добором ----------
    def _pick_keyholes(self, mnorm: np.ndarray) -> List[Tuple[int, int]]:
        H, W = mnorm.shape
        pad = self.S // 2

        # собираем кандидатов с их «неплоскостью»
        # entry: (score, frac, mean, y, x)
        candidates: List[Tuple[float, float, float, int, int]] = []
        for y in range(pad, H - pad):
            for x in range(pad, W - pad):
                y0, y1, x0, x1 = self._window_bounds(y, x, H, W)
                tile = mnorm[y0:y1, x0:x1]
                mean = float(tile.mean())
                frac = float((tile > self.mag_eps).mean())
                score = 0.5 * mean + 0.5 * frac  # гибридная метрика «неплоскости»
                candidates.append((score, frac, mean, y, x))

        # сортируем по убыванию «качества»
        candidates.sort(key=lambda t: t[0], reverse=True)

        thr_frac = float(self.min_active_frac)
        thr_mean = float(self.mag_eps)

        while True:
            picked: List[Tuple[int, int]] = []
            for (_, frac, mean, y, x) in candidates:
                if (frac >= thr_frac) and (mean >= thr_mean):
                    picked.append((y, x))
                    if len(picked) >= self.K:
                        break

            if len(picked) >= self.K or not self.adaptive_fill:
                return picked[: self.K]

            # если не хватает — ослабляем пороги и повторяем
            thr_frac *= self.adaptive_decay
            thr_mean *= self.adaptive_decay

            # пороги обнулились — дальше нечего ослаблять; вернём сколько нашли
            if thr_frac <= 0.0 and thr_mean <= 0.0:
                return picked  # может быть < K на полностью гладком изображении

    # ---------------------------------------------------------------------------

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        r = self.S // 2
        y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

    def _bits_for(self, y: int, x: int, angle: float) -> List[int]:
        key = (int(y), int(x), float(angle))
        if key not in self._codebook:
            self._codebook[key] = self._rand_bits(self.bits_per_keyhole)
            for bit in self._codebook[key]:
                self.bit2info.setdefault(bit, {"y": int(y), "x": int(x), "angle": float(angle)})
        return self._codebook[key]

    def code_dominant_orientation(self, code: set[int]) -> tuple[float, float]:
        if not code:
            return (0.0, 0.0)
        angles: List[float] = []
        for bit in code:
            info = self.bit2info.get(int(bit))
            if info is not None:
                angles.append(float(info.get("angle", 0.0)))
        if not angles:
            return (0.0, 0.0)
        vectors = np.exp(1j * np.array(angles, dtype=np.float64))
        vec_sum = vectors.sum()
        angle = float(np.angle(vec_sum) % (2 * np.pi))
        selectivity = float(abs(vec_sum) / len(vectors))
        return angle, selectivity

    def _rand_bits(self, k: int) -> List[int]:
        return self.rng.sample(range(self.B), k)

    @staticmethod
    def _sobel(img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)
        gx = RandomKeyholeSamplingEncoder._conv2_same(img01, kx)
        gy = RandomKeyholeSamplingEncoder._conv2_same(img01, ky)
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

    @staticmethod
    def _dominant_angle(tile_a: np.ndarray, tile_m: np.ndarray) -> float:
        weights = tile_m.astype(np.float64)
        total = float(weights.sum())
        if total <= 1e-8:
            return 0.0
        vectors = np.exp(1j * tile_a.astype(np.float64)) * weights
        vec_sum = vectors.sum()
        if vec_sum == 0:
            return 0.0
        angle = float(np.angle(vec_sum) % (2 * np.pi))
        return angle

    # ==================== ШИРОКИЕ ДЕТЕКТОРЫ УГЛА ====================

    def _build_angle_detectors(self) -> None:
        """Строим список детекторов: (центр дуги mu, полуширина width, закреплённые биты)."""
        self._angle_detectors.clear()
        for L, (n, width) in enumerate(self.angle_layers):
            for j in range(n):
                mu = 2.0 * np.pi * float(j) / float(n)  # равномерно по окружности
                bits = self._det_bits(f"ANG_DET:{L}:{j}:{self.seed}", self.bits_per_detector)
                self._angle_detectors.append((float(mu), float(width), bits))

    @staticmethod
    def _circ_delta(a: float, b: float) -> float:
        """Минимальная круговая разность углов (0..π]."""
        return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    def _det_bits(self, key: str, k: int) -> List[int]:
        """Детерминированное отображение детектора в k битов (устойчиво между запусками)."""
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest(), 'little')
        step = 0x9E3779B97F4A7C15  # «золотое» смещение
        return [int((h + i * step) % self.B) for i in range(k)]

    def _angle_bits(self, angle: float) -> List[int]:
        """Выдаёт биты детекторов, активных для данного угла (без квантизации угла)."""
        # 1) бинарная активация детекторов, у которых угол попадает в дугу
        candidates: List[Tuple[float, List[int]]] = []
        for (mu, width, bits) in self._angle_detectors:
            d = self._circ_delta(angle, mu)
            if d <= width:
                # чем ближе к центру дуги — тем выше приоритет
                candidates.append((float(width - d), bits))

        # 2) если ни один не попал — берём ближайшие по «гауссовой» близости
        if not candidates:
            tmp: List[Tuple[float, List[int]]] = []
            for (mu, width, bits) in self._angle_detectors:
                d = self._circ_delta(angle, mu)
                s = float(math.exp(-0.5 * (d / (width + 1e-9)) ** 2))
                tmp.append((s, bits))
            tmp.sort(key=lambda t: t[0], reverse=True)
            candidates = tmp[: self.max_detectors_per_center]
        else:
            candidates.sort(key=lambda t: t[0], reverse=True)
            candidates = candidates[: self.max_detectors_per_center]

        ids: List[int] = []
        for _, bits in candidates:
            ids.extend(bits)
        return ids

    def print_barcode(self, active, ang):
        # --- Печать «штрихкода»: '|' для 1 и ' ' для 0 ---
        bar = [' '] * self.B
        for b in active:
            bi = int(b)
            if 0 <= bi < self.B:
                bar[bi] = '|'
        print(''.join(bar))
        print(ang)
