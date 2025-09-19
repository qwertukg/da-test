import random
import math
import hashlib
from typing import Optional, Dict, List, Set, Tuple

import numpy as np


class RandomKeyholeSamplingEncoder:
    """
    Эncодер «случайных скважин» с угловыми широкими детекторами (без квантизации угла).

    Идея:
      1) По всему изображению считаем градиенты Собеля (gx, gy) → модуль/угол.
      2) Выбираем K центров-«скважин» (окна S×S), отсеивая «плоские» окна по метрике неплоскости.
         При нехватке подходящих окон пороги ослабляются (опционально), чтобы добрать до K.
      3) Для каждой скважины вычисляем доминирующий угол (взвешенный модулем).
      4) Активируем биты через набор глобальных «широких детекторов» по углу (перекрывающиеся дуги).
         Биты не зависят от координат (инвариантность к сдвигу).
      5) Итоговый код — объединение битов по всем скважинам.

    Примечание:
      Координаты (y, x) влияют только на выбор патча и отсев «плоских» окон;
      позиция в сам код не зашита. Это увеличивает перекрытие между похожими углами
      в разных местах и разных изображениях.
    """

    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 256,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 seed: int = 42,
                 # ---- Детекторы по углу (настройка похожести) ----
                 angle_layers: Optional[List[Tuple[int, float]]] = None,
                 bits_per_detector: int = 8,
                 max_detectors_per_center: int = 3,
                 # ---- Отсев «плоских» окон и добор до K ----
                 mag_eps: float = 0.03,
                 min_active_frac: float = 0.05,
                 adaptive_fill: bool = True,
                 adaptive_decay: float = 0.5,
                 # ---- Поведение печати ----
                 print_code: bool = False,
                 print_angle: bool = False,
                 barcode_on: str = "|",
                 barcode_off: str = " ",
                 ):
        """
        Параметры
        ----------
        img_hw : (H, W)
            Ожидаемый размер входного изображения (серое, 2D).
        bits : int
            Размерность бинарного кода (длина вектора/число бит).
        keyholes_per_img : int
            Число скважин (окон) на изображение, которые будут учитываться.
        keyhole_size : int (нечётный)
            Размер стороны окна S×S для каждой скважины.
        seed : int
            Сид для детерминированного хэш-мэппинга детекторов в биты и RNG.

        angle_layers : список (N, width)
            Конфигурация «широких детекторов» по углу:
            - N   : число детекторов, равномерно распределённых по окружности [0, 2π)
            - width: полуширина дуги для бинарной активации (в радианах)
            Бóльшие width дают «дальний порядок» (большее типовое перекрытие),
            меньшие — «ближний порядок» (более острое различение углов).
        bits_per_detector : int
            Сколько бит выдаёт один детектор (для снижения коллизий и вариативности).
        max_detectors_per_center : int
            Сколько детекторов максимум может сработать на одну скважину
            (выбираются по приоритету: ближе к центру дуги — выше).

        mag_eps : float
            Порог пиксельной магнитуды (в нормировке [0..1]) для признака «не ноль».
        min_active_frac : float
            Минимальная доля пикселей в окне с магнитудой > mag_eps, чтобы окно не считалось «плоским».
        adaptive_fill : bool
            Если True — пороги `mag_eps` и `min_active_frac` будут понижаться (×adaptive_decay),
            чтобы добрать до K окон.
        adaptive_decay : float
            Множитель ослабления порогов (0 < adaptive_decay < 1).

        print_code : bool
            Печатать ли «штрихкод» результата ('|' и пробелы).
        print_angle : bool
            Печатать ли средний (по кругу) угол Собеля по выбранным скважинам.
        barcode_on / barcode_off : str
            Символы для «1» и «0» в строке-штрихкоде.
        """
        # --- базовые параметры ---
        self.H, self.W = img_hw
        self.B = int(bits)
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        # --- детекторы по углу ---
        if angle_layers is None:
            angle_layers = [
                (64, float(np.pi / 16)),  # высокая «разборчивость»
                (32, float(np.pi / 8)),   # средняя
                (16, float(np.pi / 4)),   # сглаживание (дальний порядок)
                (8,  float(np.pi / 2)),   # широкий слой
                (4,  float(3 * np.pi / 4))
            ]
        self.angle_layers: List[Tuple[int, float]] = [
            (int(n), float(width)) for (n, width) in angle_layers
        ]
        self.bits_per_detector = int(bits_per_detector)
        self.max_detectors_per_center = int(max_detectors_per_center)

        # --- пороги неплоскости и добор ---
        self.mag_eps = float(mag_eps)
        self.min_active_frac = float(min_active_frac)
        self.adaptive_fill = bool(adaptive_fill)
        self.adaptive_decay = float(adaptive_decay)
        assert 0.0 < self.adaptive_decay <= 1.0, "adaptive_decay должен быть в (0, 1]"

        # --- печать ---
        self.print_code = bool(print_code)
        self.print_angle = bool(print_angle)
        self.barcode_on = str(barcode_on)
        self.barcode_off = str(barcode_off)

        # --- внутренние структуры ---
        self._angle_detectors: List[Tuple[float, float, List[int]]] = []
        self._build_angle_detectors()
        self.bit2info: Dict[int, Dict[str, float]] = {}

    # ==================== ПАБЛИК API ====================

    def encode(self, img: np.ndarray) -> Set[int]:
        """
        Закодировать изображение в множество активных битов.

        Параметры
        ----------
        img : np.ndarray shape (H, W), dtype float/uint8...
            Одноканальное изображение ожидаемого размера `img_hw`.
            Если значения не нормированы, это не критично — внутри Собель и
            нормировка магнитуды выполнены автоматически.

        Возврат
        -------
        active : Set[int]
            Множество индексов активных битов (0 <= bit < bits).
        """
        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        # 1) Градиенты Собеля, модуль/угол
        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)  # [0, 2π)

        # Нормировка магнитуды для порогов неплоскости
        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        # 2) Выбор K неплоских центров (с адаптивным добором при нехватке)
        centers = self._pick_keyholes(mnorm)

        # 3) Активация детекторов по углу
        active: Set[int] = set()
        angles_accum: List[float] = []
        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            angle = self._dominant_angle(tile_a, tile_m)
            angles_accum.append(float(angle))

            for bit in self._angle_bits(angle):
                active.add(bit)
                # для отладки/аналитики:
                self.bit2info.setdefault(bit, {"angle": float(angle)})

        # 4) Диагностика (по желанию)
        if self.print_code or self.print_angle:
            sobel_angle = self._circular_mean(angles_accum) if angles_accum else None
            self._print_barcode_and_angle(active, sobel_angle)

        return active

    def print_density_and_overlap(self, codes: List[Set[int]]) -> Tuple[float, float]:
        """
        Печать и возврат метрик качества кодов по батчу.

        avg_density  = средняя плотность = mean_i ( |code_i| / B )
        avg_overlap  = средний Яккард по всем парам:
                       J(A,B) = |A ∩ B| / |A ∪ B|
                       (для двух пустых кодов принимаем J=1.0)

        Параметры
        ----------
        codes : список множеств битов для разных изображений.

        Возврат
        -------
        (avg_density, avg_overlap) : Tuple[float, float]
        """
        n = len(codes)
        if n == 0:
            print("avg_density=0.0, avg_overlap=0.0 (нет кодов)")
            return 0.0, 0.0

        densities = [len(c) / float(self.B) for c in codes]
        avg_density = float(sum(densities) / n)

        pair_sum = 0.0
        pair_cnt = 0
        for i in range(n):
            A = codes[i]
            for j in range(i + 1, n):
                B = codes[j]
                if not A and not B:
                    jacc = 1.0
                else:
                    union = len(A | B)
                    jacc = (len(A & B) / union) if union > 0 else 0.0
                pair_sum += jacc
                pair_cnt += 1

        avg_overlap = (pair_sum / pair_cnt) if pair_cnt > 0 else 0.0

        print(f"avg_density = {avg_density:.6f}")
        print(f"avg_overlap (Jaccard) = {avg_overlap:.6f}")
        return avg_density, avg_overlap

    # ==================== ВНУТРЕННИЕ МЕТОДЫ ====================

    def _pick_keyholes(self, mnorm: np.ndarray) -> List[Tuple[int, int]]:
        """
        Выбор центров скважин среди всех допустимых координат
        с отсевом «плоских» и адаптивным добором до K.

        Кандидат (окно) описывается:
          mean = средняя нормированная магнитуда в окне
          frac = доля пикселей с m > mag_eps
          score = 0.5*mean + 0.5*frac (сортировка по убыванию)

        Порог неплоскости:
          frac >= min_active_frac и mean >= mag_eps

        Если выбранных < K и adaptive_fill=True — пороги умножаются на adaptive_decay
        и отбор повторяется, пока не наберём K или пороги не обнулятся.
        """
        H, W = mnorm.shape
        pad = self.S // 2

        candidates: List[Tuple[float, float, float, int, int]] = []  # (score, frac, mean, y, x)
        for y in range(pad, H - pad):
            for x in range(pad, W - pad):
                y0, y1, x0, x1 = self._window_bounds(y, x, H, W)
                tile = mnorm[y0:y1, x0:x1]
                mean = float(tile.mean())
                frac = float((tile > self.mag_eps).mean())
                score = 0.5 * mean + 0.5 * frac
                candidates.append((score, frac, mean, y, x))

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

            # ослабляем пороги и повторяем
            thr_frac *= self.adaptive_decay
            thr_mean *= self.adaptive_decay
            if thr_frac <= 0.0 and thr_mean <= 0.0:
                return picked  # на полностью гладком изображении может быть < K

    @staticmethod
    def _window_bounds(cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        """Координаты окна S×S вокруг (cy, cx) внутри границ изображения."""
        # r = S//2 вычислим через отношение к self; здесь статик — ради скорости в цикле
        # но границу по S контролируем снаружи
        raise NotImplementedError  # см. реализацию ниже (не статик)

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:  # type: ignore[override]
        r = self.S // 2
        y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

    @staticmethod
    def _sobel(img01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Собель 3×3 по X/Y с паддингом по краю."""
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
        """Свертка ядром k с сохранением размера (edge padding)."""
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
        H, W = img.shape
        out = np.zeros_like(img, dtype=np.float32)
        kh, kw = k.shape
        for y in range(H):
            for x in range(W):
                patch = pad[y:y + kh, x:x + kw]
                out[y, x] = float(np.sum(patch * k))
        return out

    @staticmethod
    def _dominant_angle(tile_a: np.ndarray, tile_m: np.ndarray) -> float:
        """
        Доминирующий угол в окне: аргумент векторной суммы единичных векторов
        направлений пикселей, взвешенных магнитудой.
        Возвращает угол в [0, 2π). При «плоском» окне вернёт 0.0.
        """
        weights = tile_m.astype(np.float64)
        total = float(weights.sum())
        if total <= 1e-8:
            return 0.0
        vectors = np.exp(1j * tile_a.astype(np.float64)) * weights
        vec_sum = vectors.sum()
        if vec_sum == 0:
            return 0.0
        return float(np.angle(vec_sum) % (2 * np.pi))

    # --------- Широкие детекторы по углу ---------

    def _build_angle_detectors(self) -> None:
        """Создаёт список детекторов: (центр дуги mu, полуширина width, закреплённые биты)."""
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
        """Детерминированное отображение «детектор → k битов» (устойчиво между запусками)."""
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest(), 'little')
        step = 0x9E3779B97F4A7C15  # псевдо-золотое смещение
        return [int((h + i * step) % self.B) for i in range(k)]

    def _angle_bits(self, angle: float) -> List[int]:
        """
        Вернуть биты детекторов, активных для данного угла.

        Логика:
          1) Бинарная активация детекторов, где |angle - mu|_circle <= width.
             Приоритезируем более «центральные» детекторы (больше width - d).
          2) Если ничего не попало — выбираем ближайшие по «гауссовой» близости.
          3) Обрезаем до max_detectors_per_center.
        """
        candidates: List[Tuple[float, List[int]]] = []
        for (mu, width, bits) in self._angle_detectors:
            d = self._circ_delta(angle, mu)
            if d <= width:
                candidates.append((float(width - d), bits))

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

    # --------- Вспомогательные ---------

    @staticmethod
    def _circular_mean(angles: List[float]) -> Optional[float]:
        """Круговое среднее углов в [0, 2π). Возвращает None, если список пуст."""
        if not angles:
            return None
        v = np.exp(1j * np.array(angles, dtype=np.float64))
        vec_sum = v.sum()
        return float(np.angle(vec_sum) % (2 * np.pi))

    def _print_barcode_and_angle(self, active: Set[int], mean_angle: Optional[float]) -> None:
        """Печатает строку-«штрихкод» и средний угол (если запрошено флагами)."""
        if self.print_code:
            bar = [self.barcode_off] * self.B
            for b in active:
                bi = int(b)
                if 0 <= bi < self.B:
                    bar[bi] = self.barcode_on
            print(''.join(bar))
        if self.print_angle:
            if mean_angle is None:
                print("sobel_angle ≈ n/a")
            else:
                print(f"sobel_angle ≈ {mean_angle:.6f} rad ({mean_angle * 180 / np.pi:.2f}°)")

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