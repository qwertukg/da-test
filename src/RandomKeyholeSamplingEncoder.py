import random
import math
import hashlib
from typing import Optional, Dict, List, Set, Tuple

import numpy as np


class RandomKeyholeSamplingEncoder:
    """
    Энкодер «случайных скважин» с широкими угловыми детекторами.
    1 скважина -> 1 код (множество битов). Коды зависят ТОЛЬКО от угла в скважине.
    """

    def __init__(self,
                 img_hw: Tuple[int, int] = (28, 28),
                 bits: int = 128,
                 keyholes_per_img: int = 20,
                 keyhole_size: int = 5,
                 seed: int = 42,
                 # --- угловые слои: (число детекторов, полуширина дуги) ---
                 angle_layers: Optional[List[Tuple[int, float]]] = None,
                 # --- сколько ближайших детекторов брать на каждом слое ---
                 detectors_per_layer: Optional[List[int]] = None,
                 # --- сколько бит выдает один детектор ---
                 bits_per_detector: int = 4,
                 # --- отбор «неплоских» скважин ---
                 mag_eps: float = 0.03,
                 min_active_frac: float = 0.05,
                 adaptive_fill: bool = True,
                 adaptive_decay: float = 0.5,
                 # --- символы для штрихкода (используются в print_keyhole_records) ---
                 barcode_on: str = "|",
                 barcode_off: str = " ",
                 ):
        self.H, self.W = img_hw
        self.B = int(bits)
        self.K = int(keyholes_per_img)
        self.S = int(keyhole_size)
        assert self.S % 2 == 1, "keyhole_size должно быть нечётным (например, 5)"
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        # слои по углу (узкий + широкий по умолчанию)
        if angle_layers is None:
            angle_layers = [
                (128, float(np.pi / 64)),  # узкий (детализация)
                (16,  float(np.pi / 3)),   # широкий (стабильность/перекрытие)
            ]
        self.angle_layers: List[Tuple[int, float]] = [(int(n), float(w)) for (n, w) in angle_layers]

        if detectors_per_layer is None:
            detectors_per_layer = [1] * len(self.angle_layers)
        assert len(detectors_per_layer) == len(self.angle_layers)
        self.detectors_per_layer: List[int] = [max(1, int(k)) for k in detectors_per_layer]

        self.bits_per_detector = int(bits_per_detector)

        # отбор «неплоских»
        self.mag_eps = float(mag_eps)
        self.min_active_frac = float(min_active_frac)
        self.adaptive_fill = bool(adaptive_fill)
        self.adaptive_decay = float(adaptive_decay)
        assert 0.0 < self.adaptive_decay <= 1.0, "adaptive_decay должен быть в (0, 1]"

        # отображение штрихкода
        self.barcode_on = str(barcode_on)
        self.barcode_off = str(barcode_off)

        # слои детекторов (центры и закреплённые за ними биты)
        self._layers: List[Dict[str, object]] = []
        self._build_angle_layers()

        # отладочная инфа по битам (угол)
        self.bit2info: Dict[int, Dict[str, float]] = {}

        # --- НОВОЕ: сюда пишем по каждой скважине (angle, code) ---
        # После каждого encode() список перезаписывается заново.
        self.keyhole_records: List[Tuple[float, Set[int]]] = []

    # ==================== ПУБЛИЧНОЕ API ====================

    def encode(self, img: np.ndarray) -> List[Set[int]]:
        """
        Кодирует изображение в СПИСОК кодов скважин (1 скважина -> 1 код).
        Также заполняет self.keyhole_records списком (angle, code).
        """
        H, W = img.shape
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Ожидался размер {(self.H, self.W)}, а пришёл {(H, W)}")

        # сбрасываем записи прошлых вызовов
        self.keyhole_records = []

        # 1) Собель
        gx, gy = self._sobel(img)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi)  # [0, 2π)

        # 2) Нормировка магнитуды для отбора неплоских
        mmax = float(np.max(mag))
        mnorm = mag / (mmax + 1e-8) if mmax > 1e-8 else mag

        # 3) Выбор центров
        centers = self._pick_keyholes(mnorm)

        # 4) По скважине — код (только от угла)
        codes_per_keyhole: List[Set[int]] = []
        for (cy, cx) in centers:
            y0, y1, x0, x1 = self._window_bounds(cy, cx, H, W)
            tile_m = mnorm[y0:y1, x0:x1]
            tile_a = ang[y0:y1, x0:x1]

            angle = self._dominant_angle(tile_a, tile_m)  # [0, 2π)
            bits_set: Set[int] = set(self._angle_bits_per_layer(angle))

            for bit in bits_set:
                self.bit2info.setdefault(int(bit), {"angle": float(angle)})

            # --- НОВОЕ: накапливаем в поле класса ---
            self.keyhole_records.append((float(angle), bits_set))

            codes_per_keyhole.append(bits_set)

        return codes_per_keyhole

    def print_keyhole_records(self, as_barcode: bool = True) -> None:
        """
        Печатает список (angle, code) по возрастанию угла.
        as_barcode=True: выводит штрихкоды длиной B; иначе — индексы активных битов.
        """
        if not self.keyhole_records:
            print("Записей нет. Сначала вызовите encode().")
            return

        # сортировка по возрастанию угла
        recs = sorted(self.keyhole_records, key=lambda t: t[0])

        for ang, code in recs:
            deg = ang * 180.0 / np.pi
            if as_barcode:
                s = self._bits_to_barcode(code)
            else:
                inds = sorted(int(b) for b in code)
                s = f"indices={inds}"
            print(f"{ang:.6f} rad ({deg:7.2f}°): {s}")

    def average_density(self, codes_per_keyhole: List[Set[int]]) -> float:
        """
        Средняя плотность по скважинам: mean_k(|code_k| / B).
        """
        if not codes_per_keyhole:
            print("avg_density (per keyhole) = 0.000000")
            return 0.0
        densities = [len(c) / float(self.B) for c in codes_per_keyhole]
        avg_density = float(sum(densities) / len(densities))
        print(f"avg_density (per keyhole) = {avg_density:.6f}")
        return avg_density

    def print_density_and_overlap(self, codes_per_keyhole: List[Set[int]]) -> Tuple[float, float]:
        """
        Метрики ПО СКВАЖИНАМ (для одного изображения).
        avg_density = mean_k( |code_k| / B )
        avg_overlap = средний Яккард по всем парам скважин (J=1.0 для двух пустых).
        """
        n = len(codes_per_keyhole)
        if n == 0:
            print("avg_density=0.0, avg_overlap=0.0 (скважин нет)")
            return 0.0, 0.0

        densities = [len(c) / float(self.B) for c in codes_per_keyhole]
        avg_density = float(sum(densities) / n)

        pair_sum = 0.0
        pair_cnt = 0
        for i in range(n):
            A = codes_per_keyhole[i]
            for j in range(i + 1, n):
                B = codes_per_keyhole[j]
                if not A and not B:
                    jacc = 1.0
                else:
                    union = len(A | B)
                    jacc = (len(A & B) / union) if union > 0 else 0.0
                pair_sum += jacc
                pair_cnt += 1

        avg_overlap = (pair_sum / pair_cnt) if pair_cnt > 0 else 0.0
        print(f"avg_density (per keyhole) = {avg_density:.6f}")
        print(f"avg_overlap (Jaccard, between keyholes) = {avg_overlap:.6f}")
        return avg_density, avg_overlap

    # ==================== ВНУТРЕННЕЕ ====================

    def _pick_keyholes(self, mnorm: np.ndarray) -> List[Tuple[int, int]]:
        """Отбираем до K центров по неплоскости; при нехватке — ослабляем пороги."""
        H, W = mnorm.shape
        pad = self.S // 2

        candidates: List[Tuple[float, float, float, int, int]] = []
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

            thr_frac *= self.adaptive_decay
            thr_mean *= self.adaptive_decay
            if thr_frac <= 0.0 and thr_mean <= 0.0:
                return picked

    def _window_bounds(self, cy: int, cx: int, H: int, W: int) -> Tuple[int, int, int, int]:
        r = self.S // 2
        y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
        return y0, y1, x0, x1

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
        kh, kw = k.shape
        for y in range(H):
            for x in range(W):
                patch = pad[y:y + kh, x:x + kw]
                out[y, x] = float(np.sum(patch * k))
        return out

    @staticmethod
    def _dominant_angle(tile_a: np.ndarray, tile_m: np.ndarray) -> float:
        """Доминирующий угол на окне (в радианах, [0, 2π))."""
        weights = tile_m.astype(np.float64)
        total = float(weights.sum())
        if total <= 1e-8:
            return 0.0
        vec_sum = (np.exp(1j * tile_a.astype(np.float64)) * weights).sum()
        if vec_sum == 0:
            return 0.0
        return float(np.angle(vec_sum) % (2 * np.pi))

    # ---- многослойные детекторы по углу ----
    def _build_angle_layers(self) -> None:
        """Создаёт слои детекторов: для каждого центра фиксируем свой список битов."""
        self._layers.clear()
        for L, (n, width) in enumerate(self.angle_layers):
            n = int(n)
            width = float(width)
            mus = [2.0 * np.pi * j / n for j in range(n)]
            bits_tbl = [self._det_bits(f"L{L}:DET{j}:{self.seed}", self.bits_per_detector)
                        for j in range(n)]
            self._layers.append({"n": n, "width": width, "mu": mus, "bits": bits_tbl})

    @staticmethod
    def _circ_delta(a: float, b: float) -> float:
        """Минимальная круговая разность углов (0..π]."""
        return abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    def _det_bits(self, key: str, k: int) -> List[int]:
        """Детерминированное отображение «детектор -> k битов»."""
        h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=16).digest(), 'little')
        step = 0x9E3779B97F4A7C15
        return [int((h + i * step) % self.B) for i in range(k)]

    def _angle_bits_per_layer(self, angle: float) -> List[int]:
        """
        Для данного угла собираем биты с каждого слоя:
        берём k_L детекторов с лучшим приоритетом (внутри дуги — приоритет по близости к центру,
        если никого нет — k_L ближайших по круговой дистанции).
        """
        out: List[int] = []
        for L, layer in enumerate(self._layers):
            n = layer["n"]            # type: ignore[assignment]
            width = layer["width"]    # type: ignore[assignment]
            mus = layer["mu"]         # type: ignore[assignment]
            bits_tbl = layer["bits"]  # type: ignore[assignment]
            kL = min(self.detectors_per_layer[L], n)

            cand: List[Tuple[float, int]] = []
            for j in range(n):  # type: ignore[arg-type]
                d = self._circ_delta(angle, mus[j])  # type: ignore[index]
                if d <= width:
                    cand.append((width - d, j))

            if not cand:
                dist = [(self._circ_delta(angle, mus[j]), j) for j in range(n)]  # type: ignore[index]
                dist.sort(key=lambda t: t[0])
                chosen = [j for _, j in dist[:kL]]
            else:
                cand.sort(key=lambda t: t[0], reverse=True)
                chosen = [j for _, j in cand[:kL]]

            for j in chosen:
                out.extend(bits_tbl[j])  # type: ignore[index]

        return out

    # ---- служебное: формирование штрихкода из множества битов ----
    def _bits_to_barcode(self, code: Set[int]) -> str:
        bar = [self.barcode_off] * self.B
        for b in code:
            bi = int(b)
            if 0 <= bi < self.B:
                bar[bi] = self.barcode_on
        return ''.join(bar)

    # (опционально) восстановление кругового среднего угла из объединённого кода
    def code_dominant_orientation(self, code: Set[int]) -> Tuple[float, float]:
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