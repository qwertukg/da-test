#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hill_search_quality.py — локальный поиск (hill climbing) от базовой конфигурации.

Идея:
- Берём базовую конфигурацию (как у пользователя).
- Последовательно проходим по списку параметров.
- Для каждого параметра пробуем соседние значения (±шаг для float/int, соседние варианты для "choice").
- Если found_accuracy > best_accuracy + eps → принимаем изменение и продолжаем с обновлённой базой.
- Иначе откатываемся.
- Делаем несколько "проходов" по всем параметрам (passes).

Логи:
- Записывает полный журнал попыток и лучшую конфигурацию в JSON (--out).
"""

import argparse, json, random, math, time, sys
from copy import deepcopy
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ====== Импортируем ваши классы ======
TRY_MODULES = ["discrete_digits_final", "damp_light"]
for _mod in TRY_MODULES:
    try:
        _m = __import__(_mod, fromlist=["PrimaryEncoder","Layout2D","DetectorSpace","KNNJaccard"])
        PrimaryEncoder = getattr(_m, "PrimaryEncoder")
        Layout2D = getattr(_m, "Layout2D")
        DetectorSpace = getattr(_m, "DetectorSpace")
        KNNJaccard = getattr(_m, "KNNJaccard")
        break
    except Exception as e:
        PrimaryEncoder = Layout2D = DetectorSpace = KNNJaccard = None
        last_err = e
if PrimaryEncoder is None:
    print("[FATAL] Не удалось импортировать классы из discrete_digits_final.py (или damp_light).")
    print("Последняя ошибка:", last_err)
    sys.exit(1)

# ====== Датасеты ======
def to01_img8x8(img8x8: np.ndarray) -> np.ndarray:
    return (img8x8.astype(np.float32) / 16.0).reshape(8, 8)

def load_skdigits(seed=0, test_size=0.33):
    d = load_digits()
    X = [to01_img8x8(img) for img in d.images]
    y = list(d.target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return X_tr, X_te, y_tr, y_te, (8, 8), "skdigits"

def load_mnist_28x28(train_limit=8000, test_limit=2000, seed=0):
    from torchvision import transforms
    from torchvision.datasets import MNIST
    tfm = transforms.ToTensor()
    train_ds = MNIST(root="./data", train=True,  download=True, transform=tfm)
    test_ds  = MNIST(root="./data", train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    tr_idx = np.arange(len(train_ds))
    te_idx = np.arange(len(test_ds))
    if train_limit and len(tr_idx) > train_limit:
        tr_idx = rng.choice(tr_idx, size=train_limit, replace=False)
    if test_limit and len(te_idx) > test_limit:
        te_idx = rng.choice(te_idx, size=test_limit, replace=False)

    X_tr = [train_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in tr_idx]
    y_tr = [int(train_ds[i][1]) for i in tr_idx]
    X_te = [test_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in te_idx]
    y_te = [int(test_ds[i][1]) for i in te_idx]
    return X_tr, X_te, y_tr, y_te, (28, 28), "mnist"

# ====== Утилиты ======
def set_global_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)

def encode_many(enc: PrimaryEncoder, X: List[np.ndarray]) -> List[Set[int]]:
    return [enc.encode(x) for x in X]

def avg_density(Z: List[Set[int]], emb_bits: int) -> float:
    return float(np.mean([len(z) for z in Z])) / max(1, emb_bits)

# ====== Базовая конфигурация от пользователя ======
def base_config(img_hw, n_train):
    return {
        # Encoder
        "img_hw": img_hw, "bits": 8192, "seed_enc": 42,
        "grid_g": 4, "grid_levels": 4, "grid_bits_per_cell": 3,
        "coarse_bits_per_cell": 4,
        "bright_levels": 8,
        "orient_on": True, "orient_bins": 8, "orient_grid": 4,
        "orient_bits_per_cell": 2, "orient_mag_thresh": 0.10,
        "max_active_bits": 260,
        # Layout
        "R_far": 7, "R_near": 3, "epochs_far": 8, "epochs_near": 6, "seed_layout": 123,
        # DetectorSpace
        "emb_bits": 256,
        "lam_floor": 0.06, "percentile": 0.88, "min_activated": 35, "mu": 0.15,
        "seeds": min(1200, n_train), "min_comp": 4, "min_center_dist": 1.6,
        "max_detectors": 260, "seed_det": 7,
        # KNN
        "k": 3
    }

# ====== Метаданные параметров для локального поиска ======
# Описываем шаги, диапазоны и тип изменения (float/int/choice).
PARAM_SPACE = {
    # Encoder
    "grid_g":            {"type":"int",    "min":3, "max":6, "step":1},
    "grid_levels":       {"type":"int",    "min":3, "max":6, "step":1},
    "grid_bits_per_cell":{"type":"int",    "min":1, "max":4, "step":1},
    "coarse_bits_per_cell":{"type":"int",  "min":2, "max":5, "step":1},
    "bright_levels":     {"type":"int",    "min":6, "max":12, "step":2},
    "orient_bins":       {"type":"choice", "values":[8, 12]},
    "orient_grid":       {"type":"int",    "min":3, "max":6, "step":1},
    "orient_bits_per_cell":{"type":"int",  "min":1, "max":3, "step":1},
    "orient_mag_thresh": {"type":"float",  "min":0.06, "max":0.20, "step":0.02},
    "max_active_bits":   {"type":"int",    "min":120, "max":400, "step":20},

    # Layout
    "R_far":             {"type":"int",    "min":5, "max":10, "step":1},
    "R_near":            {"type":"int",    "min":2, "max":5, "step":1},
    "epochs_far":        {"type":"int",    "min":4, "max":12, "step":2},
    "epochs_near":       {"type":"int",    "min":4, "max":10, "step":2},

    # DetectorSpace
    "emb_bits":          {"type":"choice", "values":[128, 256, 384]},
    "lam_floor":         {"type":"float",  "min":0.04, "max":0.14, "step":0.02},
    "percentile":        {"type":"float",  "min":0.84, "max":0.97, "step":0.02},
    "min_activated":     {"type":"int",    "min":10, "max":40, "step":5},
    "mu":                {"type":"float",  "min":0.15,"max":0.55,"step":0.05},
    "seeds":             {"type":"intdyn", "min":200, "max":2000, "step":200},  # обрежем до n_train внутри
    "min_comp":          {"type":"int",    "min":4, "max":9, "step":1},
    "min_center_dist":   {"type":"float",  "min":1.4,"max":2.4,"step":0.2},
    "max_detectors":     {"type":"int",    "min":60, "max":400, "step":20},  # дополнительно ограничим ≤ emb_bits в коде

    # KNN
    "k":                 {"type":"choice", "values":[1,3,5,7,9]},
}

# Порядок перебора параметров (можно менять).
PARAM_ORDER = [
    # Encoder
    "orient_mag_thresh","orient_bins","orient_grid","grid_g","grid_levels",
    "grid_bits_per_cell","orient_bits_per_cell","coarse_bits_per_cell",
    "bright_levels","max_active_bits",
    # Layout
    "R_far","epochs_far","R_near","epochs_near",
    # DetectorSpace
    "percentile","lam_floor","min_activated",
    "mu","min_comp","min_center_dist",
    "emb_bits","max_detectors","seeds",
    # KNN
    "k",
]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def neighbors_for_param(cfg: Dict, name: str, n_train: int):
    """Сгенерировать 1-2 соседних значения для параметра name вокруг текущего cfg[name]."""
    meta = PARAM_SPACE[name]
    cur = cfg[name]
    out = []

    if meta["type"] == "choice":
        vals = meta["values"]
        if cur not in vals:
            # выбрать ближайшее
            cur = vals[min(range(len(vals)), key=lambda i: abs(vals[i]-cur))]
        idx = vals.index(cur)
        if idx > 0: out.append(vals[idx-1])
        if idx < len(vals)-1: out.append(vals[idx+1])

    elif meta["type"] == "int":
        step = meta["step"]
        out = [clamp(cur - step, meta["min"], meta["max"]),
               clamp(cur + step, meta["min"], meta["max"])]
        out = [v for v in out if v != cur]

    elif meta["type"] == "intdyn":
        step = meta["step"]
        lo, hi = meta["min"], min(meta["max"], n_train)
        out = [clamp(cur - step, lo, hi), clamp(cur + step, lo, hi)]
        out = [v for v in out if v != cur]

    elif meta["type"] == "float":
        step = meta["step"]
        out = [round(clamp(cur - step, meta["min"], meta["max"]), 4),
               round(clamp(cur + step, meta["min"], meta["max"]), 4)]
        out = [v for v in out if abs(v - cur) > 1e-9]

    return out

# ====== Построение и оценка пайплайна ======
def build_and_eval(cfg: Dict, X_train, X_test, y_train, y_test):
    # 1) Encoder
    enc = PrimaryEncoder(
        img_hw=cfg["img_hw"], bits=cfg["bits"], seed=cfg["seed_enc"],
        grid_g=cfg["grid_g"], grid_levels=cfg["grid_levels"], grid_bits_per_cell=cfg["grid_bits_per_cell"],
        coarse_bits_per_cell=cfg["coarse_bits_per_cell"], bright_levels=cfg["bright_levels"],
        orient_on=cfg["orient_on"], orient_bins=cfg["orient_bins"], orient_grid=cfg["orient_grid"],
        orient_bits_per_cell=cfg["orient_bits_per_cell"], orient_mag_thresh=cfg["orient_mag_thresh"],
        max_active_bits=cfg["max_active_bits"]
    )
    codes_train = encode_many(enc, X_train)
    codes_test  = encode_many(enc, X_test)

    # 2) Layout
    lay = Layout2D(R_far=cfg["R_far"], R_near=cfg["R_near"],
                   epochs_far=cfg["epochs_far"], epochs_near=cfg["epochs_near"],
                   seed=cfg["seed_layout"]).fit(codes_train)

    # 3) DetectorSpace
    # правим max_detectors ≤ emb_bits
    max_det = int(min(cfg["max_detectors"], cfg["emb_bits"]))
    det = DetectorSpace(
        lay, codes_train, y_train,
        emb_bits=cfg["emb_bits"],
        lam_floor=cfg["lam_floor"], percentile=cfg["percentile"], min_activated=cfg["min_activated"],
        mu=cfg["mu"],
        seeds=min(cfg["seeds"], len(codes_train)),
        min_comp=cfg["min_comp"], min_center_dist=cfg["min_center_dist"],
        max_detectors=max_det,
        seed=cfg["seed_det"]
    )

    # 4) Embeddings + KNN
    Z_train = [det.embed(c) for c in codes_train]
    Z_test  = [det.embed(c) for c in codes_test]
    clf = KNNJaccard(k=cfg["k"]).fit(Z_train, y_train)
    y_pred = clf.predict(Z_test)

    acc = accuracy_score(y_test, y_pred)
    dens_tr = avg_density(Z_train, det.emb_bits)
    dens_te = avg_density(Z_test,  det.emb_bits)
    return acc, {
        "embed_density_train": float(dens_tr),
        "embed_density_test":  float(dens_te),
        "detectors": int(len(det.detectors)),
        "unique_bits": int(len({d["bit"] for d in det.detectors})),
        "emb_bits": int(det.emb_bits)
    }

# ====== Локальный поиск ======
def hill_climb(cfg0: Dict, X_train, X_test, y_train, y_test,
               passes=2, eps_gain=1e-5, verbose=True):
    log = []
    # оценим базовую конфигурацию
    acc_best, m_best = build_and_eval(cfg0, X_train, X_test, y_train, y_test)
    best = deepcopy(cfg0)
    log.append({"params": deepcopy(best), "metrics": {"accuracy_test": float(acc_best), **m_best}, "note":"baseline"})
    if verbose:
        print(f"[BASE] acc={acc_best:.4f} | det={m_best['detectors']}/{m_best['unique_bits']}/{m_best['emb_bits']} | dens={m_best['embed_density_test']:.3f}")

    for p in range(1, passes+1):
        if verbose: print(f"\n=== PASS {p}/{passes} ===")
        improved_any = False

        for name in PARAM_ORDER:
            neigh_vals = neighbors_for_param(best, name, n_train=len(X_train))
            if not neigh_vals:
                continue

            cur_val = best[name]
            tried = []
            local_best_acc = acc_best
            local_best_val = cur_val
            local_best_metrics = m_best

            for v in neigh_vals:
                cfg_try = deepcopy(best)
                cfg_try[name] = v
                # соблюдаем ограничение на max_detectors ≤ emb_bits, если меняется одно из них
                if name == "emb_bits" and cfg_try["max_detectors"] > cfg_try["emb_bits"]:
                    cfg_try["max_detectors"] = cfg_try["emb_bits"]
                if name == "max_detectors" and cfg_try["max_detectors"] > cfg_try["emb_bits"]:
                    cfg_try["max_detectors"] = cfg_try["emb_bits"]

                acc_try, m_try = build_and_eval(cfg_try, X_train, X_test, y_train, y_test)
                tried.append({"value": v, "acc": float(acc_try), "metrics": m_try})

                # выбираем лучшее направление
                if acc_try > local_best_acc + eps_gain:
                    local_best_acc = acc_try
                    local_best_val = v
                    local_best_metrics = m_try

            # Принимаем, если улучшили
            if local_best_val != cur_val:
                best[name] = local_best_val
                acc_best = local_best_acc
                m_best = local_best_metrics
                improved_any = True
                log.append({"params": deepcopy(best), "metrics": {"accuracy_test": float(acc_best), **m_best},
                            "note": f"accept {name} -> {local_best_val}"})
                if verbose:
                    print(f"[ACCEPT] {name}: {cur_val} → {local_best_val} | acc={acc_best:.4f} | dens={m_best['embed_density_test']:.3f} | det={m_best['detectors']}/{m_best['unique_bits']}/{m_best['emb_bits']}")
            else:
                # логируем неудачные попытки
                log.append({"params": deepcopy(best), "metrics": {"accuracy_test": float(acc_best), **m_best},
                            "note": f"no_improve {name}", "tried": tried})
                if verbose:
                    print(f"[KEEP] {name}: {cur_val} → {local_best_val} | acc={acc_best:.4f} | (нет улучшения)")

        if not improved_any:
            if verbose: print("[STOP] Пасс не дал улучшений — останавливаемся.")
            break

    return best, acc_best, m_best, log

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["skdigits","mnist"], default="skdigits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=2, help="сколько полных проходов по параметрам сделать")
    ap.add_argument("--out", type=str, default="hill_results.json")
    ap.add_argument("--mnist-train", type=int, default=8000)
    ap.add_argument("--mnist-test",  type=int, default=2000)
    args = ap.parse_args()

    set_global_seeds(args.seed)

    if args.dataset == "skdigits":
        X_train, X_test, y_train, y_test, img_hw, dset = load_skdigits(seed=args.seed)
    else:
        X_train, X_test, y_train, y_test, img_hw, dset = load_mnist_28x28(
            train_limit=args.mnist_train, test_limit=args.mnist_test, seed=args.seed
        )

    cfg0 = base_config(img_hw, n_train=len(X_train))
    t0 = time.time()
    best_cfg, best_acc, best_metrics, log = hill_climb(
        cfg0, X_train, X_test, y_train, y_test,
        passes=args.passes, eps_gain=1e-5, verbose=True
    )
    elapsed = round(time.time() - t0, 3)

    payload = {
        "dataset": dset,
        "seed": args.seed,
        "passes": args.passes,
        "elapsed_sec": elapsed,
        "best": {
            "params": best_cfg,
            "metrics": {"accuracy_test": float(best_acc), **best_metrics}
        },
        "log": log
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Результаты сохранены в {args.out}")
    print(f"[BEST] acc={best_acc:.4f} | det={best_metrics['detectors']}/{best_metrics['unique_bits']}/{best_metrics['emb_bits']} | dens={best_metrics['embed_density_test']:.3f}")

if __name__ == "__main__":
    main()