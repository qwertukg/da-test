#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вспомогательные функции для проекта DAMP-light."""

import math
from typing import Set, Tuple, List

import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST


def cosbin(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))


def to01(img8x8: np.ndarray) -> np.ndarray:
    """Нормирует массив 0..16 в диапазон 0..1."""
    return (img8x8.astype(np.float32) / 16.0).reshape(8, 8)


def load_mnist_28x28(train_limit=8000, test_limit=2000, seed=0):
    """Грузит и подсэмплирует MNIST (28×28, 0..1)."""
    tfm = transforms.ToTensor()
    train_ds = MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = MNIST(root="./data", train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    train_idx = np.arange(len(train_ds))
    test_idx = np.arange(len(test_ds))

    if train_limit and len(train_idx) > train_limit:
        train_idx = rng.choice(train_idx, size=train_limit, replace=False)
    if test_limit and len(test_idx) > test_limit:
        test_idx = rng.choice(test_idx, size=test_limit, replace=False)

    X_train = [train_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in train_idx]
    y_train = [int(train_ds[i][1]) for i in train_idx]
    X_test = [test_ds[i][0].squeeze(0).numpy().astype(np.float32) for i in test_idx]
    y_test = [int(test_ds[i][1]) for i in test_idx]
    return X_train, X_test, y_train, y_test

