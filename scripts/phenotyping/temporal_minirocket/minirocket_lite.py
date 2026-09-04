from __future__ import annotations

from typing import Tuple

import numpy as np


class MiniRocketLite:
    def __init__(self, n_kernels: int = 256, kernel_lengths: Tuple[int, ...] = (7, 9, 11), random_state: int = 42):
        self.n_kernels = n_kernels
        self.kernel_lengths = kernel_lengths
        self.random_state = random_state
        self.kernels: list[tuple[np.ndarray, np.ndarray]] = []
        self.biases: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n_channels = X.shape[1]
        self.kernels = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(self.kernel_lengths))
            # Use single channel per kernel to avoid over-averaging signals
            n_ch = 1
            channels = rng.choice(n_channels, size=n_ch, replace=False)
            weights = rng.choice([-1.0, 1.0], size=length)
            self.kernels.append((channels, weights))

        sample_size = min(32, X.shape[0])
        sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        biases = []
        for channels, weights in self.kernels:
            vals = []
            for idx in sample_idx:
                series = X[idx, channels[0], :]
                conv = np.convolve(series, weights, mode="valid")
                vals.append(np.mean(conv) if conv.size else 0.0)
            biases.append(float(np.median(vals)) if vals else 0.0)
        self.biases = np.array(biases)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.biases is None:
            raise RuntimeError("MiniRocketLite not fitted")
        n_samples = X.shape[0]
        feats = np.zeros((n_samples, len(self.kernels)), dtype=float)
        for k, ((channels, weights), bias) in enumerate(zip(self.kernels, self.biases)):
            for i in range(n_samples):
                series = X[i, channels[0], :]
                conv = np.convolve(series, weights, mode="valid")
                if conv.size == 0:
                    feats[i, k] = 0.0
                else:
                    feats[i, k] = float(np.mean(conv > bias))
        return feats
