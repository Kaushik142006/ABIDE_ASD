# src/functional_connectivity.py

import numpy as np

def compute_functional_connectivity_matrix(ts_scaled, n_rois=200):
    r = (ts_scaled.T @ ts_scaled) / ts_scaled.shape[0]
    np.fill_diagonal(r, 1.0)
    fc = np.arctanh(np.clip(r, -0.9999, 0.9999)).astype(np.float32)
    return fc[np.triu_indices(n_rois, 1)]