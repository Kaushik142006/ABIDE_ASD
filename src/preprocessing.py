# src/preprocessing.py

import numpy as np

def clean_time_series(ts_list):
    cleaned = []
    for ts in ts_list:
        upper = np.percentile(ts, 95, axis=0)
        lower = np.percentile(ts, 5, axis=0)
        ts_clean = np.clip(ts, lower, upper)
        cleaned.append(ts_clean)
    return cleaned