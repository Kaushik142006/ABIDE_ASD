# src/roi_extraction.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_rois_and_standardize(ts_list, n_rois=200):
    processed = []
    for ts in ts_list:
        ts_scaled = StandardScaler().fit_transform(ts)
        processed.append(ts_scaled)
    return processed