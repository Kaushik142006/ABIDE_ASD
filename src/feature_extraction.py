# src/feature_extraction.py

import numpy as np
from sklearn.preprocessing import StandardScaler

def get_scaled_features(X_all, selected_edges):
    X_selected = X_all[:, selected_edges]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    return X_scaled, scaler