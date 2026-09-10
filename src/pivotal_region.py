# src/pivotal_region.py

import numpy as np
from scipy.stats import ttest_ind

def extract_global_pivotal_edges(X_all, labels_all, k_edges=600):
    t_stats, _ = ttest_ind(X_all[labels_all == 1], X_all[labels_all == 0], equal_var=False)
    selected_edges = np.argsort(np.abs(np.nan_to_num(t_stats)))[-k_edges:]
    return selected_edges