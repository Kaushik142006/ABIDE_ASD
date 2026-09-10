# src/dataset_loader.py

import re
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from PIL import Image

ALLOWED_EXTENSIONS = ('.nii', '.jpg', '.png')

def load_abide_metadata(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")
    meta = pd.read_csv(csv_path)
    meta.columns = meta.columns.str.strip()
    sub_col = "SUB_ID" if "SUB_ID" in meta.columns else "participant"
    dx_col = "DX_GROUP" if "DX_GROUP" in meta.columns else "dx_group"
    
    meta["SUB_ID"] = meta[sub_col].astype(int)
    meta["label"] = (meta[dx_col] == 1).astype(np.int64)
    return dict(zip(meta["SUB_ID"], meta["label"])), meta

def load_input_file(p: Path, n_rois: int = 200) -> np.ndarray:
    ext = p.name.lower()
    if not ext.endswith(ALLOWED_EXTENSIONS):
        raise ValueError(f"Unsupported file format for {p.name}. Allowed formats: {ALLOWED_EXTENSIONS}")
    
    if ext.endswith('.nii'):
        ts = nib.load(p).get_fdata().astype(np.float32)
    else:
        img = Image.open(p).convert('L')
        ts = np.array(img, dtype=np.float32)

    if ts.ndim == 3:
        ts = ts.mean(axis=-1)
    elif ts.ndim == 4:
        ts = ts.reshape(-1, ts.shape[-1]).T
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    if ts.shape[1] < n_rois or ts.shape[0] < 50:
        img_temp = Image.fromarray(ts)
        new_w = max(n_rois, ts.shape[1])
        new_h = max(50, ts.shape[0])
        img_temp = img_temp.resize((new_w, new_h))
        ts = np.array(img_temp, dtype=np.float32)

    if ts.shape[1] >= n_rois:
        ts = ts[:, :n_rois]
        
    return ts

def get_functional_time_series(func_dir: Path, label_map: dict, n_rois: int = 200):
    print(f"Scanning all files recursively inside: {func_dir.absolute()}")
    
    all_files = [p for p in func_dir.rglob("*") if p.is_file()]
    print(f"Total files found in directory tree: {len(all_files)}")
    
    # Strictly allow ONLY .nii, .jpg, and .png input files
    files = [p for p in all_files if p.name.lower().endswith(ALLOWED_EXTENSIONS)]
    print(f"Filtered functional data candidates: {len(files)}")
    
    if len(files) == 0 and len(all_files) > 0:
        print(f"Sample files found in directory: {[p.name for p in all_files[:5]]}")

    def sid_of(p):
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else None

    # Group candidate files by (top_folder, subject_id) and select 1 scan per subject per dataset folder
    groups = {}
    for p in files:
        sid = sid_of(p)
        if sid is None or sid not in label_map:
            continue
        rel = p.relative_to(func_dir)
        top_folder = rel.parts[0]
        key = (top_folder, sid)
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    selected = []
    ext_order = {'.nii': 0, '.png': 1, '.jpg': 2}
    for key, file_list in groups.items():
        file_list.sort(key=lambda f: ext_order.get(f.suffix.lower(), 99))
        selected.append(file_list[0])

    ts_list, labels = [], []
    for p in selected:
        sid = sid_of(p)
        try:
            ts = load_input_file(p, n_rois=n_rois)
            if ts.shape[1] != n_rois or ts.shape[0] < 50 or not np.isfinite(ts).all(): 
                continue
            ts_list.append(ts)
            labels.append(label_map[sid])
        except Exception:
            continue
            
    print(f"Successfully matched and loaded {len(ts_list)} subject time-series files.")
    return ts_list, np.array(labels, dtype=np.int64)