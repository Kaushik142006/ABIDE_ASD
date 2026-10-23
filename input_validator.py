# input_validator.py
# -*- coding: utf-8 -*-
"""
input_validator.py
------------------
Dedicated input validation module for the ABIDE ASD Detection Dashboard.

Objective:
    Gate every uploaded file BEFORE it reaches the prediction model to prevent
    unsuitable, corrupted, out-of-distribution (OOD), or incompatible images
    (e.g., arbitrary photographs, memes, screenshots, landscapes, animals,
    objects, or static structural MRI scans) from reaching the fMRI-trained
    ASD classification model.

Requirements satisfied:
    - Pure validation layer: does not modify or retrain model code, weights,
      preprocessing pipeline, or evaluation scripts.
    - Content-based discrimination: evaluates empirical statistical properties
      derived from the full ABIDE fMRI training set (2,221 images).
    - Prevents false rejections: thresholds calibrated with safety margins
      relative to the observed extremes of genuine training data.
    - Prevents false acceptances: multi-signal checks catch natural photos,
      cartoons, screenshots, memes, textured noise, and dark scenes.
    - Explicit handling of static anatomical MRI scans: detects and rejects
      static structural MRI scans (both 3D volumes and 2D slices) with clear
      domain explanations, because the model requires resting-state fMRI
      functional dynamics rather than static anatomical images.
"""

from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
from PIL import Image


# =============================================================================
# Whitelist and Dimensionality Constraints
# =============================================================================

ALLOWED_EXTENSIONS: tuple[str, ...] = (".nii", ".jpg", ".jpeg", ".png")

# Minimum image dimensions for 2D inputs
MIN_WIDTH_PX: int = 50
MIN_HEIGHT_PX: int = 50

# Aspect ratio constraint: genuine ABIDE connectivity images are square (224x224)
# Allow up to 4.0:1 for rectangular connectivity matrices / time-series crops
MAX_ASPECT_RATIO: float = 4.0

# Minimum required time-points for 4D fMRI or 2D time-series matrices
MIN_TIMEPOINTS: int = 50


# =============================================================================
# Content-Based Statistical Thresholds (Calibrated on all 2,221 ABIDE PNGs)
#
# Training population statistics (N = 2,221):
#   - Mean brightness:        [0.0575, 0.3712]
#   - Standard deviation:     [0.0679, 0.2807]
#   - HF FFT energy ratio:    [0.3653, 0.6859]
#   - Shannon entropy:        [5.0483, 7.5842] bits
#   - Unique intensity levels: min = 137, median = 231
# =============================================================================

# Hard limits with safety margins
MAX_MEAN_BRIGHTNESS: float = 0.40    # Training max: 0.3712; photos/screenshots: > 0.45
MAX_STD_DEV: float = 0.30            # Training max: 0.2807; high-contrast memes: > 0.30
MAX_HF_RATIO: float = 0.72           # Training max: 0.6859; natural photos/noise: > 0.80
MIN_ENTROPY: float = 4.80            # Training min: 5.0483; screenshots/memes: < 4.0
MIN_UNIQUE_LEVELS: int = 80          # Training min: 137; digital graphics/UI: < 40

# Colour chroma threshold for RGB/RGBA uploads
MAX_COLOR_CHROMA: float = 8.0        # Average inter-channel difference (0 = greyscale)
MIN_COLOR_MEAN: float = 0.25         # Chroma check active when mean brightness > 0.25

# Structural/Anatomical Brain MRI Detection
# In anatomical scans (e.g. T1/T2 slices), bright brain tissue sits inside a black
# background (empty air outside the skull), yielding bg_frac > 0.35 and fg_mean > 0.40
ANAT_BG_THRESHOLD: float = 0.35      # Fraction of background pixels (< 5 intensity)
ANAT_FG_MEAN_THRESHOLD: float = 0.40  # Mean brightness of foreground tissue
ANAT_SCORE_THRESHOLD: float = 0.16   # Product of bg_frac * fg_mean (training max < 0.11)
ANAT_STD_THRESHOLD: float = 0.24     # High tissue-to-air contrast


# =============================================================================
# Helper Functions: NIfTI Analysis
# =============================================================================

def _parse_nii_header(data: bytes) -> Tuple[bool, Optional[int], Optional[Tuple[int, ...]], Optional[str]]:
    """
    Parse the NIfTI header bytes (supporting both uncompressed and gzip NIfTI).
    Returns: (is_valid, ndim, shape, error_msg)
    """
    if len(data) < 8:
        return False, None, None, "File too small to contain a valid NIfTI header."

    # Decompress gzip header if compressed
    raw = data
    if raw[:2] == b"\x1f\x8b":
        try:
            # Decompress at least first 1024 bytes to read the header
            raw = gzip.decompress(data[:65536])
        except Exception:
            try:
                raw = gzip.decompress(data)
            except Exception as e:
                return False, None, None, f"Failed to decompress gzipped NIfTI file: {e}"

    # Check NIfTI-1 (header size 348)
    if len(raw) >= 348:
        sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
        if sizeof_hdr == 348:
            magic = raw[344:348]
            if magic in (b"n+1\x00", b"ni1\x00"):
                dim = struct.unpack_from("<8h", raw, 40)
                ndim = int(dim[0])
                if 1 <= ndim <= 7:
                    shape = tuple(int(dim[i]) for i in range(1, ndim + 1))
                    return True, ndim, shape, None

    # Check NIfTI-2 (header size 540)
    if len(raw) >= 540:
        sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
        if sizeof_hdr == 540:
            magic = raw[4:8]
            if magic in (b"n+2\x00", b"ni2\x00"):
                dim = struct.unpack_from("<8q", raw, 16)
                ndim = int(dim[0])
                if 1 <= ndim <= 7:
                    shape = tuple(int(dim[i]) for i in range(1, ndim + 1))
                    return True, ndim, shape, None

    return False, None, None, "Invalid NIfTI magic bytes or header structure."


def _validate_nii_input(data: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate a NIfTI (.nii) file for domain suitability.
    Distinguishes 4D fMRI or 2D preprocessed matrices from incompatible 3D static MRI.
    """
    is_valid_hdr, ndim, shape, err = _parse_nii_header(data)
    if not is_valid_hdr or ndim is None or shape is None:
        return False, (
            "❌ **Invalid NIfTI File**\n\n"
            f"The uploaded file failed NIfTI header validation: {err}. "
            "Please ensure the file is a valid, uncorrupted `.nii` neuroimaging scan."
        )

    # Incompatible: 3D static structural/anatomical MRI volume
    if ndim == 3:
        return False, (
            "❌ **Incompatible Input: Static 3D Structural/Anatomical MRI Scan Detected**\n\n"
            f"The uploaded scan is a static 3D anatomical volume (shape: {shape}, ndim=3) "
            "with no time-series dimension.\n\n"
            "**Domain Requirement:** The ABIDE ASD detection model was developed and trained "
            "exclusively on **functional neuroimaging** data (resting-state fMRI). It requires "
            "multi-timepoint functional BOLD signals to compute functional connectivity matrices.\n\n"
            "Static structural MRI scans (such as T1-weighted MPRAGE or T2-weighted anatomical images) "
            "cannot be processed by this model because spatial voxel rows cannot substitute for "
            "temporal recordings."
        )

    # Incompatible: 1D or >4D
    if ndim not in (2, 4):
        return False, (
            f"❌ **Incompatible Input Dimensions ({ndim}D)**\n\n"
            f"Expected either a 4D fMRI scan (X, Y, Z, Time) or a 2D functional connectivity matrix. "
            f"Received a scan with shape {shape} ({ndim} dimensions)."
        )

    # 4D fMRI: check time points
    if ndim == 4:
        timepoints = shape[3]
        if timepoints < MIN_TIMEPOINTS:
            return False, (
                f"❌ **Insufficient Time Points in 4D fMRI Scan**\n\n"
                f"The uploaded 4D fMRI scan has only {timepoints} temporal volumes. "
                f"At least {MIN_TIMEPOINTS} time points are required to compute stable "
                "resting-state functional connectivity."
            )
        return True, None

    # 2D matrix: check dimensions
    if ndim == 2:
        rows, cols = shape[0], shape[1]
        if rows < MIN_TIMEPOINTS or cols < MIN_WIDTH_PX:
            return False, (
                f"❌ **Incompatible 2D Matrix Dimensions ({rows}x{cols})**\n\n"
                f"Expected at least {MIN_TIMEPOINTS} rows and {MIN_WIDTH_PX} columns "
                "for a functional time series / connectivity matrix."
            )
        return True, None

    return True, None


# =============================================================================
# Helper Functions: Image Feature Extraction
# =============================================================================

def _compute_image_features(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Extract content-based diagnostic metrics from an uploaded PIL image.
    """
    # Color mode and inter-channel chroma
    mode = pil_img.mode
    has_color = False
    chroma = 0.0
    if mode in ("RGB", "RGBA"):
        rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
        # Average pairwise difference between R, G, B channels
        diff_rg = np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1])
        diff_rb = np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 2])
        diff_gb = np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2])
        chroma = float((diff_rg.mean() + diff_rb.mean() + diff_gb.mean()) / 3.0)
        has_color = chroma > MAX_COLOR_CHROMA

    # Greyscale representation
    grey_img = pil_img.convert("L")
    arr = np.array(grey_img, dtype=np.float32)
    norm = arr / 255.0

    mean_val = float(norm.mean())
    std_val = float(norm.std())

    # Unique intensity levels
    arr_u8 = arr.clip(0, 255).astype(np.uint8)
    n_unique = int(len(np.unique(arr_u8)))

    # Shannon Entropy of 256-bin intensity histogram
    counts, _ = np.histogram(arr_u8, bins=256, range=(0, 256))
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))

    # High-frequency spatial energy ratio (2D FFT)
    # Resize to fixed 128x128 for scale-invariant FFT computation
    small_grey = grey_img.resize((128, 128), Image.BILINEAR)
    small_arr = np.array(small_grey, dtype=np.float64)
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(small_arr)))
    total_energy = float(fft_mag.sum() + 1e-12)
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    ry, rx = h // 8, w // 8  # central 25% linear band (low frequencies)
    centre_energy = float(fft_mag[cy - ry:cy + ry, cx - rx:cx + rx].sum())
    hf_ratio = float((total_energy - centre_energy) / total_energy)

    # Structural MRI foreground/background metrics
    # Background air: pixels with intensity < 5
    bg_frac = float((arr < 5).mean())
    fg_pixels = arr[arr >= 10]
    fg_mean = float((fg_pixels.mean() / 255.0)) if len(fg_pixels) > 0 else 0.0
    anat_score = float(bg_frac * fg_mean)

    return {
        "mode": mode,
        "chroma": chroma,
        "has_color": has_color,
        "mean": mean_val,
        "std": std_val,
        "n_unique": n_unique,
        "entropy": entropy,
        "hf_ratio": hf_ratio,
        "bg_frac": bg_frac,
        "fg_mean": fg_mean,
        "anat_score": anat_score,
        "width": pil_img.width,
        "height": pil_img.height,
    }


# =============================================================================
# Helper Functions: Image Domain Validation
# =============================================================================

def _validate_image_content(features: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate content metrics against the ABIDE domain requirements.
    Returns (is_valid, rejection_reason).
    """
    reasons: List[str] = []
    category_note: Optional[str] = None

    # Check 1: Static Anatomical/Structural Brain MRI Slice
    # An image with a bright brain surrounded by black background air outside skull
    if (features["anat_score"] >= ANAT_SCORE_THRESHOLD and
        features["bg_frac"] >= ANAT_BG_THRESHOLD and
        features["fg_mean"] >= ANAT_FG_MEAN_THRESHOLD and
        features["std"] >= ANAT_STD_THRESHOLD):
        return False, (
            "❌ **Incompatible Input: Static Anatomical/Structural Brain MRI Slice Detected**\n\n"
            "The uploaded image appears to be a 2D static anatomical/structural MRI slice "
            "(e.g., T1/T2 brain anatomy scan showing brain tissue surrounded by background air).\n\n"
            "**Domain Requirement:** The ABIDE ASD model is trained on **functional neuroimaging** "
            "(resting-state fMRI BOLD signals and functional connectivity matrices). "
            "A static structural photograph of brain anatomy does not contain time-series dynamics "
            "and cannot produce valid functional connectivity features.\n\n"
            f"*Metrics: background fraction={features['bg_frac']:.1%}, "
            f"tissue brightness={features['fg_mean']:.2f}, anatomical score={features['anat_score']:.3f}*"
        )

    # Check 2: Natural Color Photograph
    if features["has_color"] and features["mean"] >= MIN_COLOR_MEAN:
        reasons.append(
            f"Image contains vivid color chroma (channel difference = {features['chroma']:.1f}). "
            "ABIDE functional connectivity inputs are natively greyscale representations."
        )
        category_note = "Natural color photograph or illustration"

    # Check 3: Excessive Mean Brightness
    if features["mean"] > MAX_MEAN_BRIGHTNESS:
        reasons.append(
            f"Mean brightness is {features['mean']:.3f} (exceeds domain maximum of {MAX_MEAN_BRIGHTNESS:.2f}). "
            "Natural photographs, screenshots, and documents have high average luminance, whereas "
            "valid fMRI connectivity representations are predominantly dark (training mean: 0.06 - 0.37)."
        )
        if not category_note:
            category_note = "Overly bright image (document, screenshot, or light photograph)"

    # Check 4: Excessive Standard Deviation (High Contrast)
    if features["std"] > MAX_STD_DEV:
        reasons.append(
            f"Pixel intensity standard deviation is {features['std']:.3f} (exceeds domain maximum of {MAX_STD_DEV:.2f}). "
            "High-contrast patterns, memes, or extreme black-and-white images violate fMRI signal continuity."
        )

    # Check 5: High-Frequency Spatial Energy Ratio (Textures, Noise, Real-World Photos)
    if features["hf_ratio"] > MAX_HF_RATIO:
        reasons.append(
            f"High-frequency spatial energy ratio is {features['hf_ratio']:.3f} (exceeds domain maximum of {MAX_HF_RATIO:.2f}). "
            "Real-world photographs, fine textures, camera sensor noise, and random images possess sharp "
            "spatial gradients absent in smooth neuroimaging connectivity matrices."
        )
        if not category_note:
            category_note = "High-frequency textured image or photograph"

    # Check 6: Unusually Low Entropy (Memes, Cartoons, UI Screenshots, Charts)
    if features["entropy"] < MIN_ENTROPY:
        reasons.append(
            f"Intensity entropy is {features['entropy']:.2f} bits (below domain minimum of {MIN_ENTROPY:.2f} bits). "
            "Memes, UI screenshots, digital charts, and cartoons contain flat color blocks and sharp text, "
            "producing artificially low entropy compared to continuous fMRI connectivity matrices (training: 5.05 - 7.58 bits)."
        )
        if not category_note:
            category_note = "Digital graphic, screenshot, or meme"

    # Check 7: Too Few Unique Intensity Levels (Flat Graphics, Icons, Diagrams)
    if features["n_unique"] < MIN_UNIQUE_LEVELS:
        reasons.append(
            f"Image contains only {features['n_unique']} unique intensity levels (below domain minimum of {MIN_UNIQUE_LEVELS}). "
            "Neuroimaging signals contain continuous gradients with at least 130+ distinct levels."
        )
        if not category_note:
            category_note = "Synthetic diagram or low-color graphic"

    if reasons:
        note_str = f"**Identified Type:** {category_note}\n\n" if category_note else ""
        details = "\n".join(f"- {r}" for r in reasons)
        message = (
            "❌ **Image Rejected: Domain Mismatch Detected**\n\n"
            f"{note_str}"
            "The uploaded file is not a valid fMRI-derived functional connectivity matrix "
            "compatible with the ABIDE ASD detection model.\n\n"
            "**Triggered Criteria:**\n"
            f"{details}\n\n"
            f"**Diagnostic Summary:** mean={features['mean']:.3f}, std={features['std']:.3f}, "
            f"HF-ratio={features['hf_ratio']:.3f}, entropy={features['entropy']:.2f} bits, "
            f"unique levels={features['n_unique']}\n\n"
            "Please upload a genuine `.nii` fMRI file or a 224x224 greyscale fMRI connectivity image."
        )
        return False, message

    return True, None


# =============================================================================
# Public Validation Entry Point
# =============================================================================

def validate_upload(filename: str, file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded file before it reaches the prediction model.

    Parameters:
        filename: Original file name.
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        (True, None) if the input passes all validation checks and is domain-compatible.
        (False, error_message) if the input is rejected.
    """
    # ── Step 1: File Extension Whitelist ─────────────────────────────────────
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"❌ **Unsupported File Format: `{ext or '(no extension)'}`**\n\n"
            f"Supported formats are strictly: {', '.join('`' + e + '`' for e in ALLOWED_EXTENSIONS)}.\n"
            "Please upload a valid neuroimaging file."
        )

    # ── Step 2: Non-Empty File Guard ─────────────────────────────────────────
    if not file_bytes or len(file_bytes) == 0:
        return False, (
            "❌ **Empty File Uploaded**\n\n"
            "The uploaded file contains 0 bytes. Please select a valid file."
        )

    # ── Step 3: NIfTI Path ───────────────────────────────────────────────────
    if ext == ".nii":
        return _validate_nii_input(file_bytes)

    # ── Step 4: Image Decodability (PIL Integrity Verification) ──────────────
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
    except Exception as exc:
        return False, (
            "❌ **Corrupted or Unreadable Image File**\n\n"
            f"The uploaded image could not be decoded (`{type(exc).__name__}: {exc}`). "
            "Please ensure the file is not corrupted or truncated."
        )

    # Re-open image stream after verify()
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        return False, (
            "❌ **Could Not Read Image Data**\n\n"
            f"Error opening image stream: {exc}"
        )

    # ── Step 5: Geometric and Resolution Constraints ─────────────────────────
    width, height = pil_img.size
    if width < MIN_WIDTH_PX or height < MIN_HEIGHT_PX:
        return False, (
            f"❌ **Image Resolution Too Small ({width}x{height} px)**\n\n"
            f"A minimum dimension of {MIN_WIDTH_PX}x{MIN_HEIGHT_PX} pixels is required. "
            "Genuine ABIDE functional connectivity images are 224x224 pixels."
        )

    aspect_ratio = max(width, height) / max(min(width, height), 1)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, (
            f"❌ **Extreme Aspect Ratio ({width}x{height} px, ratio {aspect_ratio:.1f}:1)**\n\n"
            f"The image aspect ratio exceeds the allowed limit of {MAX_ASPECT_RATIO:.1f}:1. "
            "ABIDE connectivity images are square matrices (1:1). Extreme rectangular images "
            "(such as banners, panoramic photos, or strip screenshots) are not supported."
        )

    # ── Step 6: Content-Based Domain Validation & OOD Detection ──────────────
    features = _compute_image_features(pil_img)
    return _validate_image_content(features)
