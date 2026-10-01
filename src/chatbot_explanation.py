# src/chatbot_explanation.py
# ---------------------------------------------------------------
# Post-hoc model explanation engine.
# Analyses actual model outputs WITHOUT modifying the model.
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from src.chatbot_context import InferenceContext


# ================================================================
# Confidence interpretation
# ================================================================

def interpret_confidence(prob: float) -> str:
    """Map a probability value to a qualitative description."""
    if prob >= 0.95:
        return "very high confidence"
    if prob >= 0.85:
        return "high confidence"
    if prob >= 0.70:
        return "moderate confidence"
    if prob >= 0.55:
        return "low-to-moderate confidence"
    return "low confidence"


def interpret_decision_margin(df_value: float) -> str:
    """Describe how far the sample is from the SVM decision boundary."""
    adv = abs(df_value)
    if adv > 2.0:
        return "well beyond the decision boundary"
    if adv > 1.0:
        return "clearly on one side of the decision boundary"
    if adv > 0.5:
        return "moderately separated from the decision boundary"
    return "relatively close to the decision boundary"


# ================================================================
# Feature contribution approximation
# ================================================================

def top_feature_contributions(
    ctx: InferenceContext,
    top_n: int = 10,
) -> Optional[List[Tuple[int, float]]]:
    """Identify which of the selected features have the largest
    absolute scaled values for the current sample.

    Returns a list of (feature_index, absolute_value) tuples,
    sorted descending by absolute value.

    This does NOT modify the model – it only reads the already-
    computed scaled feature vector.
    """
    if ctx.feature_values is None:
        return None
    try:
        vals = np.asarray(ctx.feature_values).flatten()
        indices = np.argsort(np.abs(vals))[::-1][:top_n]
        return [(int(idx), float(vals[idx])) for idx in indices]
    except Exception:
        return None


# ================================================================
# Explanation text generators
# ================================================================

def explain_prediction(ctx: InferenceContext) -> str:
    """Generate a natural-language explanation of why the model
    produced its current prediction."""
    if not ctx.has_prediction():
        return "No prediction is available yet. Please upload a scan file first."

    parts: list = []
    label = ctx.prediction_label or ("ASD" if ctx.prediction == 1 else "Control")

    # Core prediction statement
    parts.append(
        f"The model classified this sample as **{label}**."
    )

    # Probability reasoning
    if ctx.asd_probability is not None:
        asd_p = ctx.asd_probability
        con_p = ctx.control_probability or (1 - asd_p)
        dominant_p = asd_p if ctx.prediction == 1 else con_p
        conf_desc = interpret_confidence(dominant_p)
        parts.append(
            f"It assigned a {dominant_p:.1%} probability to the predicted class, "
            f"which represents {conf_desc}."
        )

    # Decision function insight
    if ctx.decision_function_value is not None:
        margin_desc = interpret_decision_margin(ctx.decision_function_value)
        side = "positive (ASD)" if ctx.decision_function_value > 0 else "negative (Control)"
        parts.append(
            f"The SVM decision function value is {ctx.decision_function_value:+.4f}, "
            f"placing this sample on the {side} side, "
            f"{margin_desc}."
        )

    # Feature contribution summary
    contributions = top_feature_contributions(ctx, top_n=5)
    if contributions:
        parts.append(
            "Among the selected connectivity features, the ones with the "
            "largest influence on this particular prediction (by absolute "
            "scaled value) are feature indices: "
            + ", ".join(f"#{idx} ({val:+.3f})" for idx, val in contributions)
            + ". Larger absolute values indicate a stronger contribution "
            "to the model's decision for this specific sample."
        )

    # Caveat
    parts.append(
        "Note: These explanations describe the **model's behaviour**, not "
        "biological causation. A feature being influential in the model's "
        "decision does not imply it is an inherent biomarker for ASD."
    )

    return "\n\n".join(parts)


def explain_probability(ctx: InferenceContext) -> str:
    """Describe the probability distribution for the current sample."""
    if not ctx.has_prediction():
        return "No prediction has been made yet. Please upload a file first."

    if ctx.asd_probability is None:
        return "Probability information is not available for this prediction."

    asd_p = ctx.asd_probability
    con_p = ctx.control_probability or (1 - asd_p)
    label = ctx.prediction_label or ("ASD" if ctx.prediction == 1 else "Control")
    conf = interpret_confidence(max(asd_p, con_p))

    return (
        f"For the current sample, the model estimates:\n\n"
        f"- **ASD probability**: {asd_p:.2%}\n"
        f"- **Control probability**: {con_p:.2%}\n\n"
        f"The predicted class is **{label}** with {conf}. "
        f"These probabilities are derived from the SVM's calibrated "
        f"probability estimates using Platt scaling."
    )


def explain_probability_qualifier(ctx: InferenceContext) -> str:
    """Answer 'is that high?' style follow-ups about probability."""
    if not ctx.has_prediction() or ctx.asd_probability is None:
        return "I don't have probability information to evaluate right now."

    dominant_p = max(ctx.asd_probability, ctx.control_probability or 0)
    conf = interpret_confidence(dominant_p)
    label = ctx.prediction_label or ("ASD" if ctx.prediction == 1 else "Control")

    if dominant_p >= 0.85:
        assessment = (
            f"Yes, {dominant_p:.1%} is a relatively high probability. "
            f"The model is fairly confident in its **{label}** classification."
        )
    elif dominant_p >= 0.65:
        assessment = (
            f"The probability of {dominant_p:.1%} indicates {conf}. "
            f"The model leans toward **{label}** but is not highly certain."
        )
    else:
        assessment = (
            f"At {dominant_p:.1%}, this is {conf}. The model's classification "
            f"of **{label}** is not particularly strong, meaning the sample's "
            f"features are relatively close to the decision boundary."
        )
    return assessment


def explain_model(ctx: InferenceContext) -> str:
    """Describe the model used for classification."""
    return (
        f"The classifier is a **Support Vector Machine (SVM)** with a "
        f"**Radial Basis Function (RBF)** kernel.\n\n"
        f"- **Model type**: {ctx.model_type}\n"
        f"- **Kernel**: {ctx.svm_kernel.upper()} (Radial Basis Function)\n"
        f"- **C (regularisation)**: {ctx.svm_c}\n"
        f"- **Gamma**: {ctx.svm_gamma}\n"
        f"- **Class weights**: {ctx.svm_class_weight}\n\n"
        f"SVM finds a decision boundary (hyperplane) that maximally separates "
        f"ASD from Control in a high-dimensional feature space. The RBF "
        f"kernel allows it to handle non-linear relationships between features."
    )


def explain_model_params(ctx: InferenceContext) -> str:
    """Explain SVM hyperparameters."""
    return (
        "Here's what the key hyperparameters mean:\n\n"
        f"- **C = {ctx.svm_c}**: The regularisation parameter. Higher C values "
        f"make the model try harder to correctly classify every training sample, "
        f"at the risk of overfitting. C = {ctx.svm_c} is a moderately high value "
        f"that balances accuracy with generalisation.\n\n"
        f"- **Kernel = RBF**: The Radial Basis Function (Gaussian) kernel maps "
        f"input features into a higher-dimensional space where a linear separator "
        f"can be found. It's effective for problems with non-linear decision "
        f"boundaries.\n\n"
        f"- **Gamma = '{ctx.svm_gamma}'**: When set to 'scale', gamma is "
        f"automatically computed as 1 / (n_features × variance). This adapts "
        f"the kernel's sensitivity to the data.\n\n"
        f"- **Class weights = '{ctx.svm_class_weight}'**: The 'balanced' setting "
        f"automatically adjusts weights inversely proportional to class frequencies, "
        f"giving the minority class more importance during training."
    )


def explain_features(ctx: InferenceContext) -> str:
    """Explain the feature extraction and selection process."""
    return (
        f"The system uses **{ctx.selected_feature_count}** selected "
        f"functional connectivity features (edges) out of a total of "
        f"**{ctx.total_edge_count:,}** possible edges.\n\n"
        f"**How features are generated:**\n"
        f"1. The brain is parcellated into **{ctx.n_rois} regions of interest "
        f"(ROIs)** using the CC200 atlas.\n"
        f"2. Pearson correlation is computed between every pair of ROIs, "
        f"producing a {ctx.n_rois}×{ctx.n_rois} correlation matrix.\n"
        f"3. Fisher's r-to-z transformation (arctanh) is applied to normalise "
        f"the correlation values.\n"
        f"4. The upper triangle of this matrix contains "
        f"{ctx.total_edge_count:,} unique edges "
        f"({ctx.n_rois}×{ctx.n_rois - 1}/2).\n"
        f"5. **Welch's t-test** compares each edge between the ASD and Control "
        f"groups, and the top {ctx.selected_feature_count} edges with the "
        f"largest absolute t-statistic are selected as the most discriminative "
        f"features.\n"
        f"6. These selected features are standardised using a StandardScaler "
        f"before being fed to the SVM."
    )


def explain_pipeline(ctx: InferenceContext) -> str:
    """Describe the end-to-end processing pipeline."""
    return (
        "The system follows this end-to-end pipeline:\n\n"
        "1. **Input**: A brain scan file (.nii, .jpg, or .png) is uploaded.\n"
        f"2. **ROI Extraction**: The scan data is mapped to {ctx.n_rois} "
        f"regions of interest (ROIs) using the CC200 brain atlas.\n"
        "3. **Preprocessing**: Time-series data is cleaned by clipping values "
        "at the 5th and 95th percentiles to remove outliers.\n"
        "4. **Standardisation**: Each ROI's time-series is Z-score standardised "
        "(mean=0, std=1).\n"
        f"5. **Functional Connectivity**: A {ctx.n_rois}×{ctx.n_rois} Pearson "
        f"correlation matrix is computed, then Fisher-transformed (arctanh).\n"
        f"6. **Feature Selection**: The top {ctx.selected_feature_count} most "
        f"discriminative edges are selected from {ctx.total_edge_count:,} total "
        f"using Welch's t-test.\n"
        "7. **Scaling**: Selected features are standardised with a fitted "
        "StandardScaler.\n"
        "8. **Classification**: The RBF-SVM model predicts ASD vs Control "
        "with probability estimates."
    )


def explain_roi(ctx: InferenceContext) -> str:
    """Explain ROIs and the CC200 atlas."""
    return (
        f"**ROI** stands for **Region of Interest**. In this system, the "
        f"brain is divided into **{ctx.n_rois} ROIs** using the **CC200 "
        f"(Craddock 200) atlas**.\n\n"
        f"The CC200 atlas is a functional brain parcellation that groups "
        f"brain voxels into {ctx.n_rois} spatially contiguous, functionally "
        f"homogeneous regions. Each ROI represents a cluster of brain voxels "
        f"that tend to activate together.\n\n"
        f"The time-series signal from each ROI is extracted and used to "
        f"compute pairwise functional connectivity between all "
        f"{ctx.n_rois} regions."
    )


def explain_fc(ctx: InferenceContext) -> str:
    """Explain functional connectivity and Fisher transform."""
    return (
        "**Functional connectivity (FC)** measures the statistical "
        "relationship between the neural activity of different brain "
        "regions.\n\n"
        f"In this system, FC is computed as the **Pearson correlation** "
        f"between the time-series of every pair of the {ctx.n_rois} ROIs, "
        f"producing a {ctx.n_rois}×{ctx.n_rois} correlation matrix.\n\n"
        "The **Fisher r-to-z transformation** (arctanh) is then applied:\n\n"
        "  z = arctanh(r) = ½ ln((1+r)/(1−r))\n\n"
        "This transformation converts bounded correlation values (−1 to +1) "
        "into an approximately normal distribution, making them more suitable "
        "for statistical analysis and machine learning."
    )


def explain_preprocessing(ctx: InferenceContext) -> str:
    """Explain the preprocessing steps."""
    return (
        "The preprocessing pipeline involves two main steps:\n\n"
        "1. **Percentile clipping (outlier removal)**: For each ROI, values "
        "below the 5th percentile and above the 95th percentile are clipped. "
        "This reduces the impact of extreme values caused by motion artefacts "
        "or scanner noise.\n\n"
        "2. **Z-score standardisation**: Each ROI's time-series is transformed "
        "to have zero mean and unit variance using scikit-learn's "
        "StandardScaler. This ensures that the subsequent Pearson correlation "
        "computation works correctly and that no ROI dominates due to "
        "differences in signal amplitude."
    )


def explain_ttest(ctx: InferenceContext) -> str:
    """Explain Welch's t-test and feature selection."""
    return (
        "**Welch's two-sample t-test** is used for feature selection in "
        "this system.\n\n"
        "For each of the {:,} functional connectivity edges, the t-test "
        "compares the edge values between the ASD group and the Control "
        "group. Unlike the standard t-test, Welch's version does **not** "
        "assume equal variance between the two groups, making it more "
        "robust.\n\n"
        "The absolute t-statistic |t| measures how different the two "
        "groups are for that edge. Edges with the largest |t| values "
        "show the most significant differences between ASD and Control.\n\n"
        "The top {} edges with the highest |t| values are selected as "
        "the most discriminative features for the SVM classifier."
    ).format(ctx.total_edge_count, ctx.selected_feature_count)


def explain_evaluation(ctx: InferenceContext) -> str:
    """Explain evaluation metrics at a conceptual level."""
    return (
        "The model is evaluated using several standard metrics:\n\n"
        "- **Accuracy**: The proportion of all samples correctly classified.\n"
        "- **Balanced Accuracy**: The average of sensitivity and specificity, "
        "accounting for class imbalance.\n"
        "- **Sensitivity (Recall)**: The proportion of actual ASD cases "
        "correctly identified. High sensitivity means fewer missed ASD cases.\n"
        "- **Specificity**: The proportion of actual Control cases correctly "
        "identified. High specificity means fewer false alarms.\n"
        "- **F1-Score**: The harmonic mean of precision and sensitivity, "
        "providing a single score that balances both.\n"
        "- **ROC-AUC**: Area Under the Receiver Operating Characteristic "
        "curve. Values closer to 1.0 indicate better discrimination.\n"
        "- **MCC (Matthews Correlation Coefficient)**: A balanced metric "
        "ranging from −1 to +1 that accounts for all four quadrants of "
        "the confusion matrix. +1 is perfect, 0 is random, −1 is total "
        "disagreement."
    )


def explain_confusion_matrix(ctx: InferenceContext) -> str:
    """Explain the confusion matrix and its quadrants."""
    return (
        "The **confusion matrix** is a 2×2 table that summarises "
        "classification results:\n\n"
        "```\n"
        "                 Predicted Control | Predicted ASD\n"
        "Actual Control |       TN         |      FP\n"
        "Actual ASD     |       FN         |      TP\n"
        "```\n\n"
        "- **True Positive (TP)**: ASD correctly identified as ASD.\n"
        "- **True Negative (TN)**: Control correctly identified as Control.\n"
        "- **False Positive (FP)**: Control incorrectly classified as ASD "
        "(Type I error).\n"
        "- **False Negative (FN)**: ASD incorrectly classified as Control "
        "(Type II error).\n\n"
        "In medical screening, **false negatives** (missed ASD cases) are "
        "typically considered more concerning than false positives."
    )


def explain_limitations(ctx: InferenceContext) -> str:
    """Explain the system's limitations."""
    return (
        "Important limitations to keep in mind:\n\n"
        "1. **Not a clinical diagnosis**: This is a research/prototype "
        "system. Its output is a statistical prediction, not a medical "
        "diagnosis.\n"
        "2. **Model can be wrong**: No classifier is 100% accurate. The "
        "model can produce both false positives and false negatives.\n"
        "3. **Training data dependent**: The model's performance depends "
        "on the ABIDE dataset it was trained on, which may not represent "
        "all populations equally.\n"
        "4. **Feature limitations**: The system uses only functional "
        "connectivity patterns and does not incorporate structural imaging, "
        "clinical assessments, behavioural data, or demographic factors.\n"
        "5. **Input sensitivity**: Prediction quality depends on the "
        "quality and format of the uploaded scan file.\n"
        "6. **Single time-point**: The analysis is based on a single scan "
        "and does not capture longitudinal changes."
    )


def explain_medical_disclaimer(ctx: InferenceContext) -> str:
    """Respond to diagnosis-related questions with a clear disclaimer."""
    label_info = ""
    if ctx.has_prediction():
        label = ctx.prediction_label or ("ASD" if ctx.prediction == 1 else "Control")
        label_info = (
            f" The model classified this particular input as **{label}**, "
            f"but this classification "
        )
    else:
        label_info = " Any classification this system produces "

    return (
        "**This system is NOT a clinical diagnostic tool.**\n\n"
        f"{label_info}is a statistical prediction from a machine learning "
        f"model trained on research data. It does **not** constitute a "
        f"medical diagnosis of Autism Spectrum Disorder.\n\n"
        f"A clinical ASD diagnosis requires comprehensive evaluation by "
        f"qualified healthcare professionals, including:\n"
        f"- Developmental history assessment\n"
        f"- Behavioural observation\n"
        f"- Standardised diagnostic instruments (e.g., ADOS-2, ADI-R)\n"
        f"- Clinical interview\n\n"
        f"If you have concerns about ASD, please consult a qualified "
        f"healthcare professional such as a developmental paediatrician, "
        f"psychiatrist, or psychologist."
    )


def explain_file_info(ctx: InferenceContext) -> str:
    """Describe the currently uploaded file."""
    if not ctx.has_upload():
        return "No file has been uploaded yet."

    parts = [f"**Uploaded file**: {ctx.filename or 'Unknown'}"]
    if ctx.extension:
        parts.append(f"**Format**: {ctx.extension}")
    if ctx.input_shape:
        parts.append(f"**Input dimensions**: {ctx.input_shape}")
    parts.append(f"**Processing status**: {ctx.processing_status}")
    if ctx.processing_error:
        parts.append(f"**Error**: {ctx.processing_error}")

    return "\n".join(parts)
