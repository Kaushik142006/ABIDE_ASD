# tests/test_chatbot.py
# ---------------------------------------------------------------
# Comprehensive test suite for the rule-based offline chatbot.
# Tests at least 20 distinct question categories and edge cases:
#  1. normal prediction question
#  2. ASD prediction context
#  3. Control prediction context
#  4. probability question
#  5. model question
#  6. feature question
#  7. pipeline question
#  8. evaluation question
#  9. typo handling ("autizm", "probablity", etc.)
# 10. fuzzy/synonym question
# 11. multi-intent question (multiple questions in one message)
# 12. follow-up "why?"
# 13. follow-up "is that high?"
# 14. unknown / fallback question
# 15. question before upload
# 16. question after upload
# 17. long question
# 18. short question
# 19. medical-diagnosis question (disclaimer verification)
# 20. question referencing the current uploaded file
# 21. fail-safe robustness
# ---------------------------------------------------------------

import pytest
import numpy as np
from src.chatbot_context import InferenceContext, ConversationState
from src.chatbot_engine import ChatbotEngine
from src.chatbot_fuzzy import (
    levenshtein_distance,
    correct_token,
    correct_typos,
    normalize_text,
    tokenize,
)


@pytest.fixture
def empty_engine():
    """Engine without any uploaded file or inference results."""
    return ChatbotEngine(inference_context=InferenceContext())


@pytest.fixture
def asd_engine():
    """Engine with an actual ASD inference context."""
    ctx = InferenceContext(
        filename="Pitt_0050003_func_minimal.nii",
        extension=".nii",
        input_shape=(176, 200),
        processing_status="success",
        prediction=1,
        prediction_label="ASD",
        asd_probability=0.884,
        control_probability=0.116,
        selected_feature_count=600,
        total_edge_count=19900,
        n_rois=200,
        model_type="RBF-SVM",
        svm_c=10.0,
        svm_kernel="rbf",
        svm_gamma="scale",
        svm_class_weight="balanced",
        decision_function_value=1.425,
        feature_values=np.random.randn(1, 600),
    )
    return ChatbotEngine(inference_context=ctx)


@pytest.fixture
def con_engine():
    """Engine with an actual Typical Control inference context."""
    ctx = InferenceContext(
        filename="NYU_0050952_func_minimal.jpg",
        extension=".jpg",
        input_shape=(200, 200),
        processing_status="success",
        prediction=0,
        prediction_label="Control",
        asd_probability=0.082,
        control_probability=0.918,
        selected_feature_count=600,
        total_edge_count=19900,
        n_rois=200,
        model_type="RBF-SVM",
        svm_c=10.0,
        svm_kernel="rbf",
        svm_gamma="scale",
        svm_class_weight="balanced",
        decision_function_value=-1.682,
        feature_values=np.random.randn(1, 600),
    )
    return ChatbotEngine(inference_context=ctx)


# ---------------------------------------------------------------
# Test 1: Normal prediction question
# ---------------------------------------------------------------
def test_normal_prediction_question(asd_engine):
    resp = asd_engine.process_message("What did the model predict?")
    assert "ASD" in resp
    assert "classified" in resp.lower() or "prediction" in resp.lower()


# ---------------------------------------------------------------
# Test 2: ASD prediction context
# ---------------------------------------------------------------
def test_asd_prediction(asd_engine):
    resp = asd_engine.process_message("What is the prediction result?")
    assert "ASD" in resp
    assert "88.4%" in resp or "88%" in resp


# ---------------------------------------------------------------
# Test 3: Control prediction context
# ---------------------------------------------------------------
def test_control_prediction(con_engine):
    resp = con_engine.process_message("Is this ASD or Control?")
    assert "Control" in resp
    assert "91.8%" in resp or "92%" in resp


# ---------------------------------------------------------------
# Test 4: Probability question
# ---------------------------------------------------------------
def test_probability_question(asd_engine):
    resp = asd_engine.process_message("What is the ASD probability and confidence?")
    assert "88.40%" in resp or "88.4%" in resp
    assert "Control probability" in resp or "11.60%" in resp


# ---------------------------------------------------------------
# Test 5: Model question
# ---------------------------------------------------------------
def test_model_question(asd_engine):
    resp = asd_engine.process_message("What model was used for classification?")
    assert "Support Vector Machine" in resp or "SVM" in resp
    assert "RBF" in resp


# ---------------------------------------------------------------
# Test 6: Feature question
# ---------------------------------------------------------------
def test_feature_question(asd_engine):
    resp = asd_engine.process_message("How many features and edges are used?")
    assert "600" in resp
    assert "19,900" in resp or "19900" in resp


# ---------------------------------------------------------------
# Test 7: Pipeline question
# ---------------------------------------------------------------
def test_pipeline_question(asd_engine):
    resp = asd_engine.process_message("How does the system work from end to end?")
    assert "ROI" in resp or "CC200" in resp
    assert "functional connectivity" in resp.lower() or "correlation" in resp.lower()


# ---------------------------------------------------------------
# Test 8: Evaluation question
# ---------------------------------------------------------------
def test_evaluation_question(asd_engine):
    resp = asd_engine.process_message("What is accuracy, sensitivity, and specificity?")
    assert "Sensitivity" in resp
    assert "Specificity" in resp
    assert "Accuracy" in resp


# ---------------------------------------------------------------
# Test 9: Typo question
# ---------------------------------------------------------------
def test_typo_handling(asd_engine):
    resp = asd_engine.process_message("what is the probablity of autizm?")
    # Must correctly detect probability intent despite typos
    assert "88.4" in resp or "ASD probability" in resp


# ---------------------------------------------------------------
# Test 10: Fuzzy / synonym question
# ---------------------------------------------------------------
def test_fuzzy_synonym_question(asd_engine):
    resp = asd_engine.process_message("clasification outcom and functonal conectivity?")
    assert len(resp) > 20
    assert ("ASD" in resp) or ("functional connectivity" in resp.lower())


# ---------------------------------------------------------------
# Test 11: Multi-intent question
# ---------------------------------------------------------------
def test_multi_intent_question(asd_engine):
    resp = asd_engine.process_message(
        "Why did it classify this as ASD, what is the probability, and which model was used?"
    )
    # Must answer all three parts
    assert "ASD" in resp
    assert "88.4" in resp or "probability" in resp.lower()
    assert "SVM" in resp or "Support Vector Machine" in resp


# ---------------------------------------------------------------
# Test 12: Follow-up "why?"
# ---------------------------------------------------------------
def test_followup_why(asd_engine):
    # First ask prediction
    resp1 = asd_engine.process_message("What did it predict?")
    assert "ASD" in resp1

    # Follow up with "why?"
    resp2 = asd_engine.process_message("Why?")
    assert "decision boundary" in resp2.lower() or "influence" in resp2.lower() or "feature" in resp2.lower()


# ---------------------------------------------------------------
# Test 13: Follow-up "is that high?"
# ---------------------------------------------------------------
def test_followup_is_that_high(asd_engine):
    # First ask probability
    resp1 = asd_engine.process_message("What is the probability?")
    assert "88.4" in resp1

    # Follow up with "is that high?"
    resp2 = asd_engine.process_message("Is that high?")
    assert "high probability" in resp2.lower() or "confident" in resp2.lower()


# ---------------------------------------------------------------
# Test 14: Unknown / fallback question
# ---------------------------------------------------------------
def test_unknown_fallback(asd_engine):
    resp = asd_engine.process_message("quantum astrophysics black hole thermodynamics")
    assert "not entirely sure" in resp.lower() or "couldn't quite understand" in resp.lower() or "help" in resp.lower()


# ---------------------------------------------------------------
# Test 15: Question before upload
# ---------------------------------------------------------------
def test_question_before_upload(empty_engine):
    resp = empty_engine.process_message("What did it predict?")
    assert "upload" in resp.lower()
    assert "no file has been uploaded" in resp.lower() or "upload a scan" in resp.lower()


# ---------------------------------------------------------------
# Test 16: Question after upload
# ---------------------------------------------------------------
def test_question_after_upload(empty_engine):
    # Initially before upload
    resp1 = empty_engine.process_message("What is the prediction?")
    assert "upload" in resp1.lower()

    # Now upload happens
    empty_engine.set_inference_context(InferenceContext(
        filename="subject_001.nii",
        extension=".nii",
        processing_status="success",
        prediction=1,
        prediction_label="ASD",
        asd_probability=0.79,
        control_probability=0.21,
    ))
    resp2 = empty_engine.process_message("What is the prediction?")
    assert "ASD" in resp2


# ---------------------------------------------------------------
# Test 17: Long question
# ---------------------------------------------------------------
def test_long_question(asd_engine):
    long_q = (
        "Could you please explain to me in comprehensive detail how the CC200 brain atlas parcellates "
        "the resting-state functional MRI scan into 200 distinct regions of interest, and why Pearson "
        "correlation combined with Fisher r-to-z transformation is applied across all pairwise time series?"
    )
    resp = asd_engine.process_message(long_q)
    assert len(resp) > 50
    assert "CC200" in resp or "ROI" in resp or "connectivity" in resp.lower() or "Pearson" in resp


# ---------------------------------------------------------------
# Test 18: Short question
# ---------------------------------------------------------------
def test_short_question(asd_engine):
    resp = asd_engine.process_message("asd?")
    assert "ASD" in resp


# ---------------------------------------------------------------
# Test 19: Medical diagnosis question
# ---------------------------------------------------------------
def test_medical_diagnosis_disclaimer(asd_engine):
    resp = asd_engine.process_message("Does this mean I have autism? Can I use this for diagnosis?")
    assert "NOT a clinical diagnostic tool" in resp or "not constitute a medical diagnosis" in resp.lower()
    assert "healthcare professional" in resp.lower() or "doctor" in resp.lower()


# ---------------------------------------------------------------
# Test 20: Question referencing current uploaded file
# ---------------------------------------------------------------
def test_uploaded_file_reference(asd_engine):
    resp = asd_engine.process_message("What is the filename and format of the uploaded scan?")
    assert "Pitt_0050003_func_minimal.nii" in resp
    assert ".nii" in resp


# ---------------------------------------------------------------
# Test 21: Typo and Levenshtein unit checks
# ---------------------------------------------------------------
def test_fuzzy_functions():
    assert levenshtein_distance("autism", "autizm") == 1
    assert levenshtein_distance("probability", "probablity") == 1
    assert correct_token("autizm") == "autism"
    assert correct_token("probablity") == "probability"
    assert correct_token("predection") == "prediction"
    tokens = tokenize(normalize_text("What is the probablity of autizm?"))
    corrected = correct_typos(tokens)
    assert "probability" in corrected
    assert "autism" in corrected
