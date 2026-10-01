# src/chatbot_response.py
# ---------------------------------------------------------------
# Response template library and composition engine.
# Selects varied templates, inserts actual runtime values, and
# composes multi-intent answers.  No LLM, no API.
# ---------------------------------------------------------------

from __future__ import annotations
import hashlib
from typing import List, Optional

from src.chatbot_context import InferenceContext, ConversationState
from src.chatbot_explanation import (
    explain_prediction,
    explain_probability,
    explain_probability_qualifier,
    explain_model,
    explain_model_params,
    explain_features,
    explain_pipeline,
    explain_roi,
    explain_fc,
    explain_preprocessing,
    explain_ttest,
    explain_evaluation,
    explain_confusion_matrix,
    explain_limitations,
    explain_medical_disclaimer,
    explain_file_info,
)


# ================================================================
# Template selection helper
# ================================================================

def _pick_template(templates: list, seed_text: str) -> str:
    """Deterministically pick a template variation based on message hash."""
    idx = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % len(templates)
    return templates[idx]


# ================================================================
# Greeting / Help templates
# ================================================================

_GREETING_TEMPLATES = [
    "Hello! I'm the ABIDE ASD Assistant. I can answer questions about "
    "the current prediction, model details, features, preprocessing, "
    "and more. How can I help you?",

    "Hi there! I'm here to help you understand the ASD detection "
    "results. Ask me about the prediction, probability, model, "
    "features, or the processing pipeline.",

    "Hey! Welcome to the ABIDE ASD Assistant. Feel free to ask me "
    "about the classification results, how the model works, feature "
    "selection, or anything else about this system.",
]

_HELP_TEMPLATES = [
    "Here are some things you can ask me about:\n\n"
    "🔹 **Prediction**: \"What did the model predict?\" or \"Is this ASD?\"\n"
    "🔹 **Probability**: \"What is the ASD probability?\" or \"How confident is the model?\"\n"
    "🔹 **Why**: \"Why did it predict ASD?\" or \"Explain the result\"\n"
    "🔹 **Model**: \"What model is used?\" or \"What is RBF-SVM?\"\n"
    "🔹 **Parameters**: \"What is C=10?\" or \"What is gamma?\"\n"
    "🔹 **Features**: \"How many features?\" or \"Why 600 edges?\"\n"
    "🔹 **Pipeline**: \"How does the system work?\"\n"
    "🔹 **ROI / Atlas**: \"What is CC200?\" or \"What is an ROI?\"\n"
    "🔹 **Connectivity**: \"What is functional connectivity?\"\n"
    "🔹 **Preprocessing**: \"What preprocessing is done?\"\n"
    "🔹 **Evaluation**: \"What is accuracy?\" or \"What is F1-Score?\"\n"
    "🔹 **Limitations**: \"Can the model be wrong?\"\n"
    "🔹 **Medical**: \"Is this a clinical diagnosis?\"\n"
    "🔹 **File info**: \"What file was uploaded?\"\n\n"
    "You can also ask follow-up questions like \"Why?\" or \"Is that high?\"",
]

_NO_UPLOAD_TEMPLATES = [
    "No file has been uploaded yet. Please upload a brain scan file "
    "(.nii, .jpg, or .png) using the uploader above, and I'll be able "
    "to answer questions about the prediction results.",

    "I don't have any prediction data to work with yet. Upload a scan "
    "file first, and then I can help you understand the results.",

    "Please upload a subject scan file first. Once the system processes "
    "it, I can answer questions about the prediction, probabilities, "
    "and model reasoning.",
]

_FALLBACK_TEMPLATES = [
    "I'm not entirely sure what you're asking. You can ask me about "
    "the current prediction, probability, model reasoning, features, "
    "preprocessing, functional connectivity, evaluation metrics, or "
    "the project workflow. Type \"help\" for a full list.",

    "I couldn't quite understand that question. Try asking about the "
    "prediction result, ASD probability, model details, feature "
    "selection, or the processing pipeline. Say \"help\" to see all "
    "available topics.",

    "Sorry, I'm not sure how to answer that. I can help with questions "
    "about predictions, probabilities, the SVM model, features, "
    "preprocessing steps, or limitations. Type \"help\" for guidance.",
]


# ================================================================
# Intent → response dispatcher
# ================================================================

def generate_response_for_intent(
    intent: str,
    ctx: InferenceContext,
    conv: ConversationState,
    raw_text: str,
) -> str:
    """Generate a response for a single resolved intent."""

    # Intents that require a prediction
    prediction_required = {
        "PREDICTION", "PROBABILITY", "WHY_PREDICTION",
        "PROBABILITY_QUALIFIER",
    }

    if intent in prediction_required and not ctx.has_prediction():
        return _pick_template(_NO_UPLOAD_TEMPLATES, raw_text)

    # Dispatch table
    if intent == "PREDICTION":
        response = explain_prediction(ctx)
        conv.update_context(intent, topic="prediction",
                            value=ctx.prediction_label, entity="prediction")
        return response

    if intent == "PROBABILITY":
        response = explain_probability(ctx)
        conv.update_context(intent, topic="probability",
                            value=ctx.asd_probability, entity="probability")
        return response

    if intent == "PROBABILITY_QUALIFIER":
        response = explain_probability_qualifier(ctx)
        conv.update_context(intent, topic="probability",
                            value=ctx.asd_probability, entity="probability")
        return response

    if intent == "WHY_PREDICTION":
        response = explain_prediction(ctx)
        conv.update_context(intent, topic="prediction",
                            value=ctx.prediction_label, entity="prediction")
        return response

    if intent == "MODEL_INFO":
        response = explain_model(ctx)
        conv.update_context(intent, topic="model", entity="model")
        return response

    if intent == "MODEL_PARAMS":
        response = explain_model_params(ctx)
        conv.update_context(intent, topic="model_params", entity="model")
        return response

    if intent == "FEATURES":
        response = explain_features(ctx)
        conv.update_context(intent, topic="features", entity="features")
        return response

    if intent == "PIPELINE":
        response = explain_pipeline(ctx)
        conv.update_context(intent, topic="pipeline", entity="pipeline")
        return response

    if intent == "ROI_INFO":
        response = explain_roi(ctx)
        conv.update_context(intent, topic="roi", entity="roi")
        return response

    if intent == "FC_INFO":
        response = explain_fc(ctx)
        conv.update_context(intent, topic="connectivity", entity="connectivity")
        return response

    if intent == "PREPROCESSING":
        response = explain_preprocessing(ctx)
        conv.update_context(intent, topic="preprocessing", entity="preprocessing")
        return response

    if intent == "TTEST_INFO":
        response = explain_ttest(ctx)
        conv.update_context(intent, topic="ttest", entity="features")
        return response

    if intent == "EVALUATION":
        response = explain_evaluation(ctx)
        conv.update_context(intent, topic="evaluation", entity="evaluation")
        return response

    if intent == "CONFUSION_MATRIX":
        response = explain_confusion_matrix(ctx)
        conv.update_context(intent, topic="confusion_matrix", entity="evaluation")
        return response

    if intent == "LIMITATIONS":
        response = explain_limitations(ctx)
        conv.update_context(intent, topic="limitations", entity="limitations")
        return response

    if intent == "MEDICAL_DISCLAIMER":
        response = explain_medical_disclaimer(ctx)
        conv.update_context(intent, topic="medical", entity="medical")
        return response

    if intent == "FILE_INFO":
        response = explain_file_info(ctx)
        conv.update_context(intent, topic="file", entity="file")
        return response

    if intent == "GREETING":
        return _pick_template(_GREETING_TEMPLATES, raw_text)

    if intent == "HELP":
        return _HELP_TEMPLATES[0]

    return _pick_template(_FALLBACK_TEMPLATES, raw_text)


# ================================================================
# Multi-intent composition
# ================================================================

def compose_multi_response(
    intents: List[str],
    ctx: InferenceContext,
    conv: ConversationState,
    raw_text: str,
) -> str:
    """Generate and combine responses for multiple intents."""
    if not intents:
        return _pick_template(_FALLBACK_TEMPLATES, raw_text)

    if len(intents) == 1:
        return generate_response_for_intent(intents[0], ctx, conv, raw_text)

    # Multiple intents — compose with separators
    parts = []
    for i, intent in enumerate(intents):
        resp = generate_response_for_intent(intent, ctx, conv, raw_text)
        if i > 0:
            parts.append("---")  # visual separator
        parts.append(resp)

    return "\n\n".join(parts)


def generate_fallback(raw_text: str) -> str:
    """Generate a fallback response when no intent matched."""
    return _pick_template(_FALLBACK_TEMPLATES, raw_text)
