# src/chatbot_context.py
# ---------------------------------------------------------------
# Context management: inference context (actual runtime values) and
# lightweight conversation state for follow-up resolution.
# ---------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional, Any, List


@dataclass
class InferenceContext:
    """Actual runtime inference data produced by the existing ML pipeline.

    Every field must be populated from *real* values – never hardcoded.
    """
    # --- uploaded file info ---
    filename: Optional[str] = None
    extension: Optional[str] = None
    input_shape: Optional[tuple] = None
    processing_status: str = "no_upload"      # "no_upload" | "success" | "error"
    processing_error: Optional[str] = None

    # --- prediction results ---
    prediction: Optional[int] = None           # 0 or 1
    prediction_label: Optional[str] = None     # "ASD" or "Control"
    asd_probability: Optional[float] = None
    control_probability: Optional[float] = None

    # --- feature / model metadata ---
    selected_feature_count: int = 0
    total_edge_count: int = 19_900             # 200*(200-1)/2
    n_rois: int = 200

    model_type: str = "RBF-SVM"
    svm_c: float = 10.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    svm_class_weight: str = "balanced"

    # --- post-hoc explanation helpers ---
    decision_function_value: Optional[float] = None
    feature_values: Optional[Any] = None       # scaled feature vector (1, K)

    # convenience predicates
    def has_prediction(self) -> bool:
        return self.prediction is not None

    def has_upload(self) -> bool:
        return self.processing_status != "no_upload"


@dataclass
class ConversationMessage:
    role: str       # "user" | "assistant"
    content: str


class ConversationState:
    """Lightweight conversation context for follow-up resolution."""

    def __init__(self):
        self.history: List[ConversationMessage] = []
        self.last_intent: Optional[str] = None
        self.last_topic: Optional[str] = None
        self.last_value: Optional[Any] = None
        self.last_entity: Optional[str] = None

    # ------ mutation helpers ------

    def add_message(self, role: str, content: str):
        self.history.append(ConversationMessage(role=role, content=content))

    def update_context(
        self,
        intent: str,
        topic: Optional[str] = None,
        value: Any = None,
        entity: Optional[str] = None,
    ):
        self.last_intent = intent
        if topic is not None:
            self.last_topic = topic
        if value is not None:
            self.last_value = value
        if entity is not None:
            self.last_entity = entity

    def get_recent_messages(self, n: int = 6) -> List[ConversationMessage]:
        return self.history[-n:] if self.history else []

    def clear(self):
        self.history.clear()
        self.last_intent = None
        self.last_topic = None
        self.last_value = None
        self.last_entity = None
