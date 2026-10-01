# src/chatbot_engine.py
# ---------------------------------------------------------------
# Chatbot Orchestrator Engine.
# Implements the full NLP pipeline without external APIs/LLMs:
# Normalization -> Typo correction -> Intent detection ->
# Multi-intent resolution -> Context resolution -> Explanation engine
# -> Response composition.
# ---------------------------------------------------------------

from typing import List, Optional
from src.chatbot_context import InferenceContext, ConversationState
from src.chatbot_fuzzy import normalize_text, tokenize, correct_typos
from src.chatbot_intents import (
    detect_intents,
    detect_multi_intents,
    resolve_followup,
    is_followup,
    IntentMatch,
)
from src.chatbot_response import (
    compose_multi_response,
    generate_response_for_intent,
    generate_fallback,
)


class ChatbotEngine:
    """Deterministic, rule-based chatbot engine for ABIDE ASD Detection."""

    def __init__(self, inference_context: Optional[InferenceContext] = None):
        self.inference_context = inference_context or InferenceContext()
        self.conversation_state = ConversationState()

    def set_inference_context(self, context: InferenceContext):
        """Update runtime inference context from the frontend."""
        self.inference_context = context

    def reset_conversation(self):
        """Reset conversation history and state."""
        self.conversation_state.clear()

    def process_message(self, user_message: str) -> str:
        """Process a user message through the full modular pipeline and return the answer."""
        if not user_message or not user_message.strip():
            return "Please type a question or request, and I'll be glad to help."

        raw_text = user_message.strip()

        # Step 1: Text Normalization
        normalized = normalize_text(raw_text)

        # Step 2: Tokenization
        raw_tokens = tokenize(normalized)

        # Step 3: Typo Correction & Fuzzy vocabulary mapping
        corrected_tokens = correct_typos(raw_tokens)
        corrected_text = " ".join(corrected_tokens)

        # Step 4: Check for Follow-Up context resolution
        followup_intent = resolve_followup(
            corrected_tokens, corrected_text, self.conversation_state
        )

        if followup_intent:
            response = generate_response_for_intent(
                followup_intent,
                self.inference_context,
                self.conversation_state,
                raw_text,
            )
            self.conversation_state.add_message("user", raw_text)
            self.conversation_state.add_message("assistant", response)
            return response

        # Step 5: Multi-intent & standard intent detection with scoring
        matches: List[IntentMatch] = detect_multi_intents(
            raw_text=raw_text,
            corrected_tokens=corrected_tokens,
            corrected_text=corrected_text,
            threshold=0.20,
        )

        # Step 6: Filter and select top distinct intents
        if not matches:
            response = generate_fallback(raw_text)
            self.conversation_state.add_message("user", raw_text)
            self.conversation_state.add_message("assistant", response)
            return response

        # Pick confident intents (score >= 0.22, or best match if highest is lower)
        top_score = matches[0].score
        selected_intents: List[str] = []

        for m in matches:
            # If multiple intents, require reasonable confidence relative to top score
            if m.score >= max(0.25, top_score * 0.65):
                if m.intent not in selected_intents:
                    selected_intents.append(m.intent)

        # Limit to top 3 intents to avoid overwhelming responses
        selected_intents = selected_intents[:3]

        if not selected_intents:
            selected_intents = [matches[0].intent]

        # Step 7: Response composition using explanation engine & templates
        response = compose_multi_response(
            intents=selected_intents,
            ctx=self.inference_context,
            conv=self.conversation_state,
            raw_text=raw_text,
        )

        # Update history
        self.conversation_state.add_message("user", raw_text)
        self.conversation_state.add_message("assistant", response)

        return response
