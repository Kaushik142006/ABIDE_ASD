# src/chatbot_intents.py
# ---------------------------------------------------------------
# Intent detection, multi-intent resolution, and follow-up
# resolution.  Uses keyword matching, fuzzy phrase scoring with
# content-word filtering, and conversation context – no LLM, no API.
# ---------------------------------------------------------------

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

from src.chatbot_fuzzy import (
    normalize_text, tokenize, correct_typos,
    fuzzy_phrase_score, tokens_contain_any, text_contains_phrase,
)
from src.chatbot_rules import (
    INTENT_PATTERNS, SYNONYM_MAP,
    FOLLOWUP_TRIGGERS, FOLLOWUP_REFERENCE_WORDS,
    FOLLOWUP_QUALIFIER_PHRASES,
)
from src.chatbot_context import ConversationState



@dataclass
class IntentMatch:
    """A scored intent candidate."""
    intent: str
    score: float
    matched_keywords: list = field(default_factory=list)
    matched_phrase: Optional[str] = None


STOP_WORDS = {
    "what", "is", "the", "a", "an", "does", "did", "do", "how", "why",
    "are", "of", "in", "to", "for", "it", "this", "that", "can", "be",
    "tell", "me", "about", "show", "give", "was", "were", "which",
}

_SPLIT_RE = re.compile(r"\band\b|,\s*|\?\s*(?=[a-z])", re.IGNORECASE)


def _keyword_score(tokens: list, keywords: set) -> Tuple[float, list]:
    """Return (score, matched_words) for keyword overlap."""
    matched = [t for t in tokens if t in keywords]
    if not matched:
        return 0.0, []
    return len(matched) / max(len(tokens), 1), matched


def _phrase_score(text: str, tokens: list, phrases: list) -> Tuple[float, Optional[str]]:
    """Return (best_score, best_phrase) via substring check + content-aware fuzzy window."""
    best_s, best_p = 0.0, None

    for phrase in phrases:
        # Exact substring gets top score
        if phrase in text:
            return 1.0, phrase

        p_words = [w for w in phrase.split() if w]
        content_words = [w for w in p_words if w not in STOP_WORDS]

        if content_words:
            has_content = False
            for cw in content_words:
                for t in tokens:
                    if t == cw or (len(cw) > 3 and SequenceMatcher(None, t, cw).ratio() >= 0.82):
                        has_content = True
                        break
                if has_content:
                    break
            if not has_content:
                continue

        p_len = len(p_words)
        for w_len in (p_len - 1, p_len, p_len + 1):
            if w_len <= 0 or w_len > len(tokens):
                continue
            for i in range(len(tokens) - w_len + 1):
                window = " ".join(tokens[i:i + w_len])
                r = SequenceMatcher(None, window, phrase).ratio()
                if r > best_s:
                    best_s = r
                    best_p = phrase

    if best_s >= 0.78:
        return best_s, best_p
    return 0.0, None


def detect_intents(
    raw_text: str,
    corrected_tokens: list,
    corrected_text: str,
    threshold: float = 0.20,
) -> List[IntentMatch]:
    """Score every intent and return those above *threshold*,
    sorted descending by score."""

    results: List[IntentMatch] = []

    for intent_name, patterns in INTENT_PATTERNS.items():
        kw_score, kw_matched = _keyword_score(
            corrected_tokens, patterns["keywords"]
        )
        ph_score, ph_matched = _phrase_score(
            corrected_text, corrected_tokens, patterns["phrases"]
        )
        weight = patterns.get("weight", 1.0)

        # If neither keyword nor content phrase matched, skip
        if kw_score == 0 and ph_score == 0:
            continue

        if ph_score >= 0.85:
            combined = (0.2 * kw_score + 0.8 * ph_score) * weight
        elif ph_score > 0:
            combined = (0.4 * kw_score + 0.6 * ph_score) * weight
        else:
            combined = (kw_score * 0.8) * weight

        if combined >= threshold:
            results.append(IntentMatch(
                intent=intent_name,
                score=combined,
                matched_keywords=kw_matched,
                matched_phrase=ph_matched,
            ))

    results.sort(key=lambda m: m.score, reverse=True)
    return results


# ================================================================
# Multi-intent splitting
# ================================================================

def split_multi_intent(text: str) -> List[str]:
    """Split a message that may contain several questions joined by
    'and', commas, or question-marks into sub-questions."""
    parts = _SPLIT_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [text]
    return [p for p in parts if len(p.split()) >= 2] or [text]


def detect_multi_intents(
    raw_text: str,
    corrected_tokens: list,
    corrected_text: str,
    threshold: float = 0.20,
) -> List[IntentMatch]:
    """Detect intents across the whole message, then also across
    sub-questions if multiple questions are present."""

    all_matches: dict[str, IntentMatch] = {}

    # Whole-message detection
    for m in detect_intents(raw_text, corrected_tokens, corrected_text, threshold):
        if m.intent not in all_matches or m.score > all_matches[m.intent].score:
            all_matches[m.intent] = m

    # Sub-question detection
    sub_qs = split_multi_intent(corrected_text)
    if len(sub_qs) > 1:
        for sq in sub_qs:
            sq_tokens = tokenize(sq)
            for m in detect_intents(sq, sq_tokens, sq, threshold):
                if m.intent not in all_matches or m.score > all_matches[m.intent].score:
                    all_matches[m.intent] = m

    ranked = sorted(all_matches.values(), key=lambda m: m.score, reverse=True)
    return ranked


# ================================================================
# Follow-up / context resolution
# ================================================================

def is_followup(tokens: list, text: str) -> bool:
    """Return True if the message looks like a short follow-up."""
    if len(tokens) <= 3 and tokens_contain_any(tokens, FOLLOWUP_TRIGGERS):
        return True
    if len(tokens) <= 5 and tokens_contain_any(tokens, FOLLOWUP_REFERENCE_WORDS):
        for phrase in FOLLOWUP_QUALIFIER_PHRASES:
            if phrase in text:
                return True
    if len(tokens) == 1 and tokens[0] in FOLLOWUP_TRIGGERS:
        return True
    return False


def resolve_followup(
    tokens: list,
    text: str,
    conv_state: ConversationState,
) -> Optional[str]:
    """If the message is a follow-up, return the implied intent based
    on conversation context. Otherwise return None."""
    if not is_followup(tokens, text):
        return None

    last = conv_state.last_intent

    # Check qualifier phrases first (e.g., "is that high?")
    for phrase in FOLLOWUP_QUALIFIER_PHRASES:
        if phrase in text:
            if last in ("PROBABILITY", "PROBABILITY_QUALIFIER"):
                return "PROBABILITY_QUALIFIER"
            elif last == "PREDICTION":
                return "PREDICTION"
            elif last == "EVALUATION":
                return "EVALUATION"
            return last

    # "why?" after a prediction → WHY_PREDICTION
    if tokens_contain_any(tokens, {"why", "how", "explain", "reason"}):
        if last in ("PREDICTION", "WHY_PREDICTION"):
            return "WHY_PREDICTION"
        if last in ("PROBABILITY", "PROBABILITY_QUALIFIER"):
            return "WHY_PREDICTION"
        if last == "MODEL_INFO":
            return "MODEL_PARAMS"
        if last:
            return "WHY_PREDICTION"

    # "more" / "details" / "elaborate" → repeat last intent
    if tokens_contain_any(tokens, {"more", "details", "elaborate", "detail"}):
        return last

    return last
