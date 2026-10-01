# src/chatbot_fuzzy.py
# ---------------------------------------------------------------
# Text normalisation, typo correction, and fuzzy matching for the
# ABIDE ASD chatbot.  Pure-Python – no external fuzzy library.
# ---------------------------------------------------------------

import re
from difflib import SequenceMatcher

# ----------------------------------------------------------------
# Levenshtein edit distance (dynamic programming, pure Python)
# ----------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

# ----------------------------------------------------------------
# Text normalisation
# ----------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation (keep alphanumeric/space/hyphen/dot),
    collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Split normalised text into tokens."""
    return text.split()

# ----------------------------------------------------------------
# Vocabulary & common misspellings
# ----------------------------------------------------------------

KNOWN_VOCABULARY = {
    # Core domain terms
    "autism", "asd", "control", "prediction", "probability", "confidence",
    "model", "svm", "classifier", "kernel", "rbf", "gamma", "balanced",
    "features", "edges", "selected", "connectivity", "functional",
    "correlation", "pearson", "fisher", "transformation", "preprocessing",
    "standardization", "scaler", "accuracy", "sensitivity", "specificity",
    "precision", "recall", "f1", "roc", "auc", "mcc", "confusion",
    "matrix", "false", "positive", "negative", "true", "roi", "region",
    "brain", "atlas", "cc200", "welch", "ttest", "diagnosis", "clinical",
    "limitations", "predict", "classify", "classification", "result",
    "percentage", "certain", "sure", "wrong", "error", "file", "upload",
    "uploaded", "image", "scan", "dimensions", "shape", "class", "label",
    "weights", "hyperparameter", "parameter", "training", "test",
    "evaluation", "performance", "metric", "metrics", "score",
    "support", "vector", "machine", "radial", "basis", "function",
    "decision", "boundary", "margin", "spectrum", "disorder",
    "typical", "atypical", "normal", "abnormal", "neurodevelopmental",
    "fmri", "neuroimaging", "abide", "dataset", "pipeline", "workflow",
    "step", "process", "processing", "clean", "cleaning", "clipping",
    "percentile", "edge", "feature", "selection", "extraction",
    "predicted", "classified", "chance", "chances", "high", "low",
    "explain", "why", "how", "what", "which", "does", "mean",
    "important", "influence", "contribute", "contribution", "significant",
    "probability", "probabilities", "confident", "certainty",
    "dense", "network", "deep", "learning", "fuzzy", "recurrent",
    "optimize", "optimization", "henry", "gas", "jellyfish",
    "autoencoder", "sparse", "gru",
}

COMMON_MISSPELLINGS: dict = {
    # autism variants
    "autizm": "autism", "autsim": "autism", "autismm": "autism",
    "autisim": "autism", "autisam": "autism", "autisum": "autism",
    "autisem": "autism", "autistic": "autism", "autisitc": "autism",
    "autims": "autism", "austim": "autism", "autusm": "autism",
    # probability
    "probablity": "probability", "probabilty": "probability",
    "probabiilty": "probability", "probibility": "probability",
    "probality": "probability", "probbability": "probability",
    "probabiliy": "probability", "probabiltiy": "probability",
    # prediction
    "predection": "prediction", "predction": "prediction",
    "predicton": "prediction", "prediciton": "prediction",
    "preidction": "prediction", "perdiction": "prediction",
    # classification
    "clasification": "classification", "classificaton": "classification",
    "classifcation": "classification", "classificiation": "classification",
    "classfication": "classification", "classifiation": "classification",
    # functional
    "functonal": "functional", "funtional": "functional",
    "funcional": "functional", "fucntional": "functional",
    # connectivity
    "conectivity": "connectivity", "connectvity": "connectivity",
    "connnectivity": "connectivity", "conectivty": "connectivity",
    "conncetivity": "connectivity", "connectivty": "connectivity",
    # standard / scaler
    "standrd": "standard", "standared": "standard", "stanard": "standard",
    "scaller": "scaler", "scalr": "scaler", "scalrer": "scaler",
    # machine
    "machne": "machine", "machien": "machine", "machin": "machine",
    "mahcine": "machine",
    # confusion
    "confuson": "confusion", "confuison": "confusion",
    "confusin": "confusion", "confusoin": "confusion",
    # matrix
    "matirx": "matrix", "matrx": "matrix", "matix": "matrix",
    # sensitivity / specificity
    "sensitvity": "sensitivity", "sensitivty": "sensitivity",
    "sensitiviy": "sensitivity",
    "specifity": "specificity", "specificty": "specificity",
    "specficity": "specificity",
    # accuracy
    "acuracy": "accuracy", "accurracy": "accuracy",
    "accurcy": "accuracy", "accurasy": "accuracy", "accuarcy": "accuracy",
    # features
    "faetures": "features", "featurs": "features", "fetures": "features",
    "featuers": "features",
    # preprocessing
    "preprocesing": "preprocessing", "preprocessng": "preprocessing",
    "preproccessing": "preprocessing", "preproessing": "preprocessing",
    # evaluation
    "evalution": "evaluation", "evaluaton": "evaluation",
    "evalutaion": "evaluation",
    # diagnosis
    "diagnsis": "diagnosis", "diagnsois": "diagnosis",
    "diagosis": "diagnosis", "diagnosos": "diagnosis",
    "diagnoisis": "diagnosis",
    # limitations
    "limitaions": "limitations", "limitatons": "limitations",
    "limitaitons": "limitations",
    # transformation
    "trasformation": "transformation", "transformaton": "transformation",
    "transfromation": "transformation",
    # correlation
    "correlaton": "correlation", "correlatoin": "correlation",
    "correltion": "correlation", "correaltion": "correlation",
    # support / vector
    "suport": "support", "supprt": "support", "supoort": "support",
    "vectr": "vector", "vctor": "vector", "vecotr": "vector",
    # percentage
    "percentge": "percentage", "percantage": "percentage",
    "percetage": "percentage",
    # confidence
    "confidance": "confidence", "confidnce": "confidence",
    "confindence": "confidence",
    # kernel
    "kernal": "kernel", "kernl": "kernel", "kernerl": "kernel",
    # specifics
    "hyperparamter": "hyperparameter", "hyperparamater": "hyperparameter",
    "paramter": "parameter", "paramater": "parameter",
    "performace": "performance", "performnce": "performance",
}

COMMON_ENGLISH_WORDS = {
    "that", "this", "it", "its", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "doing",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "mine", "yours", "ours", "theirs",
    "the", "a", "an", "and", "or", "but", "if", "so", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "with", "about", "above", "after",
    "before", "between", "both", "down", "during", "each", "few", "more", "most",
    "other", "some", "such", "than", "then", "through", "up", "very",
    "good", "bad", "yes", "no", "not", "tell", "give", "show", "please", "like",
    "also", "any", "all", "just", "now", "well", "there", "here", "where", "when",
    "who", "whom", "whose", "which", "why", "how", "what", "high", "low", "much",
    "many", "get", "got", "say", "said", "see", "saw", "know", "known",
}

# ----------------------------------------------------------------
# Token-level correction
# ----------------------------------------------------------------

def correct_token(token: str) -> str:
    """Correct a single token via direct map then fuzzy match."""
    if token in KNOWN_VOCABULARY or token in COMMON_ENGLISH_WORDS:
        return token
    if token in COMMON_MISSPELLINGS:
        return COMMON_MISSPELLINGS[token]

    best_match = None
    best_dist = float("inf")
    for word in KNOWN_VOCABULARY:
        if abs(len(token) - len(word)) > 3:
            continue
        dist = levenshtein_distance(token, word)
        if dist < best_dist:
            best_dist = dist
            best_match = word

    max_allowed = min(2, max(1, len(token) // 3))
    if best_match and best_dist <= max_allowed:
        return best_match
    return token


def correct_typos(tokens: list) -> list:
    """Correct every token in a list."""
    return [correct_token(t) for t in tokens]

# ----------------------------------------------------------------
# Phrase-level fuzzy matching
# ----------------------------------------------------------------

def fuzzy_phrase_score(text: str, phrase: str) -> float:
    """SequenceMatcher ratio between *text* and *phrase*."""
    return SequenceMatcher(None, text, phrase).ratio()


def best_phrase_match(text: str, phrases: list, threshold: float = 0.55):
    """Return (phrase, score) for the best match above *threshold*,
    or (None, 0.0)."""
    best, best_s = None, 0.0
    for p in phrases:
        s = fuzzy_phrase_score(text, p)
        if s > best_s:
            best_s = s
            best = p
    return (best, best_s) if best_s >= threshold else (None, 0.0)


def tokens_contain_any(tokens: list, keywords: set) -> bool:
    """Return True if *tokens* contains any word from *keywords*."""
    return bool(set(tokens) & keywords)


def text_contains_phrase(text: str, phrase: str) -> bool:
    """Check whether *phrase* appears as a substring of *text*."""
    return phrase in text
