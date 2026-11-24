"""Inference helper for the trained binary sarcasm detector."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from sarcasm_sentiment import extract_sarcasm_features


MODEL_PATH = Path(__file__).resolve().parent / "trained_models" / "sarcasm_detector.joblib"


@lru_cache(maxsize=1)
def _load_detector():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict_sarcasm(text: str) -> dict:
    features = extract_sarcasm_features(text)
    has_positive_polarity_cue = features[0] > 0
    has_negative_polarity_cue = features[1] > 0

    detector = _load_detector()
    if detector is not None:
        vector = detector["vectorizer"].transform([text])
        classifier = detector["classifier"]
        probabilities = classifier.predict_proba(vector)[0]
        classes = list(classifier.classes_)
        sarcasm_index = classes.index(1)
        confidence = float(probabilities[sarcasm_index])
        is_sarcastic = bool(
            confidence >= 0.5
            or has_positive_polarity_cue
            or has_negative_polarity_cue
        )
        method = (
            "trained detector and polarity cues"
            if has_positive_polarity_cue or has_negative_polarity_cue
            else "trained detector"
        )
    else:
        is_sarcastic = bool(
            has_positive_polarity_cue or has_negative_polarity_cue
        )
        confidence = None
        method = "sarcasm cue rules (model not trained)"

    if has_positive_polarity_cue and has_negative_polarity_cue:
        sarcasm_type = "mixed_sarcasm"
    elif has_positive_polarity_cue:
        sarcasm_type = "positive_sarcasm"
    elif has_negative_polarity_cue:
        sarcasm_type = "negative_sarcasm"
    elif is_sarcastic:
        sarcasm_type = "sarcasm_detected"
    else:
        sarcasm_type = "genuine"

    return {
        "is_sarcastic": is_sarcastic,
        "confidence": confidence,
        "sarcasm_type": sarcasm_type,
        "method": method,
    }


def resolve_sentiment(raw_sentiment: str, sarcasm_result: dict) -> dict:
    polarity_by_sarcasm_type = {
        "positive_sarcasm": "positive",
        "negative_sarcasm": "negative",
    }
    sentiment = polarity_by_sarcasm_type.get(
        sarcasm_result["sarcasm_type"], raw_sentiment
    )
    return {"label": sentiment, "adjusted": sentiment != raw_sentiment}
