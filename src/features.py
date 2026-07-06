"""Feature extraction for prompt-injection / jailbreak detection.

Turns a raw user prompt into a numeric feature vector that a classifier
can learn from. Features are intentionally interpretable so that a
Trust and Safety analyst can understand *why* a prompt was flagged,
rather than relying on an opaque score. This mirrors real red-teaming
workflows where explainability matters as much as accuracy.

The features combine two families:
1. Lexical / structural signals (length, capitalization, suspicious phrases)
2. TF-IDF character and word n-grams (captured separately in the pipeline)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# Phrases commonly seen in real jailbreak and prompt-injection attempts.
# These are defensive detection signals, not instructions for attacks.
SUSPICIOUS_PHRASES = [
    "ignore previous",
    "ignore the above",
    "disregard",
    "forget your instructions",
    "you are now",
    "pretend to be",
    "act as",
    "developer mode",
    "no restrictions",
    "without any filter",
    "do anything now",
    "jailbreak",
    "bypass",
    "override",
    "system prompt",
    "reveal your",
    "print your instructions",
]

ROLEPLAY_MARKERS = ["roleplay", "role play", "hypothetically", "in a fictional", "imagine you"]


@dataclass
class PromptFeatures:
    """Interpretable features extracted from a single prompt."""

    char_count: int
    word_count: int
    uppercase_ratio: float
    suspicious_phrase_count: int
    roleplay_marker_count: int
    imperative_ratio: float
    special_char_ratio: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.char_count,
            self.word_count,
            self.uppercase_ratio,
            self.suspicious_phrase_count,
            self.roleplay_marker_count,
            self.imperative_ratio,
            self.special_char_ratio,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "char_count",
            "word_count",
            "uppercase_ratio",
            "suspicious_phrase_count",
            "roleplay_marker_count",
            "imperative_ratio",
            "special_char_ratio",
        ]


IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "pretend", "act", "bypass",
    "override", "reveal", "print", "output", "write", "generate",
    "tell", "show", "give", "explain", "list",
}


def extract_features(prompt: str) -> PromptFeatures:
    """Compute interpretable features from a raw prompt string."""
    text = prompt.strip()
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    word_count = max(len(words), 1)
    char_count = max(len(text), 1)

    uppercase_ratio = sum(1 for c in text if c.isupper()) / char_count
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / char_count

    suspicious = sum(1 for phrase in SUSPICIOUS_PHRASES if phrase in lower)
    roleplay = sum(1 for marker in ROLEPLAY_MARKERS if marker in lower)

    first_words = [w for w in words[:3]]
    imperative_hits = sum(1 for w in first_words if w in IMPERATIVE_VERBS)
    imperative_ratio = imperative_hits / max(len(first_words), 1)

    return PromptFeatures(
        char_count=char_count,
        word_count=word_count,
        uppercase_ratio=round(uppercase_ratio, 4),
        suspicious_phrase_count=suspicious,
        roleplay_marker_count=roleplay,
        imperative_ratio=round(imperative_ratio, 4),
        special_char_ratio=round(special_char_ratio, 4),
    )


def explain(prompt: str) -> Dict[str, float]:
    """Return a human-readable dict of the features for a prompt."""
    feats = extract_features(prompt)
    return dict(zip(feats.feature_names(), feats.to_vector().tolist()))
