"""Jailbreak / prompt-injection classifier.

Combines TF-IDF text features with the interpretable structural features
from features.py, and trains a logistic regression classifier. Logistic
regression is chosen deliberately: it is fast, robust on small datasets,
and its coefficients are interpretable, which matters for a Trust and
Safety setting where analysts need to justify why a prompt was flagged.

The classifier exposes predict_proba so downstream systems can set their
own decision threshold depending on how conservative they need to be.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .features import extract_features


class JailbreakClassifier:
    """Hybrid TF-IDF + structural-feature jailbreak detector."""

    def __init__(self, C: float = 1.0):
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=5000
        )
        self.model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        self._fitted = False

    def _structural_matrix(self, prompts: List[str]) -> csr_matrix:
        rows = [extract_features(p).to_vector() for p in prompts]
        return csr_matrix(np.vstack(rows))

    def _transform(self, prompts: List[str], fit: bool = False) -> csr_matrix:
        if fit:
            word = self.word_vectorizer.fit_transform(prompts)
            char = self.char_vectorizer.fit_transform(prompts)
        else:
            word = self.word_vectorizer.transform(prompts)
            char = self.char_vectorizer.transform(prompts)
        struct = self._structural_matrix(prompts)
        return hstack([word, char, struct]).tocsr()

    def fit(self, prompts: List[str], labels: List[int]) -> "JailbreakClassifier":
        X = self._transform(prompts, fit=True)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, prompts: List[str]) -> np.ndarray:
        return (self.predict_proba(prompts) >= 0.5).astype(int)

    def predict_proba(self, prompts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        X = self._transform(prompts, fit=False)
        return self.model.predict_proba(X)[:, 1]

    def predict_one(self, prompt: str) -> Tuple[int, float]:
        """Return (label, probability) for a single prompt."""
        prob = float(self.predict_proba([prompt])[0])
        return int(prob >= 0.5), prob
