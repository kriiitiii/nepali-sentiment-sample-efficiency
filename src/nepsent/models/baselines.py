"""Majority-class and character n-gram TF-IDF + logistic regression baselines."""
from __future__ import annotations

import itertools

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nepsent.eval.metrics import macro_f1

DEFAULT_GRID = {
    "ngram_range": [(1, 3), (1, 4), (2, 5), (1, 6)],
    "analyzer": ["char_wb", "char"],
    "min_df": [1, 2],
    "C": [0.25, 1.0, 4.0, 16.0],
    "class_weight": [None, "balanced"],
}


def majority() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def char_lr(ngram_range=(1, 4), analyzer="char_wb", min_df=1, C=1.0, class_weight=None,
            max_features=None, sublinear_tf=True, seed=13) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
                                  max_features=max_features, sublinear_tf=sublinear_tf, lowercase=True)),
        ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=seed)),
    ])


def tune_char_lr(x_train, y_train, x_val, y_val, grid=None, seed=13):
    """Grid search scored on the validation set only.

    Returns (fitted model, best params, validation macro F1).
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    best = (None, None, -1.0)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        model = char_lr(seed=seed, **params).fit(x_train, y_train)
        score = macro_f1(y_val, model.predict(x_val))
        if score > best[2]:
            best = (model, params, score)
    return best
