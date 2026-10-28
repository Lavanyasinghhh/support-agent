"""Intent classifier: TF-IDF features + logistic regression.

Trained on data/intents.csv (about 80 labelled examples). It returns the
predicted intent and its probability so the agent can fall back safely when
the model is unsure.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from .tools import ORDER_ID


def _normalise(text: str) -> str:
    # Replace order numbers with a token so the model learns "an order id is
    # present" rather than memorising specific ids.
    return ORDER_ID.sub(" ORDERID ", text.lower())


class IntentRouter:
    def __init__(self, intents_csv: Path):
        df = pd.read_csv(intents_csv)
        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ])
        self.model = make_pipeline(features, LogisticRegression(C=10, max_iter=2000))
        self.model.fit(df["text"].map(_normalise), df["intent"])
        self.labels = list(self.model.classes_)

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.model.predict_proba([_normalise(text)])[0]
        i = probs.argmax()
        return self.labels[i], float(probs[i])
