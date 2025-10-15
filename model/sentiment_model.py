
import logging
from functools import lru_cache

try:
    from transformers import pipeline
except ImportError:
    # transformers not installed, create dummy function
    def pipeline(*args, **kwargs):
        raise ImportError("transformers not installed. Please install it with: pip install transformers torch")


@lru_cache(maxsize=1)
def _get_analyzer():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_sentiment(text):
    try:
        analyzer = _get_analyzer()
        result = analyzer(text)
        label = str(result[0].get("label", "neutral")).lower()
        score = float(result[0].get("score", 0.5))
        # Map labels to positive/negative/neutral
        if label.startswith("pos"):
            return "positive", score
        if label.startswith("neg"):
            return "negative", score
        return "neutral", 1.0 - abs(score - 0.5) * 2
    except Exception as e:
        logging.warning(f"Sentiment analysis failed: {e}")
        return "neutral", 0.5
