
import logging

from transformers import pipeline

# Load FinBERT sentiment analysis pipeline from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']
