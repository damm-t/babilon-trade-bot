"""Tests for sentiment analysis model."""

import pytest
from unittest.mock import patch, MagicMock
from model.sentiment_model import analyze_sentiment


class TestSentimentModel:
    """Test sentiment analysis functionality."""

    @patch('model.sentiment_model.pipeline')
    def test_positive_sentiment(self, mock_pipeline):
        """Test positive sentiment analysis."""
        # Mock the pipeline to return positive sentiment
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'POSITIVE', 'score': 0.85}]
        mock_pipeline.return_value = mock_classifier
        
        sentiment, score = analyze_sentiment("Apple reports record quarterly revenue")
        
        assert sentiment == "POSITIVE"
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch('model.sentiment_model.pipeline')
    def test_negative_sentiment(self, mock_pipeline):
        """Test negative sentiment analysis."""
        # Mock the pipeline to return negative sentiment
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'NEGATIVE', 'score': 0.75}]
        mock_pipeline.return_value = mock_classifier
        
        sentiment, score = analyze_sentiment("Company faces major financial crisis")
        
        assert sentiment == "NEGATIVE"
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch('model.sentiment_model.pipeline')
    def test_neutral_sentiment(self, mock_pipeline):
        """Test neutral sentiment analysis."""
        # Mock the pipeline to return neutral sentiment
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'NEUTRAL', 'score': 0.60}]
        mock_pipeline.return_value = mock_classifier
        
        sentiment, score = analyze_sentiment("Company releases quarterly report")
        
        assert sentiment == "NEUTRAL"
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch('model.sentiment_model.pipeline')
    def test_empty_text(self, mock_pipeline):
        """Test handling of empty text."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'NEUTRAL', 'score': 0.5}]
        mock_pipeline.return_value = mock_classifier
        
        sentiment, score = analyze_sentiment("")
        
        assert sentiment == "NEUTRAL"
        assert isinstance(score, float)

    @patch('model.sentiment_model.pipeline')
    def test_model_loading_error(self, mock_pipeline):
        """Test handling of model loading errors."""
        mock_pipeline.side_effect = Exception("Model loading failed")
        
        # Should fall back to neutral sentiment
        sentiment, score = analyze_sentiment("Test text")
        
        assert sentiment == "NEUTRAL"
        assert score == 0.5

    @patch('model.sentiment_model.pipeline')
    def test_long_text(self, mock_pipeline):
        """Test handling of long text."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{'label': 'POSITIVE', 'score': 0.80}]
        mock_pipeline.return_value = mock_classifier
        
        long_text = "Apple " * 1000  # Very long text
        sentiment, score = analyze_sentiment(long_text)
        
        assert sentiment == "POSITIVE"
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
