"""Test ML-based text moderation services."""
import pytest
from text_moderation.services.toxicity_detector import ToxicityDetector
from text_moderation.services.sentiment_analyzer import SentimentAnalyzer
from text_moderation.services.profanity_filter import ProfanityFilter


class TestToxicityDetector:
    """Test ML-based toxicity detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ToxicityDetector()
    
    def test_detect_toxicity_clean_text(self):
        """Test toxicity detection on clean text."""
        is_toxic, confidence, categories = self.detector.detect_toxicity("This is a nice day")
        
        assert isinstance(is_toxic, bool)
        assert isinstance(confidence, float)
        assert isinstance(categories, list)
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_toxicity_empty_text(self):
        """Test toxicity detection on empty text."""
        is_toxic, confidence, categories = self.detector.detect_toxicity("")
        
        assert is_toxic is False
        assert confidence == 0.0
        assert categories == []
    
    def test_get_severity_score(self):
        """Test severity score calculation."""
        score = self.detector.get_severity_score(["toxicity", "hate_speech"])
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestSentimentAnalyzer:
    """Test ML-based sentiment analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis on positive text."""
        sentiment, confidence, score = self.analyzer.analyze_sentiment("I love this movie")
        
        assert sentiment in ["positive", "negative", "neutral"]
        assert isinstance(confidence, float)
        assert isinstance(score, float)
        assert 0.0 <= confidence <= 1.0
        assert -1.0 <= score <= 1.0
    
    def test_analyze_sentiment_empty(self):
        """Test sentiment analysis on empty text."""
        sentiment, confidence, score = self.analyzer.analyze_sentiment("")
        
        assert sentiment == "neutral"
        assert confidence == 0.5
        assert score == 0.0
    
    def test_is_emotionally_charged(self):
        """Test emotional charge detection."""
        charged = self.analyzer.is_emotionally_charged("positive", 0.8)
        not_charged = self.analyzer.is_emotionally_charged("neutral", 0.5)
        
        assert isinstance(charged, bool)
        assert isinstance(not_charged, bool)


class TestProfanityFilter:
    """Test profanity filtering with real datasets."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = ProfanityFilter()
    
    def test_filter_profanity_clean_text(self):
        """Test profanity filtering on clean text."""
        has_profanity, filtered_text, detected = self.filter.filter_profanity("This is clean text")
        
        assert isinstance(has_profanity, bool)
        assert isinstance(filtered_text, str)
        assert isinstance(detected, list)
    
    def test_filter_profanity_empty_text(self):
        """Test profanity filtering on empty text."""
        has_profanity, filtered_text, detected = self.filter.filter_profanity("")
        
        assert has_profanity is False
        assert filtered_text == ""
        assert detected == []
    
    def test_get_profanity_score(self):
        """Test profanity score calculation."""
        score = self.filter.get_profanity_score(["word1", "word2"])
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0