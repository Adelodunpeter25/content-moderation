"""Sentiment analysis service for emotional content classification."""
import re
from typing import Tuple, List

from core.logging import logger

class SentimentAnalyzer:
    """Analyzes sentiment and emotional tone of text."""
    
    def __init__(self):
        from .dataset_loader import ModerationDatasetLoader
        
        self.dataset_loader = ModerationDatasetLoader()
        self.positive_words, self.negative_words = self._load_sentiment_words()
        
        # Intensifiers
        self.intensifiers = [
            "very", "extremely", "really", "totally", "completely",
            "absolutely", "quite", "rather", "pretty", "so"
        ]
        
        # Negation words
        self.negations = [
            "not", "no", "never", "nothing", "nobody", "nowhere",
            "neither", "nor", "none", "hardly", "scarcely", "barely"
        ]
        
        # Compile patterns
        self.positive_pattern = re.compile(r'\b(?:' + '|'.join(self.positive_words) + r')\b', re.IGNORECASE)
        self.negative_pattern = re.compile(r'\b(?:' + '|'.join(self.negative_words) + r')\b', re.IGNORECASE)
        self.intensifier_pattern = re.compile(r'\b(?:' + '|'.join(self.intensifiers) + r')\b', re.IGNORECASE)
        self.negation_pattern = re.compile(r'\b(?:' + '|'.join(self.negations) + r')\b', re.IGNORECASE)
    
    def _load_sentiment_words(self) -> Tuple[List[str], List[str]]:
        """Load sentiment words from real dataset."""
        texts, labels = self.dataset_loader.load_sentiment_dataset()
        
        # Extract words from positive and negative examples
        positive_texts = [texts[i] for i, label in enumerate(labels) if label == 'positive']
        negative_texts = [texts[i] for i, label in enumerate(labels) if label == 'negative']
        
        # Extract common words from each sentiment category
        positive_words = set()
        negative_words = set()
        
        # Process positive texts
        for text in positive_texts[:200]:  # Limit for performance
            words = text.lower().split()
            positive_words.update([word for word in words if len(word) > 3])
        
        # Process negative texts
        for text in negative_texts[:200]:  # Limit for performance
            words = text.lower().split()
            negative_words.update([word for word in words if len(word) > 3])
        
        return list(positive_words)[:50], list(negative_words)[:50]
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence, score)
        """
        if not text or not isinstance(text, str):
            return "neutral", 0.5, 0.0
        
        # Count sentiment words
        positive_matches = len(self.positive_pattern.findall(text))
        negative_matches = len(self.negative_pattern.findall(text))
        intensifier_matches = len(self.intensifier_pattern.findall(text))
        negation_matches = len(self.negation_pattern.findall(text))
        
        # Apply intensifier boost
        intensifier_boost = min(0.3, intensifier_matches * 0.1)
        
        # Apply negation (flips sentiment)
        if negation_matches > 0:
            positive_matches, negative_matches = negative_matches, positive_matches
        
        # Calculate raw score
        total_words = len(text.split())
        positive_score = (positive_matches / max(total_words, 1)) + intensifier_boost
        negative_score = (negative_matches / max(total_words, 1)) + intensifier_boost
        
        # Determine sentiment
        score_diff = positive_score - negative_score
        
        if abs(score_diff) < 0.1:
            sentiment = "neutral"
            confidence = 0.5
            final_score = 0.0
        elif score_diff > 0:
            sentiment = "positive"
            confidence = min(0.95, 0.5 + abs(score_diff))
            final_score = min(1.0, score_diff * 2)
        else:
            sentiment = "negative"
            confidence = min(0.95, 0.5 + abs(score_diff))
            final_score = max(-1.0, score_diff * 2)
        
        logger.info(f"Sentiment analysis: {sentiment} (confidence={confidence:.3f}, score={final_score:.3f})")
        
        return sentiment, confidence, final_score
    
    def is_emotionally_charged(self, sentiment: str, confidence: float) -> bool:
        """Check if content is emotionally charged.
        
        Args:
            sentiment: Sentiment label
            confidence: Confidence score
            
        Returns:
            True if emotionally charged
        """
        return sentiment != "neutral" and confidence > 0.7