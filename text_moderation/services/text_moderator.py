"""Main text moderation service combining all analysis types."""
from typing import Dict, Any

from core.logging import logger
from .toxicity_detector import ToxicityDetector
from .profanity_filter import ProfanityFilter
from .sentiment_analyzer import SentimentAnalyzer

class TextModerator:
    """Main text moderation service combining toxicity, profanity, and sentiment analysis."""
    
    def __init__(self):
        self.toxicity_detector = ToxicityDetector()
        self.profanity_filter = ProfanityFilter()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def moderate_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text moderation.
        
        Args:
            text: Text to moderate
            
        Returns:
            Dictionary with all moderation results
        """
        logger.info(f"Moderating text: {len(text)} characters")
        
        # Toxicity detection
        is_toxic, toxic_confidence, toxic_categories = self.toxicity_detector.detect_toxicity(text)
        toxicity_result = {
            "is_toxic": is_toxic,
            "confidence": toxic_confidence,
            "categories": toxic_categories
        }
        
        # Profanity filtering
        has_profanity, filtered_text, detected_words = self.profanity_filter.filter_profanity(text)
        profanity_result = {
            "has_profanity": has_profanity,
            "filtered_text": filtered_text,
            "detected_words": detected_words
        }
        
        # Sentiment analysis
        sentiment, sentiment_confidence, sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
        sentiment_result = {
            "sentiment": sentiment,
            "confidence": sentiment_confidence,
            "score": sentiment_score
        }
        
        # Overall safety assessment
        overall_safe = self._assess_overall_safety(
            is_toxic, toxic_confidence, has_profanity, sentiment, sentiment_confidence
        )
        
        result = {
            "text": text,
            "toxicity": toxicity_result,
            "profanity": profanity_result,
            "sentiment": sentiment_result,
            "overall_safe": overall_safe
        }
        
        logger.info(f"Moderation complete: safe={overall_safe}, toxic={is_toxic}, profanity={has_profanity}")
        return result
    
    def _assess_overall_safety(self, is_toxic: bool, toxic_confidence: float, 
                              has_profanity: bool, sentiment: str, sentiment_confidence: float) -> bool:
        """Assess overall content safety.
        
        Args:
            is_toxic: Whether content is toxic
            toxic_confidence: Toxicity confidence score
            has_profanity: Whether content has profanity
            sentiment: Sentiment classification
            sentiment_confidence: Sentiment confidence score
            
        Returns:
            True if content is overall safe
        """
        # High-confidence toxic content is unsafe
        if is_toxic and toxic_confidence > 0.7:
            return False
        
        # Multiple profanity words indicate unsafe content
        if has_profanity:
            return False
        
        # Highly negative sentiment with high confidence may indicate unsafe content
        if sentiment == "negative" and sentiment_confidence > 0.8:
            return False
        
        return True
    
    def get_moderation_summary(self, moderation_result: Dict[str, Any]) -> str:
        """Get human-readable moderation summary.
        
        Args:
            moderation_result: Result from moderate_text()
            
        Returns:
            Summary string
        """
        issues = []
        
        if moderation_result["toxicity"]["is_toxic"]:
            categories = ", ".join(moderation_result["toxicity"]["categories"])
            issues.append(f"Toxic content detected: {categories}")
        
        if moderation_result["profanity"]["has_profanity"]:
            word_count = len(moderation_result["profanity"]["detected_words"])
            issues.append(f"Profanity detected: {word_count} words")
        
        sentiment = moderation_result["sentiment"]["sentiment"]
        if sentiment != "neutral":
            issues.append(f"Sentiment: {sentiment}")
        
        if not issues:
            return "Content appears safe"
        
        return "; ".join(issues)