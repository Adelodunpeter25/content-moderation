"""Toxicity detection service for harmful content."""
import re
from typing import List, Tuple

from core.logging import logger

class ToxicityDetector:
    """Detects toxic content like hate speech, harassment, and threats."""
    
    def __init__(self):
        from .dataset_loader import ModerationDatasetLoader
        
        self.dataset_loader = ModerationDatasetLoader()
        self.toxic_patterns = self._load_toxic_patterns()
        self.training_data = None
        
        # Compile patterns for better performance
        self.compiled_patterns = {}
        for category, words in self.toxic_patterns.items():
            pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words) + r')\b'
            self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def _load_toxic_patterns(self) -> dict:
        """Load toxicity patterns from real dataset."""
        # Load from dataset
        texts, labels = self.dataset_loader.load_toxicity_dataset()
        
        # Extract patterns from toxic examples
        toxic_texts = [texts[i] for i, label in enumerate(labels) if label == 1]
        
        # Extract common toxic words from dataset
        toxic_words = set()
        for text in toxic_texts[:500]:  # Process subset for performance
            words = text.lower().split()
            toxic_words.update([word for word in words if len(word) > 3])
        
        # Categorize based on common patterns (simplified)
        return {
            "general_toxic": list(toxic_words)[:100]  # Use top 100 most common
        }
    
    def detect_toxicity(self, text: str) -> Tuple[bool, float, List[str]]:
        """Detect toxicity in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (is_toxic, confidence, categories)
        """
        if not text or not isinstance(text, str):
            return False, 0.0, []
        
        detected_categories = []
        total_matches = 0
        
        # Check each toxicity category
        for category, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text.lower())
            if matches:
                detected_categories.append(category)
                total_matches += len(matches)
        
        # Calculate confidence based on matches and text length
        is_toxic = len(detected_categories) > 0
        confidence = min(0.9, (total_matches / max(len(text.split()), 1)) * 10) if is_toxic else 0.1
        
        if is_toxic:
            logger.info(f"Toxic content detected: categories={detected_categories}, confidence={confidence:.3f}")
        
        return is_toxic, confidence, detected_categories
    
    def get_severity_score(self, categories: List[str]) -> float:
        """Get severity score based on detected categories.
        
        Args:
            categories: List of detected toxicity categories
            
        Returns:
            Severity score from 0.0 to 1.0
        """
        severity_weights = {
            "threats": 1.0,
            "hate_speech": 0.9,
            "sexual_harassment": 0.8,
            "harassment": 0.7,
            "bullying": 0.5
        }
        
        if not categories:
            return 0.0
        
        max_severity = max(severity_weights.get(cat, 0.3) for cat in categories)
        return max_severity