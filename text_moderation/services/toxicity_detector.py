"""Toxicity detection service for harmful content."""
import re
from typing import List, Tuple

from core.logging import logger

class ToxicityDetector:
    """Detects toxic content like hate speech, harassment, and threats."""
    
    def __init__(self):
        # Toxicity categories and keywords
        self.toxic_patterns = {
            "hate_speech": [
                "hate", "racist", "bigot", "nazi", "supremacist",
                "inferior", "subhuman", "scum", "vermin"
            ],
            "harassment": [
                "kill yourself", "kys", "die", "suicide", "harm yourself",
                "worthless", "pathetic", "loser", "failure"
            ],
            "threats": [
                "kill you", "murder", "hurt you", "beat you up",
                "find you", "come for you", "destroy you"
            ],
            "sexual_harassment": [
                "rape", "assault", "molest", "grope", "harass sexually"
            ],
            "bullying": [
                "ugly", "fat", "stupid", "retard", "freak",
                "nobody likes you", "everyone hates you"
            ]
        }
        
        # Compile patterns for better performance
        self.compiled_patterns = {}
        for category, words in self.toxic_patterns.items():
            pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words) + r')\b'
            self.compiled_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
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