"""Profanity filtering service for inappropriate language."""
import re
from typing import List, Tuple

from core.logging import logger

class ProfanityFilter:
    """Filters profanity and inappropriate language."""
    
    def __init__(self):
        from .dataset_loader import ModerationDatasetLoader
        
        self.dataset_loader = ModerationDatasetLoader()
        self.profanity_words = self._load_profanity_words()
        
        # Leetspeak and obfuscation patterns
        self.leetspeak_map = {
            '4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's',
            '7': 't', '@': 'a', '$': 's', '!': 'i'
        }
        
        # Compile regex patterns
        self.profanity_pattern = self._build_profanity_pattern()
    
    def _load_profanity_words(self) -> List[str]:
        """Extract profanity patterns from offensive language dataset."""
        texts, labels = self.dataset_loader.load_offensive_language_dataset()
        
        # Extract words from offensive examples
        offensive_words = set()
        offensive_texts = [texts[i] for i, label in enumerate(labels) if label == 1]
        
        for text in offensive_texts[:1000]:  # Process subset
            words = text.lower().split()
            # Filter for potentially offensive words (length > 3, contains certain patterns)
            for word in words:
                if len(word) > 3 and any(char in word for char in 'aeiou'):
                    offensive_words.add(word)
        
        return list(offensive_words)[:200]  # Return top 200
        
    def _build_profanity_pattern(self) -> re.Pattern:
        """Build regex pattern for profanity detection."""
        # Create variations with common obfuscations
        patterns = []
        for word in self.profanity_words:
            # Original word
            patterns.append(re.escape(word))
            
            # With asterisks (s*ck, d*mn)
            if len(word) > 3:
                obfuscated = word[0] + '*' * (len(word) - 2) + word[-1]
                patterns.append(re.escape(obfuscated))
            
            # With numbers/symbols
            leetspeak = self._to_leetspeak(word)
            if leetspeak != word:
                patterns.append(re.escape(leetspeak))
        
        pattern = r'\b(?:' + '|'.join(patterns) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _to_leetspeak(self, word: str) -> str:
        """Convert word to common leetspeak variations."""
        result = word.lower()
        for leet, normal in self.leetspeak_map.items():
            result = result.replace(normal, leet)
        return result
    
    def filter_profanity(self, text: str) -> Tuple[bool, str, List[str]]:
        """Filter profanity from text.
        
        Args:
            text: Text to filter
            
        Returns:
            Tuple of (has_profanity, filtered_text, detected_words)
        """
        if not text or not isinstance(text, str):
            return False, text, []
        
        detected_words = []
        filtered_text = text
        
        # Find all profanity matches
        matches = self.profanity_pattern.finditer(text)
        
        for match in matches:
            word = match.group()
            detected_words.append(word.lower())
            
            # Replace with asterisks
            replacement = word[0] + '*' * (len(word) - 2) + word[-1] if len(word) > 2 else '*' * len(word)
            filtered_text = filtered_text.replace(word, replacement)
        
        has_profanity = len(detected_words) > 0
        
        if has_profanity:
            logger.info(f"Profanity detected: {len(detected_words)} words filtered")
        
        return has_profanity, filtered_text, list(set(detected_words))
    
    def get_profanity_score(self, detected_words: List[str]) -> float:
        """Calculate profanity severity score based on dataset frequency.
        
        Args:
            detected_words: List of detected profanity words
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not detected_words:
            return 0.0
        
        # Calculate score based on word frequency in offensive dataset
        total_score = 0.0
        
        for word in detected_words:
            # Calculate word severity based on its frequency in offensive examples
            word_frequency = self.profanity_words.count(word) if word in self.profanity_words else 0
            word_severity = min(1.0, word_frequency / max(len(self.profanity_words), 1) * 10)
            total_score += word_severity
        
        # Normalize by number of detected words
        average_score = total_score / len(detected_words)
        return min(1.0, average_score)