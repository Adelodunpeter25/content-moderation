"""Dataset loading service for text moderation training data."""
import pandas as pd
import requests
from pathlib import Path
from typing import List, Tuple, Dict
import json

from core.logging import logger

class ModerationDatasetLoader:
    """Loads datasets for toxicity, profanity, and sentiment analysis."""
    
    def __init__(self):
        self.data_dir = Path("data/text_moderation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_toxicity_dataset(self) -> Tuple[List[str], List[int]]:
        """Load toxicity detection dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for toxic, 0 for non-toxic
        """
        file_path = self.data_dir / "toxicity_data.json"
        
        if not file_path.exists():
            self._download_toxicity_dataset(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['toxic'] for item in data]
        
        logger.info(f"Loaded {len(texts)} toxicity samples")
        return texts, labels
    
    def load_profanity_wordlist(self) -> List[str]:
        """Load comprehensive profanity word list.
        
        Returns:
            List of profanity words
        """
        file_path = self.data_dir / "profanity_words.txt"
        
        if not file_path.exists():
            self._download_profanity_wordlist(file_path)
        
        with open(file_path, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(words)} profanity words")
        return words
    
    def load_sentiment_dataset(self) -> Tuple[List[str], List[str]]:
        """Load sentiment analysis dataset.
        
        Returns:
            Tuple of (texts, sentiment_labels)
        """
        file_path = self.data_dir / "sentiment_data.json"
        
        if not file_path.exists():
            self._download_sentiment_dataset(file_path)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['sentiment'] for item in data]
        
        logger.info(f"Loaded {len(texts)} sentiment samples")
        return texts, labels
    
    def _download_toxicity_dataset(self, file_path: Path) -> None:
        """Download and create toxicity dataset."""
        logger.info("Creating toxicity dataset from curated examples...")
        
        # Curated examples for demonstration (in production, use real datasets)
        toxic_examples = [
            {"text": "I hate you and wish you would die", "toxic": 1},
            {"text": "You're such a worthless piece of garbage", "toxic": 1},
            {"text": "Kill yourself, nobody likes you", "toxic": 1},
            {"text": "You're an idiot and should be ashamed", "toxic": 1},
            {"text": "I'm going to find you and hurt you", "toxic": 1},
            {"text": "Thanks for your help with this project", "toxic": 0},
            {"text": "I disagree with your opinion but respect it", "toxic": 0},
            {"text": "Could you please help me understand this better", "toxic": 0},
            {"text": "I think there might be a better approach", "toxic": 0},
            {"text": "Great work on this presentation", "toxic": 0}
        ]
        
        with open(file_path, 'w') as f:
            json.dump(toxic_examples, f, indent=2)
        
        logger.info(f"Toxicity dataset saved to {file_path}")
    
    def _download_profanity_wordlist(self, file_path: Path) -> None:
        """Download comprehensive profanity word list."""
        logger.info("Creating profanity word list...")
        
        # Extended profanity list (mild examples for demonstration)
        profanity_words = [
            # Mild profanity
            "damn", "hell", "crap", "suck", "sucks", "stupid", "idiot",
            "moron", "dumb", "lame", "wtf", "omg", "bs", "bullshit",
            
            # Offensive terms
            "hate", "loser", "freak", "weirdo", "creep", "jerk",
            
            # Variations and leetspeak
            "d4mn", "h3ll", "cr4p", "5tup1d", "1d10t", "m0r0n",
            
            # Obfuscated versions
            "d*mn", "h*ll", "cr*p", "s*ck", "st*pid"
        ]
        
        with open(file_path, 'w') as f:
            for word in profanity_words:
                f.write(word + '\n')
        
        logger.info(f"Profanity wordlist saved to {file_path}")
    
    def _download_sentiment_dataset(self, file_path: Path) -> None:
        """Download and create sentiment dataset."""
        logger.info("Creating sentiment dataset from curated examples...")
        
        # Curated sentiment examples
        sentiment_examples = [
            {"text": "I absolutely love this product, it's amazing!", "sentiment": "positive"},
            {"text": "This is the best thing I've ever bought", "sentiment": "positive"},
            {"text": "Excellent quality and fast shipping", "sentiment": "positive"},
            {"text": "Really happy with my purchase", "sentiment": "positive"},
            {"text": "Outstanding customer service", "sentiment": "positive"},
            
            {"text": "This product is terrible and broke immediately", "sentiment": "negative"},
            {"text": "Worst purchase I've ever made", "sentiment": "negative"},
            {"text": "Complete waste of money", "sentiment": "negative"},
            {"text": "Very disappointed with the quality", "sentiment": "negative"},
            {"text": "Would not recommend to anyone", "sentiment": "negative"},
            
            {"text": "The product arrived on time", "sentiment": "neutral"},
            {"text": "It works as described", "sentiment": "neutral"},
            {"text": "Standard quality for the price", "sentiment": "neutral"},
            {"text": "Nothing special but does the job", "sentiment": "neutral"},
            {"text": "Average product, meets expectations", "sentiment": "neutral"}
        ]
        
        with open(file_path, 'w') as f:
            json.dump(sentiment_examples, f, indent=2)
        
        logger.info(f"Sentiment dataset saved to {file_path}")
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        try:
            toxicity_texts, _ = self.load_toxicity_dataset()
            stats['toxicity_samples'] = len(toxicity_texts)
        except:
            stats['toxicity_samples'] = 0
        
        try:
            profanity_words = self.load_profanity_wordlist()
            stats['profanity_words'] = len(profanity_words)
        except:
            stats['profanity_words'] = 0
        
        try:
            sentiment_texts, _ = self.load_sentiment_dataset()
            stats['sentiment_samples'] = len(sentiment_texts)
        except:
            stats['sentiment_samples'] = 0
        
        return stats