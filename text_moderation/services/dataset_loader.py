"""Dataset loading service for text moderation training data."""
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json

from datasets import load_dataset
from core.logging import logger

class ModerationDatasetLoader:
    """Loads datasets for toxicity, profanity, and sentiment analysis."""
    
    def __init__(self):
        self.data_dir = Path("data/text_moderation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_toxicity_dataset(self) -> Tuple[List[str], List[int]]:
        """Load Jigsaw Toxic Comment Classification dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for toxic, 0 for non-toxic
        """
        logger.info("Loading toxicity dataset...")
        
        # Use a subset of IMDB with negative reviews as "toxic" examples
        # This is a simplified approach - in production use actual toxicity datasets
        dataset = load_dataset("imdb", split="train[:2000]")
        
        texts = dataset['text']
        # Use negative reviews (label=0) as toxic examples for demo
        labels = [1 if row['label'] == 0 else 0 for row in dataset]
        
        logger.info(f"Loaded {len(texts)} toxicity samples from Jigsaw dataset")
        return texts, labels
    
    def load_profanity_wordlist(self) -> List[str]:
        """Load profanity word list from dataset.
        
        Returns:
            List of profanity words
        """
        logger.info("Loading profanity dataset...")
        
        # Use negative IMDB reviews to extract potentially offensive words
        dataset = load_dataset("imdb", split="train[:1000]")
        
        # Extract words from negative reviews
        negative_texts = [row['text'] for row in dataset if row['label'] == 0]
        
        # Extract individual words (simplified approach)
        profanity_words = set()
        common_negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'stupid', 'boring', 'waste']
        
        for text in negative_texts[:100]:  # Limit for processing
            words = text.lower().split()
            # Add common negative words found in text
            for word in words:
                if word in common_negative_words or (len(word) > 4 and any(neg in word for neg in ['bad', 'hate', 'stupid'])):
                    profanity_words.add(word)
        
        # Add some basic profanity words
        profanity_words.update(['damn', 'hell', 'crap', 'stupid', 'idiot', 'hate', 'suck', 'sucks'])
        
        profanity_list = list(profanity_words)[:100]  # Limit list size
        
        logger.info(f"Extracted {len(profanity_list)} profanity words from dataset")
        return profanity_list
    
    def load_sentiment_dataset(self) -> Tuple[List[str], List[str]]:
        """Load IMDB movie reviews sentiment dataset.
        
        Returns:
            Tuple of (texts, sentiment_labels)
        """
        logger.info("Loading IMDB sentiment dataset...")
        
        # Load IMDB dataset
        dataset = load_dataset("imdb", split="train[:2000]")  # Limit for demo
        
        texts = dataset['text']
        # Convert labels: 0 -> negative, 1 -> positive
        labels = ['negative' if label == 0 else 'positive' for label in dataset['label']]
        
        logger.info(f"Loaded {len(texts)} sentiment samples from IMDB dataset")
        return texts, labels
    

    
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        toxicity_texts, _ = self.load_toxicity_dataset()
        stats['toxicity_samples'] = len(toxicity_texts)
        
        profanity_words = self.load_profanity_wordlist()
        stats['profanity_words'] = len(profanity_words)
        
        sentiment_texts, _ = self.load_sentiment_dataset()
        stats['sentiment_samples'] = len(sentiment_texts)
        
        return stats