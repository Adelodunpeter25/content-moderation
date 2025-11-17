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
        logger.info("Loading Jigsaw Toxic Comment Classification dataset...")
        
        # Load the dataset from HuggingFace
        dataset = load_dataset("unitary/toxic-bert", split="train[:5000]")  # Limit for demo
        
        texts = dataset['text']
        # Convert toxic column to binary (1 if any toxicity > 0.5, 0 otherwise)
        labels = [1 if row['toxic'] > 0.5 else 0 for row in dataset]
        
        logger.info(f"Loaded {len(texts)} toxicity samples from Jigsaw dataset")
        return texts, labels
    
    def load_profanity_wordlist(self) -> List[str]:
        """Load profanity word list from dataset.
        
        Returns:
            List of profanity words
        """
        logger.info("Loading profanity dataset...")
        
        # Load profanity dataset
        dataset = load_dataset("martin-ha/offensive-language-dataset", split="train")
        
        # Extract offensive words from the dataset
        offensive_texts = [row['text'] for row in dataset if row['label'] == 1]
        
        # Extract individual words (simplified approach)
        profanity_words = set()
        for text in offensive_texts[:1000]:  # Limit for processing
            words = text.lower().split()
            # Add words that appear frequently in offensive content
            profanity_words.update([word for word in words if len(word) > 3])
        
        profanity_list = list(profanity_words)[:500]  # Limit list size
        
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