"""Dataset loading service for text moderation training data."""
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json

from datasets import load_dataset
from core.logging import logger

class ModerationDatasetLoader:
    """Loads comprehensive datasets for toxicity, hate speech, and offensive content detection."""
    
    def __init__(self):
        self.data_dir = Path("data/text_moderation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_toxicity_dataset(self) -> Tuple[List[str], List[int]]:
        """Load Jigsaw Toxic Comment Classification dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for toxic, 0 for non-toxic
        """
        logger.info("Loading Jigsaw toxicity dataset...")
        
        try:
            # Load Jigsaw Toxic Comment Classification Challenge dataset
            dataset = load_dataset("unitary/toxic-bert", split="train")
            
            texts = dataset['text']
            labels = dataset['toxic']
            
            logger.info(f"Loaded {len(texts)} samples from Jigsaw toxicity dataset")
            return texts, labels
            
        except Exception as e:
            logger.warning(f"Failed to load Jigsaw dataset: {e}. Falling back to alternative.")
            # Fallback to Civil Comments dataset
            dataset = load_dataset("google/civil_comments", split="train[:10000]")
            texts = dataset['text']
            labels = [1 if row['toxicity'] > 0.5 else 0 for row in dataset]
            
            logger.info(f"Loaded {len(texts)} samples from Civil Comments dataset")
            return texts, labels
    
    def load_hate_speech_dataset(self) -> Tuple[List[str], List[int]]:
        """Load hate speech detection dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for hate speech, 0 for normal
        """
        logger.info("Loading hate speech dataset...")
        
        try:
            # Load HatEval dataset
            dataset = load_dataset("hateval", "english", split="train")
            
            texts = dataset['text']
            labels = dataset['HS']  # Hate Speech binary label
            
            logger.info(f"Loaded {len(texts)} samples from HatEval dataset")
            return texts, labels
            
        except Exception as e:
            logger.warning(f"Failed to load HatEval dataset: {e}. Using alternative.")
            # Fallback to Davidson et al. hate speech dataset
            dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train[:5000]")
            texts = dataset['text']
            labels = [1 if row['hate_speech_score'] > 0.5 else 0 for row in dataset]
            
            logger.info(f"Loaded {len(texts)} samples from hate speech dataset")
            return texts, labels
    
    def load_offensive_language_dataset(self) -> Tuple[List[str], List[int]]:
        """Load offensive language identification dataset.
        
        Returns:
            Tuple of (texts, labels) where labels are 1 for offensive, 0 for not offensive
        """
        logger.info("Loading offensive language dataset...")
        
        try:
            # Load OffensEval dataset
            dataset = load_dataset("cardiffnlp/tweet_eval", "offensive", split="train")
            
            texts = dataset['text']
            labels = dataset['label']
            
            logger.info(f"Loaded {len(texts)} samples from OffensEval dataset")
            return texts, labels
            
        except Exception as e:
            logger.warning(f"Failed to load OffensEval dataset: {e}. Using Founta dataset.")
            # Fallback to Founta et al. dataset
            dataset = load_dataset("tweet_eval", "offensive", split="train")
            texts = dataset['text']
            labels = dataset['label']
            
            logger.info(f"Loaded {len(texts)} samples from offensive language dataset")
            return texts, labels
    
    def load_sentiment_dataset(self) -> Tuple[List[str], List[str]]:
        """Load sentiment analysis dataset.
        
        Returns:
            Tuple of (texts, sentiment_labels)
        """
        logger.info("Loading sentiment dataset...")
        
        # Load Stanford Sentiment Treebank
        dataset = load_dataset("sst2", split="train")
        
        texts = dataset['sentence']
        labels = ['negative' if label == 0 else 'positive' for label in dataset['label']]
        
        logger.info(f"Loaded {len(texts)} samples from SST-2 sentiment dataset")
        return texts, labels
    
    def load_combined_moderation_dataset(self) -> Tuple[List[str], List[Dict[str, int]]]:
        """Load and combine multiple moderation datasets.
        
        Returns:
            Tuple of (texts, multi_labels) where multi_labels contains toxicity, hate_speech, offensive scores
        """
        logger.info("Loading combined moderation datasets...")
        
        # Load all datasets
        toxicity_texts, toxicity_labels = self.load_toxicity_dataset()
        hate_texts, hate_labels = self.load_hate_speech_dataset()
        offensive_texts, offensive_labels = self.load_offensive_language_dataset()
        
        # Combine datasets
        all_texts = toxicity_texts + hate_texts + offensive_texts
        
        # Create multi-label structure
        multi_labels = []
        
        # Add toxicity labels
        for i, label in enumerate(toxicity_labels):
            multi_labels.append({
                'toxicity': label,
                'hate_speech': 0,
                'offensive': 0
            })
        
        # Add hate speech labels
        for i, label in enumerate(hate_labels):
            multi_labels.append({
                'toxicity': 0,
                'hate_speech': label,
                'offensive': 0
            })
        
        # Add offensive language labels
        for i, label in enumerate(offensive_labels):
            multi_labels.append({
                'toxicity': 0,
                'hate_speech': 0,
                'offensive': label
            })
        
        logger.info(f"Combined {len(all_texts)} samples from multiple moderation datasets")
        return all_texts, multi_labels
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """Get statistics about loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        toxicity_texts, _ = self.load_toxicity_dataset()
        stats['toxicity_samples'] = len(toxicity_texts)
        
        hate_texts, _ = self.load_hate_speech_dataset()
        stats['hate_speech_samples'] = len(hate_texts)
        
        offensive_texts, _ = self.load_offensive_language_dataset()
        stats['offensive_samples'] = len(offensive_texts)
        
        sentiment_texts, _ = self.load_sentiment_dataset()
        stats['sentiment_samples'] = len(sentiment_texts)
        
        return stats