"""Sentiment analysis service for emotional content classification."""
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from core.logging import logger

class SentimentAnalyzer:
    """ML-based sentiment analyzer using specialized datasets."""
    
    def __init__(self):
        from .dataset_loader import ModerationDatasetLoader
        
        self.dataset_loader = ModerationDatasetLoader()
        self.model = None
        self.model_dir = Path("data/text_moderation/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_or_train_model()
    
    def _load_or_train_model(self) -> None:
        """Load existing model or train new one from dataset."""
        model_path = self.model_dir / 'sentiment_model.joblib'
        
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded sentiment model from {model_path}")
        else:
            self._train_model()
    
    def _train_model(self) -> None:
        """Train sentiment analysis model."""
        logger.info("Training sentiment model...")
        
        # Load dataset
        texts, labels = self.dataset_loader.load_sentiment_dataset()
        
        # Convert labels to binary
        binary_labels = [1 if label == 'positive' else 0 for label in labels]
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Train model
        pipeline.fit(texts[:10000], binary_labels[:10000])  # Use subset for speed
        
        # Save model
        model_path = self.model_dir / 'sentiment_model.joblib'
        joblib.dump(pipeline, model_path)
        self.model = pipeline
        
        logger.info(f"Trained and saved sentiment model to {model_path}")
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Analyze sentiment using ML model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence, score)
        """
        if not text or not isinstance(text, str):
            return "neutral", 0.5, 0.0
        
        try:
            # Get prediction
            prediction = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            
            # Calculate confidence and score
            confidence = probabilities.max()
            
            if prediction == 1:  # Positive
                sentiment = "positive"
                final_score = probabilities[1]
            else:  # Negative
                sentiment = "negative"
                final_score = -probabilities[0]
            
            # Check for neutral (low confidence)
            if confidence < 0.6:
                sentiment = "neutral"
                final_score = 0.0
            
            logger.info(f"Sentiment analysis: {sentiment} (confidence={confidence:.3f}, score={final_score:.3f})")
            
            return sentiment, confidence, final_score
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return "neutral", 0.5, 0.0
    
    def is_emotionally_charged(self, sentiment: str, confidence: float) -> bool:
        """Check if content is emotionally charged.
        
        Args:
            sentiment: Sentiment label
            confidence: Confidence score
            
        Returns:
            True if emotionally charged
        """
        return sentiment != "neutral" and confidence > 0.7