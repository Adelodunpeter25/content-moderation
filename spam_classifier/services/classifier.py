"""Spam classification service using machine learning."""
import json
import joblib
import os
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from core.logging import logger
from .data_loader import DataLoader

class SpamClassifier:
    """Machine learning-based spam text classifier."""
    
    def __init__(self, model_name: str = "SpamGuard-v1"):
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        self.model_path = 'data/spam_model.joblib'
        self.vectorizer_path = 'data/spam_vectorizer.joblib'
        self.metadata_path = 'data/model_metadata.json'
        
    def train(self, texts: list[str], labels: list[int]) -> None:
        """Train the spam classifier with text samples and labels.
        
        Args:
            texts: List of text samples
            labels: List of labels (1 for spam, 0 for ham)
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        self._save_model()
        
    def predict(self, text: str) -> tuple[bool, float]:
        """Predict if text is spam.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (is_spam, confidence_score)
        """
        if not self.is_trained:
            if not self._load_model():
                self._load_or_create_model()
        
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0].max()
        
        return bool(prediction), float(confidence)
    
    def _load_or_create_model(self) -> None:
        """Load existing model or create one with combined datasets."""
        loader = DataLoader()
        texts, labels = loader.load_combined_datasets()
        
        logger.info(f"Training '{self.model_name}' with {len(texts)} samples")
        self.train(texts, labels)
    
    def _save_model(self) -> None:
        """Save trained model, vectorizer, and metadata."""
        os.makedirs('data', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        # Save model metadata
        metadata = {
            "model_name": self.model_name,
            "trained_at": datetime.now().isoformat(),
            "algorithm": "Naive Bayes + TF-IDF",
            "version": "1.0"
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_model(self) -> bool:
        """Load saved model and vectorizer.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_trained = True
            
            # Load metadata if available
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"Loaded model '{metadata['model_name']}' trained at {metadata['trained_at']}")
            else:
                logger.info("Loaded saved spam classification model")
            return True
        return False