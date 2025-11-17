"""Spam classification service using machine learning."""
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from core.logging import logger
from .data_loader import DataLoader

class SpamClassifier:
    """Machine learning-based spam text classifier."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        self.model_path = 'data/spam_model.joblib'
        self.vectorizer_path = 'data/spam_vectorizer.joblib'
        
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
        
        logger.info(f"Training spam classifier with {len(texts)} samples")
        self.train(texts, labels)
    
    def _save_model(self) -> None:
        """Save trained model and vectorizer."""
        os.makedirs('data', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
    
    def _load_model(self) -> bool:
        """Load saved model and vectorizer.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_trained = True
            logger.info("Loaded saved spam classification model")
            return True
        return False