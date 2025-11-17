"""Spam classification service using machine learning."""
import json
import joblib
import os
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from core.logging import logger
from .feedback_manager import FeedbackManager
from .model_trainer import ModelTrainer
from .text_preprocessor import TextPreprocessor

class SpamClassifier:
    """Machine learning-based spam text classifier."""
    
    def __init__(self, model_name: str = "SpamGuard-v2"):
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        self.feedback_manager = FeedbackManager()
        self.trainer = ModelTrainer(model_name)
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.model_path = 'data/spam_model.joblib'
        self.vectorizer_path = 'data/spam_vectorizer.joblib'
        self.metadata_path = 'data/model_metadata.json'
        

        
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
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        X = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X)[0].max()
        
        return bool(prediction), float(confidence)
    
    def _load_or_create_model(self) -> None:
        """Load existing model or create one with combined datasets."""
        metrics = self.trainer.train_initial_model()
        self.vectorizer = self.trainer.vectorizer
        self.model = self.trainer.model
        self.is_trained = True
        logger.info(f"Model '{self.model_name}' trained with accuracy: {metrics['accuracy']:.3f}")
    
    def record_feedback(self, text: str, predicted_spam: bool, actual_spam: bool, 
                      confidence: float, user_id: str = "anonymous") -> None:
        """Record user feedback for model improvement.
        
        Args:
            text: The text that was classified
            predicted_spam: What the model predicted
            actual_spam: What the user says it actually is
            confidence: Model's confidence in prediction
            user_id: ID of user providing feedback
        """
        self.feedback_manager.record_feedback(
            text, predicted_spam, actual_spam, confidence, user_id
        )
    
    def retrain_model(self) -> dict:
        """Retrain model with user feedback.
        
        Returns:
            Retraining results and metrics
        """
        metrics = self.trainer.retrain_with_feedback()
        if metrics.get('status') != 'insufficient_feedback':
            self.vectorizer = self.trainer.vectorizer
            self.model = self.trainer.model
            logger.info(f"Model retrained with accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
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