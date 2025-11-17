"""Model training and retraining pipeline."""
import joblib
import json
import os
from datetime import datetime
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from core.logging import logger
from .data_loader import DataLoader
from .feedback_manager import FeedbackManager
from .text_preprocessor import TextPreprocessor

class ModelTrainer:
    """Handles model training and retraining with user feedback."""
    
    def __init__(self, model_name: str = "SpamGuard-v2"):
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        self.feedback_manager = FeedbackManager()
        self.data_loader = DataLoader()
        
        # Model components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        
        # File paths
        self.model_path = 'data/spam_model.joblib'
        self.vectorizer_path = 'data/spam_vectorizer.joblib'
        self.metadata_path = 'data/model_metadata.json'
        
    def train_initial_model(self) -> Dict:
        """Train model from scratch using base datasets.
        
        Returns:
            Training metrics and statistics
        """
        logger.info(f"Starting initial training for {self.model_name}")
        
        # Load base datasets
        texts, labels = self.data_loader.load_combined_datasets()
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Train model
        metrics = self._train_model(processed_texts, labels)
        
        # Save model
        self._save_model()
        
        logger.info(f"Initial training completed. Accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
    def retrain_with_feedback(self, min_feedback_samples: int = 10) -> Dict:
        """Retrain model incorporating user feedback.
        
        Args:
            min_feedback_samples: Minimum feedback samples needed for retraining
            
        Returns:
            Retraining metrics and statistics
        """
        # Get feedback data
        feedback_texts, feedback_labels = self.feedback_manager.get_training_data_from_feedback()
        
        if len(feedback_texts) < min_feedback_samples:
            logger.info(f"Not enough feedback samples ({len(feedback_texts)}) for retraining")
            return {"status": "insufficient_feedback", "samples": len(feedback_texts)}
        
        logger.info(f"Starting retraining with {len(feedback_texts)} feedback samples")
        
        # Load base datasets
        base_texts, base_labels = self.data_loader.load_combined_datasets()
        
        # Combine base data with feedback
        all_texts = base_texts + feedback_texts
        all_labels = base_labels + feedback_labels
        
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in all_texts]
        
        # Train model
        metrics = self._train_model(processed_texts, all_labels)
        
        # Save updated model
        self._save_model(is_retrained=True)
        
        logger.info(f"Retraining completed. Accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
    def _train_model(self, texts: List[str], labels: List[int]) -> Dict:
        """Train the model and return metrics.
        
        Args:
            texts: Preprocessed text samples
            labels: Corresponding labels
            
        Returns:
            Training metrics
        """
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate detailed report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score'],
            "total_samples": len(texts),
            "test_samples": len(X_test)
        }
    
    def _save_model(self, is_retrained: bool = False) -> None:
        """Save trained model, vectorizer, and metadata.
        
        Args:
            is_retrained: Whether this is a retrained model
        """
        os.makedirs('data', exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        # Save metadata
        feedback_stats = self.feedback_manager.get_feedback_stats()
        metadata = {
            "model_name": self.model_name,
            "trained_at": datetime.now().isoformat(),
            "algorithm": "Naive Bayes + TF-IDF + Preprocessing",
            "version": "2.0",
            "is_retrained": is_retrained,
            "feedback_stats": feedback_stats
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model saved: {self.model_name}")