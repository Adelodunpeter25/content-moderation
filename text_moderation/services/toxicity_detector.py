"""Toxicity detection service for harmful content."""
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from core.logging import logger

class ToxicityDetector:
    """ML-based toxicity detector using specialized datasets."""
    
    def __init__(self):
        from .dataset_loader import ModerationDatasetLoader
        
        self.dataset_loader = ModerationDatasetLoader()
        self.models = {}
        self.model_dir = Path("data/text_moderation/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_or_train_models()
    
    def _load_or_train_models(self) -> None:
        """Load existing models or train new ones from datasets."""
        model_files = {
            'toxicity': self.model_dir / 'toxicity_model.joblib',
            'hate_speech': self.model_dir / 'hate_speech_model.joblib',
            'offensive': self.model_dir / 'offensive_model.joblib'
        }
        
        # Load existing models if available
        for model_name, model_path in model_files.items():
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model from {model_path}")
            else:
                self._train_model(model_name)
    
    def _train_model(self, model_type: str) -> None:
        """Train ML model for specific moderation task.
        
        Args:
            model_type: Type of model to train (toxicity, hate_speech, offensive)
        """
        logger.info(f"Training {model_type} model...")
        
        # Load appropriate dataset
        if model_type == 'toxicity':
            texts, labels = self.dataset_loader.load_toxicity_dataset()
        elif model_type == 'hate_speech':
            texts, labels = self.dataset_loader.load_hate_speech_dataset()
        elif model_type == 'offensive':
            texts, labels = self.dataset_loader.load_offensive_language_dataset()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with TF-IDF and Logistic Regression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Train model
        pipeline.fit(texts[:10000], labels[:10000])  # Use subset for speed
        
        # Save model
        model_path = self.model_dir / f'{model_type}_model.joblib'
        joblib.dump(pipeline, model_path)
        self.models[model_type] = pipeline
        
        logger.info(f"Trained and saved {model_type} model to {model_path}")
    
    def detect_toxicity(self, text: str) -> Tuple[bool, float, List[str]]:
        """Detect toxicity using ML models.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (is_toxic, confidence, categories)
        """
        if not text or not isinstance(text, str):
            return False, 0.0, []
        
        detected_categories = []
        max_confidence = 0.0
        
        # Check each model
        for model_name, model in self.models.items():
            try:
                prediction = model.predict([text])[0]
                confidence = model.predict_proba([text])[0].max()
                
                if prediction == 1:  # Positive detection
                    detected_categories.append(model_name)
                    max_confidence = max(max_confidence, confidence)
            except Exception as e:
                logger.warning(f"Error in {model_name} model: {e}")
        
        is_toxic = len(detected_categories) > 0
        
        if is_toxic:
            logger.info(f"Toxic content detected: categories={detected_categories}, confidence={max_confidence:.3f}")
        
        return is_toxic, max_confidence, detected_categories
    
    def get_severity_score(self, categories: List[str]) -> float:
        """Get severity score based on detected categories using dataset statistics.
        
        Args:
            categories: List of detected toxicity categories
            
        Returns:
            Severity score from 0.0 to 1.0
        """
        if not categories:
            return 0.0
        
        # Calculate severity based on model confidence for each category
        max_severity = 0.0
        
        for category in categories:
            if category in self.models:
                # Use the model's feature importance or confidence as severity weight
                try:
                    # Get the model's coefficient magnitude as severity indicator
                    model = self.models[category]
                    if hasattr(model.named_steps['classifier'], 'coef_'):
                        coef_magnitude = abs(model.named_steps['classifier'].coef_).mean()
                        severity = min(1.0, coef_magnitude / 2.0)  # Normalize to 0-1
                    else:
                        severity = 0.7  # Default for non-linear models
                except Exception:
                    severity = 0.7  # Fallback
            else:
                severity = 0.5  # Unknown category
            
            max_severity = max(max_severity, severity)
        
        return max_severity