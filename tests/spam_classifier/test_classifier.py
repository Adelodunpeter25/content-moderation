"""Test spam classifier functionality."""
import pytest
from spam_classifier.services.classifier import SpamClassifier
from spam_classifier.services.model_trainer import ModelTrainer
from spam_classifier.services.data_loader import DataLoader


class TestSpamClassifier:
    """Test spam classification service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = SpamClassifier()
    
    def test_predict_spam_text(self):
        """Test spam prediction on spam-like text."""
        is_spam, confidence = self.classifier.predict("FREE MONEY! Click here to win $1000 NOW!")
        
        assert isinstance(is_spam, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_ham_text(self):
        """Test spam prediction on legitimate text."""
        is_spam, confidence = self.classifier.predict("Hello, how are you doing today?")
        
        assert isinstance(is_spam, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_empty_text(self):
        """Test spam prediction on empty text."""
        is_spam, confidence = self.classifier.predict("")
        
        assert isinstance(is_spam, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_record_feedback(self):
        """Test feedback recording."""
        self.classifier.record_feedback(
            "test message", 
            predicted_spam=False, 
            actual_spam=True, 
            confidence=0.8
        )
        # Should not raise exception
        assert True
    
    def test_retrain_model(self):
        """Test model retraining."""
        result = self.classifier.retrain_model()
        
        assert isinstance(result, dict)
        assert 'status' in result


class TestModelTrainer:
    """Test model training functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer()
    
    def test_train_initial_model(self):
        """Test initial model training."""
        result = self.trainer.train_initial_model()
        
        assert 'accuracy' in result
        assert isinstance(result['accuracy'], float)
        assert 0.0 <= result['accuracy'] <= 1.0
    
    def test_retrain_with_feedback(self):
        """Test model retraining with feedback."""
        result = self.trainer.retrain_with_feedback()
        
        assert 'status' in result
        assert isinstance(result['status'], str)


class TestDataLoader:
    """Test data loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
    
    def test_load_sms_spam_dataset(self):
        """Test SMS spam data loading."""
        texts, labels = self.loader.load_sms_spam_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
        assert all(isinstance(label, (str, int)) for label in labels[:10])
    
    def test_load_youtube_spam_dataset(self):
        """Test YouTube spam data loading."""
        texts, labels = self.loader.load_youtube_spam_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
        assert all(label in [0, 1] for label in labels[:10])
    
    def test_load_combined_datasets(self):
        """Test combined dataset loading."""
        texts, labels = self.loader.load_combined_datasets()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
    
    def test_loader_instantiation(self):
        """Test data loader can be instantiated."""
        assert self.loader is not None
        assert hasattr(self.loader, 'load_combined_datasets')