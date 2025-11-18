"""Test dataset loading functionality."""
import pytest
from text_moderation.services.dataset_loader import ModerationDatasetLoader


class TestModerationDatasetLoader:
    """Test dataset loading for content moderation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ModerationDatasetLoader()
    
    def test_load_toxicity_dataset(self):
        """Test toxicity dataset loading."""
        texts, labels = self.loader.load_toxicity_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
        assert all(label in [0, 1] for label in labels[:10])
    
    def test_load_hate_speech_dataset(self):
        """Test hate speech dataset loading."""
        texts, labels = self.loader.load_hate_speech_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
        assert all(label in [0, 1] for label in labels[:10])
    
    def test_load_offensive_language_dataset(self):
        """Test offensive language dataset loading."""
        texts, labels = self.loader.load_offensive_language_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
    
    def test_load_sentiment_dataset(self):
        """Test sentiment dataset loading."""
        texts, labels = self.loader.load_sentiment_dataset()
        
        assert len(texts) > 0
        assert len(labels) > 0
        assert len(texts) == len(labels)
        assert all(isinstance(text, str) for text in texts[:10])
        assert all(label in ['positive', 'negative'] for label in labels[:10])
    
    def test_load_combined_moderation_dataset(self):
        """Test combined moderation dataset loading."""
        texts, multi_labels = self.loader.load_combined_moderation_dataset()
        
        assert len(texts) > 0
        assert len(multi_labels) > 0
        assert len(texts) == len(multi_labels)
        
        # Check multi-label structure
        for label_dict in multi_labels[:5]:
            assert 'toxicity' in label_dict
            assert 'hate_speech' in label_dict
            assert 'offensive' in label_dict
            assert all(val in [0, 1] for val in label_dict.values())
    
    def test_get_dataset_stats(self):
        """Test dataset statistics."""
        stats = self.loader.get_dataset_stats()
        
        assert 'toxicity_samples' in stats
        assert 'hate_speech_samples' in stats
        assert 'offensive_samples' in stats
        assert 'sentiment_samples' in stats
        assert all(count > 0 for count in stats.values())