"""Test image dataset loader."""
import pytest
from unittest.mock import patch

from image_moderation.services.dataset_loader import ImageDatasetLoader


class TestImageDatasetLoader:
    """Test image dataset loading."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ImageDatasetLoader()
    
    def test_load_nsfw_dataset(self):
        """Test NSFW dataset loading."""
        image_paths, labels = self.loader.load_nsfw_dataset()
        
        assert len(image_paths) > 0
        assert len(labels) > 0
        assert len(image_paths) == len(labels)
        assert all(isinstance(label, int) for label in labels[:5])
    
    def test_load_violence_dataset(self):
        """Test violence dataset loading."""
        image_paths, labels = self.loader.load_violence_dataset()
        
        assert len(image_paths) > 0
        assert len(labels) > 0
        assert len(image_paths) == len(labels)
        assert all(isinstance(label, int) for label in labels[:5])
    
    def test_load_face_dataset(self):
        """Test face dataset loading."""
        image_paths, annotations = self.loader.load_face_dataset()
        
        assert len(image_paths) > 0
        assert len(annotations) > 0
        assert len(image_paths) == len(annotations)
        assert all(isinstance(ann, dict) for ann in annotations[:5])
    
    def test_get_dataset_stats(self):
        """Test dataset statistics."""
        stats = self.loader.get_dataset_stats()
        
        assert 'nsfw_samples' in stats
        assert 'violence_samples' in stats
        assert 'face_samples' in stats
        assert all(isinstance(count, int) for count in stats.values())
        assert all(count > 0 for count in stats.values())
    
    @patch('image_moderation.services.dataset_loader.load_dataset')
    def test_analyze_violence_categories(self, mock_load_dataset):
        """Test violence category analysis."""
        mock_dataset = [
            {'objects': {'category': ['tool', 'equipment']}},
            {'objects': {'category': ['person', 'tool']}},
            {'objects': {'category': ['vehicle']}}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        categories = self.loader._analyze_violence_categories(mock_dataset)
        
        assert isinstance(categories, list)
        assert len(categories) <= 10
    
    def test_generate_synthetic_nsfw_data(self):
        """Test synthetic NSFW data generation."""
        image_paths, labels = self.loader._generate_synthetic_nsfw_data()
        
        assert len(image_paths) == 1000
        assert len(labels) == 1000
        assert all(isinstance(path, str) for path in image_paths[:5])
        assert all(label in [0, 1] for label in labels)
    
    def test_generate_synthetic_violence_data(self):
        """Test synthetic violence data generation."""
        image_paths, labels = self.loader._generate_synthetic_violence_data()
        
        assert len(image_paths) == 500
        assert len(labels) == 500
        assert all(isinstance(path, str) for path in image_paths[:5])
        assert all(label in [0, 1] for label in labels)