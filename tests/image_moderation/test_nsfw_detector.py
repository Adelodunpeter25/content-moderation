"""Test NSFW detector service."""
import pytest
from PIL import Image
import io
import base64

from image_moderation.services.nsfw_detector import NSFWDetector


class TestNSFWDetector:
    """Test NSFW detection service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = NSFWDetector()
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self) -> bytes:
        """Create test image data."""
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def test_detect_nsfw(self):
        """Test NSFW detection."""
        is_nsfw, confidence, categories = self.detector.detect_nsfw(self.test_image_data)
        
        assert isinstance(is_nsfw, bool)
        assert isinstance(confidence, float)
        assert isinstance(categories, dict)
        assert 0.0 <= confidence <= 1.0
    
    def test_load_image_from_base64(self):
        """Test base64 image loading."""
        base64_data = base64.b64encode(self.test_image_data).decode()
        loaded_data = self.detector.load_image_from_base64(base64_data)
        
        assert isinstance(loaded_data, bytes)
        assert len(loaded_data) > 0
    
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.detector._extract_features(self.test_image_data)
        
        assert len(features) == 20
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_generate_category_scores(self):
        """Test category score generation."""
        import numpy as np
        probabilities = np.array([0.3, 0.7])
        features = np.random.rand(20)
        
        scores = self.detector._generate_category_scores(probabilities, features)
        
        assert isinstance(scores, dict)
        assert 'safe' in scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    def test_estimate_error_confidence(self):
        """Test error confidence estimation."""
        confidence = self.detector._estimate_error_confidence()
        
        assert isinstance(confidence, float)
        assert 0.3 <= confidence <= 0.7