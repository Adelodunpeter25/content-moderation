"""Test violence detector service."""
import pytest
from PIL import Image
import io

from image_moderation.services.violence_detector import ViolenceDetector


class TestViolenceDetector:
    """Test violence detection service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ViolenceDetector()
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self) -> bytes:
        """Create test image data."""
        img = Image.new('RGB', (100, 100), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def test_detect_violence(self):
        """Test violence detection."""
        is_violent, confidence, violence_type = self.detector.detect_violence(self.test_image_data)
        
        assert isinstance(is_violent, bool)
        assert isinstance(confidence, float)
        assert isinstance(violence_type, str)
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_features(self):
        """Test violence feature extraction."""
        features = self.detector._extract_features(self.test_image_data)
        
        assert len(features) == 15
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_classify_violence_type(self):
        """Test violence type classification."""
        import numpy as np
        features = np.random.rand(15)
        
        violence_type = self.detector._classify_violence_type(features)
        
        assert isinstance(violence_type, str)
        assert violence_type in ['blood_gore', 'weapons', 'threatening', 'action_violence', 'violence_detected']
    
    def test_generate_fallback_data(self):
        """Test fallback data generation."""
        X, y = self.detector._generate_fallback_data()
        
        assert len(X) == 400
        assert len(y) == 400
        assert all(label in [0, 1] for label in y)