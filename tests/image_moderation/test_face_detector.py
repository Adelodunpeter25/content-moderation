"""Test face detector service."""
import pytest
from PIL import Image
import io

from image_moderation.services.face_detector import FaceDetector


class TestFaceDetector:
    """Test face detection service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self) -> bytes:
        """Create test image data."""
        img = Image.new('RGB', (200, 200), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def test_detect_faces(self):
        """Test face detection."""
        face_count, faces, has_minors = self.detector.detect_faces(self.test_image_data)
        
        assert isinstance(face_count, int)
        assert isinstance(faces, list)
        assert isinstance(has_minors, bool)
        assert face_count >= 0
    
    def test_calculate_overlap(self):
        """Test face overlap calculation."""
        face1 = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
        face2 = {'x': 30, 'y': 30, 'width': 50, 'height': 50}
        
        overlap = self.detector._calculate_overlap(face1, face2)
        
        assert isinstance(overlap, float)
        assert 0.0 <= overlap <= 1.0
    
    def test_calculate_optimal_threshold(self):
        """Test optimal threshold calculation."""
        threshold = self.detector._calculate_optimal_threshold()
        
        assert isinstance(threshold, float)
        assert 0.3 <= threshold <= 0.8
    
    def test_calculate_age_threshold(self):
        """Test age threshold calculation."""
        threshold = self.detector._calculate_age_threshold()
        
        assert isinstance(threshold, float)
        assert 0.5 <= threshold <= 0.9
    
    def test_is_face_like(self):
        """Test face-like detection."""
        import numpy as np
        
        # Create face-like window
        window = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        result = self.detector._is_face_like(window)
        
        assert isinstance(result, bool)
    
    def test_calculate_face_confidence(self):
        """Test face confidence calculation."""
        import numpy as np
        
        window = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        confidence = self.detector._calculate_face_confidence(window)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0