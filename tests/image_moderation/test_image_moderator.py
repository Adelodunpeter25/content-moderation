"""Test image moderator service."""
import pytest
from unittest.mock import Mock, patch
from PIL import Image
import io
import base64

from image_moderation.services.image_moderator import ImageModerator


class TestImageModerator:
    """Test image moderation service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = ImageModerator()
        self.test_image_data = self._create_test_image()
    
    def _create_test_image(self) -> bytes:
        """Create test image data."""
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    def _create_test_base64(self) -> str:
        """Create test base64 image."""
        return base64.b64encode(self.test_image_data).decode()
    
    @patch('image_moderation.services.nsfw_detector.NSFWDetector.detect_nsfw')
    @patch('image_moderation.services.violence_detector.ViolenceDetector.detect_violence')
    @patch('image_moderation.services.face_detector.FaceDetector.detect_faces')
    def test_analyze_image_comprehensive(self, mock_faces, mock_violence, mock_nsfw):
        """Test comprehensive image analysis."""
        mock_nsfw.return_value = (True, 0.8, {'nsfw': 0.8, 'safe': 0.2})
        mock_violence.return_value = (False, 0.3, 'none')
        mock_faces.return_value = (2, [{'x': 10, 'y': 10}], False)
        
        with patch.object(self.moderator.nsfw_detector, 'load_image_from_base64', return_value=self.test_image_data):
            result = self.moderator.analyze_image(
                image_base64=self._create_test_base64(),
                check_nsfw=True,
                check_violence=True,
                check_faces=True
            )
        
        assert result['is_inappropriate'] is True
        assert 'nsfw' in result['categories']
        assert result['nsfw_score'] == 0.8
        assert result['violence_score'] == 0.0
        assert result['face_count'] == 2
    
    @patch('image_moderation.services.nsfw_detector.NSFWDetector.detect_nsfw')
    def test_detect_nsfw_only(self, mock_nsfw):
        """Test NSFW detection only."""
        mock_nsfw.return_value = (True, 0.9, {'nudity': 0.9})
        
        with patch.object(self.moderator.nsfw_detector, 'load_image_from_base64', return_value=self.test_image_data):
            result = self.moderator.detect_nsfw_only(image_base64=self._create_test_base64())
        
        assert result['is_nsfw'] is True
        assert result['confidence'] == 0.9
        assert 'nudity' in result['categories']
    
    def test_analyze_image_no_input(self):
        """Test analysis with no image input."""
        with pytest.raises(ValueError):
            self.moderator.analyze_image()
    
    def test_get_severity_thresholds(self):
        """Test severity threshold calculation."""
        thresholds = self.moderator._get_severity_thresholds()
        
        assert isinstance(thresholds, dict)
        assert 'critical' in thresholds
        assert 'high' in thresholds
        assert 'medium' in thresholds
        assert 'low' in thresholds
        assert all(0.0 <= v <= 1.0 for v in thresholds.values())
    
    def test_get_safe_confidence_score(self):
        """Test safe confidence score calculation."""
        confidence = self.moderator._get_safe_confidence_score()
        
        assert isinstance(confidence, float)
        assert 0.8 <= confidence <= 0.99