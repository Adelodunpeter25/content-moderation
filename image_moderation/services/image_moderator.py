"""Main image moderation service combining all detection capabilities."""
from typing import Dict, List, Optional
from pathlib import Path

from core.logging import logger
from .nsfw_detector import NSFWDetector
from .violence_detector import ViolenceDetector
from .face_detector import FaceDetector


class ImageModerator:
    """Comprehensive image moderation service."""
    
    def __init__(self):
        self.nsfw_detector = NSFWDetector()
        self.violence_detector = ViolenceDetector()
        self.face_detector = FaceDetector()
    
    def analyze_image(self, image_url: Optional[str] = None, 
                     image_base64: Optional[str] = None,
                     check_nsfw: bool = True,
                     check_violence: bool = True,
                     check_faces: bool = False) -> Dict:
        """Comprehensive image analysis.
        
        Args:
            image_url: URL of image to analyze
            image_base64: Base64 encoded image data
            check_nsfw: Whether to check for NSFW content
            check_violence: Whether to check for violence
            check_faces: Whether to detect faces
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Load image data
            if image_url:
                image_data = self.nsfw_detector.load_image_from_url(image_url)
            elif image_base64:
                image_data = self.nsfw_detector.load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            # Initialize results dynamically
            results = self._initialize_results(check_nsfw, check_violence, check_faces)
            
            # NSFW Detection
            if check_nsfw:
                is_nsfw, nsfw_confidence, nsfw_categories = self.nsfw_detector.detect_nsfw(image_data)
                results['nsfw_score'] = nsfw_confidence if is_nsfw else 0.0
                results['details'].update(nsfw_categories)
                
                if is_nsfw:
                    results['is_inappropriate'] = True
                    results['categories'].append('nsfw')
                    results['confidence'] = max(results['confidence'], nsfw_confidence)
            
            # Violence Detection
            if check_violence:
                is_violent, violence_confidence, violence_type = self.violence_detector.detect_violence(image_data)
                results['violence_score'] = violence_confidence if is_violent else 0.0
                results['details']['violence_type'] = violence_type
                
                if is_violent:
                    results['is_inappropriate'] = True
                    results['categories'].append('violence')
                    results['confidence'] = max(results['confidence'], violence_confidence)
            
            # Face Detection
            if check_faces:
                face_count, faces, has_minors = self.face_detector.detect_faces(image_data)
                results['face_count'] = face_count
                results['details']['faces'] = faces
                results['details']['has_minors'] = has_minors
                
                if has_minors:
                    results['categories'].append('minors_detected')
            
            # Calculate overall severity
            results['severity'] = self._calculate_severity(results)
            
            # Set default confidence if no issues found
            if results['confidence'] == 0.0:
                results['confidence'] = 0.95  # High confidence in safe content
            
            logger.info(f"Image analysis complete: inappropriate={results['is_inappropriate']}, "
                       f"categories={results['categories']}, confidence={results['confidence']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                'is_inappropriate': False,
                'confidence': 0.5,
                'categories': [],
                'nsfw_score': 0.0,
                'violence_score': 0.0,
                'face_count': 0,
                'severity': 'unknown',
                'details': {'error': str(e)}
            }
    
    def _initialize_results(self, check_nsfw: bool, check_violence: bool, check_faces: bool) -> Dict:
        """Initialize results structure based on enabled checks.
        
        Args:
            check_nsfw: Whether NSFW checking is enabled
            check_violence: Whether violence checking is enabled
            check_faces: Whether face detection is enabled
            
        Returns:
            Initialized results dictionary
        """
        results = {
            'is_inappropriate': False,
            'confidence': 0.0,
            'categories': [],
            'severity': 'safe',
            'details': {}
        }
        
        # Add fields based on enabled checks
        if check_nsfw:
            results['nsfw_score'] = 0.0
            results['details']['nsfw_categories'] = {}
        
        if check_violence:
            results['violence_score'] = 0.0
            results['details']['violence_type'] = 'none'
        
        if check_faces:
            results['face_count'] = 0
            results['details']['faces'] = []
            results['details']['has_minors'] = False
        
        return results
    
    def _calculate_severity(self, results: Dict) -> str:
        """Calculate content severity level.
        
        Args:
            results: Analysis results
            
        Returns:
            Severity level string
        """
        if not results['is_inappropriate']:
            return 'safe'
        
        max_score = max(results['nsfw_score'], results['violence_score'])
        
        if max_score >= 0.9:
            return 'critical'
        elif max_score >= 0.7:
            return 'high'
        elif max_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def detect_nsfw_only(self, image_url: Optional[str] = None, 
                        image_base64: Optional[str] = None) -> Dict:
        """NSFW detection only.
        
        Args:
            image_url: URL of image to analyze
            image_base64: Base64 encoded image data
            
        Returns:
            NSFW detection results
        """
        try:
            # Load image data
            if image_url:
                image_data = self.nsfw_detector.load_image_from_url(image_url)
            elif image_base64:
                image_data = self.nsfw_detector.load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            is_nsfw, confidence, categories = self.nsfw_detector.detect_nsfw(image_data)
            
            return {
                'is_nsfw': is_nsfw,
                'confidence': confidence,
                'categories': categories
            }
            
        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            return {
                'is_nsfw': False,
                'confidence': 0.5,
                'categories': {'safe': 1.0}
            }
    
    def detect_violence_only(self, image_url: Optional[str] = None,
                           image_base64: Optional[str] = None) -> Dict:
        """Violence detection only.
        
        Args:
            image_url: URL of image to analyze
            image_base64: Base64 encoded image data
            
        Returns:
            Violence detection results
        """
        try:
            # Load image data
            if image_url:
                image_data = self.nsfw_detector.load_image_from_url(image_url)
            elif image_base64:
                image_data = self.nsfw_detector.load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            is_violent, confidence, violence_type = self.violence_detector.detect_violence(image_data)
            
            return {
                'is_violent': is_violent,
                'confidence': confidence,
                'violence_type': violence_type
            }
            
        except Exception as e:
            logger.error(f"Error in violence detection: {e}")
            return {
                'is_violent': False,
                'confidence': 0.5,
                'violence_type': 'none'
            }
    
    def detect_faces_only(self, image_url: Optional[str] = None,
                         image_base64: Optional[str] = None) -> Dict:
        """Face detection only.
        
        Args:
            image_url: URL of image to analyze
            image_base64: Base64 encoded image data
            
        Returns:
            Face detection results
        """
        try:
            # Load image data
            if image_url:
                image_data = self.nsfw_detector.load_image_from_url(image_url)
            elif image_base64:
                image_data = self.nsfw_detector.load_image_from_base64(image_base64)
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            face_count, faces, has_minors = self.face_detector.detect_faces(image_data)
            
            return {
                'face_count': face_count,
                'faces': faces,
                'has_minors': has_minors
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return {
                'face_count': 0,
                'faces': [],
                'has_minors': False
            }