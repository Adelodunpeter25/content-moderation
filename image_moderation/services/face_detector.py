"""Face detection service for privacy and age verification."""
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from io import BytesIO

from core.logging import logger


class FaceDetector:
    """Face detection service using basic image processing."""
    
    def __init__(self):
        self.min_face_size = 20
        self.confidence_threshold = 0.5
    
    def detect_faces(self, image_data: bytes) -> Tuple[int, List[Dict[str, float]], bool]:
        """Detect faces in image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (face_count, face_boxes, has_minors)
        """
        try:
            # Load image
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Simple face detection using basic image processing
            faces = self._detect_face_regions(image)
            
            # Analyze faces for age estimation
            has_minors = self._estimate_minors(faces, image)
            
            face_count = len(faces)
            
            if face_count > 0:
                logger.info(f"Detected {face_count} faces, minors: {has_minors}")
            
            return face_count, faces, has_minors
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return 0, [], False
    
    def _detect_face_regions(self, image: Image.Image) -> List[Dict[str, float]]:
        """Detect face regions using basic image processing."""
        # Convert to grayscale
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Simple face detection using template matching approach
        faces = []
        
        # Look for face-like regions (simplified approach)
        height, width = img_array.shape
        
        # Scan image in overlapping windows
        window_sizes = [50, 75, 100, 150]
        
        for window_size in window_sizes:
            if window_size > min(height, width) // 2:
                continue
                
            step = window_size // 4
            
            for y in range(0, height - window_size, step):
                for x in range(0, width - window_size, step):
                    window = img_array[y:y+window_size, x:x+window_size]
                    
                    # Check if region looks like a face
                    if self._is_face_like(window):
                        confidence = self._calculate_face_confidence(window)
                        
                        if confidence > self.confidence_threshold:
                            faces.append({
                                'x': float(x),
                                'y': float(y),
                                'width': float(window_size),
                                'height': float(window_size),
                                'confidence': confidence
                            })
        
        # Remove overlapping detections
        faces = self._remove_overlaps(faces)
        
        return faces
    
    def _is_face_like(self, window: np.ndarray) -> bool:
        """Check if image window looks like a face."""
        if window.size == 0:
            return False
        
        # Basic face-like characteristics
        height, width = window.shape
        
        # Check aspect ratio (faces are roughly square to slightly tall)
        aspect_ratio = height / width
        if not (0.8 <= aspect_ratio <= 1.4):
            return False
        
        # Check for face-like intensity patterns
        # Eyes region (upper third) should be darker than forehead
        upper_third = window[:height//3, :]
        middle_third = window[height//3:2*height//3, :]
        
        if np.mean(upper_third) <= np.mean(middle_third):
            return False
        
        # Check for symmetry (basic)
        left_half = window[:, :width//2]
        right_half = np.fliplr(window[:, width//2:])
        
        if right_half.shape != left_half.shape:
            return False
        
        symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
        
        return symmetry > 0.6
    
    def _calculate_face_confidence(self, window: np.ndarray) -> float:
        """Calculate confidence score for face detection."""
        if window.size == 0:
            return 0.0
        
        height, width = window.shape
        
        # Multiple factors contribute to confidence
        confidence_factors = []
        
        # Aspect ratio score
        aspect_ratio = height / width
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        confidence_factors.append(max(0, aspect_score))
        
        # Intensity variation (faces have good contrast)
        intensity_var = np.std(window) / 255.0
        confidence_factors.append(min(1.0, intensity_var * 2))
        
        # Edge density (faces have moderate edges)
        edges = np.abs(np.gradient(window.astype(float)))
        edge_density = np.mean(edges) / 255.0
        confidence_factors.append(min(1.0, edge_density * 3))
        
        # Symmetry score
        left_half = window[:, :width//2]
        right_half = np.fliplr(window[:, width//2:])
        
        if right_half.shape == left_half.shape:
            symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
            confidence_factors.append(max(0, symmetry))
        
        return np.mean(confidence_factors)
    
    def _remove_overlaps(self, faces: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Remove overlapping face detections."""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
        
        filtered_faces = []
        
        for face in faces:
            # Check if this face overlaps significantly with any already accepted face
            overlaps = False
            
            for accepted_face in filtered_faces:
                if self._calculate_overlap(face, accepted_face) > 0.3:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def _calculate_overlap(self, face1: Dict[str, float], face2: Dict[str, float]) -> float:
        """Calculate overlap ratio between two face boxes."""
        # Calculate intersection
        x1 = max(face1['x'], face2['x'])
        y1 = max(face1['y'], face2['y'])
        x2 = min(face1['x'] + face1['width'], face2['x'] + face2['width'])
        y2 = min(face1['y'] + face1['height'], face2['y'] + face2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = face1['width'] * face1['height']
        area2 = face2['width'] * face2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_minors(self, faces: List[Dict[str, float]], image: Image.Image) -> bool:
        """Estimate if any detected faces belong to minors."""
        if not faces:
            return False
        
        img_array = np.array(image.convert('L'))
        
        for face in faces:
            # Extract face region
            x, y, w, h = int(face['x']), int(face['y']), int(face['width']), int(face['height'])
            face_region = img_array[y:y+h, x:x+w]
            
            if face_region.size == 0:
                continue
            
            # Simple heuristics for age estimation
            # Younger faces tend to have:
            # - Smoother texture (less variation)
            # - Rounder proportions
            # - Different intensity patterns
            
            texture_smoothness = 1.0 - (np.std(face_region) / 255.0)
            
            # Check face proportions (simplified)
            face_height, face_width = face_region.shape
            roundness = min(face_width, face_height) / max(face_width, face_height)
            
            # Combine factors
            youth_score = (texture_smoothness * 0.6 + roundness * 0.4)
            
            # If youth score is high, likely a minor
            if youth_score > 0.7:
                return True
        
        return False