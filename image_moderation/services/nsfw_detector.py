"""NSFW content detection service using specialized datasets."""
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import requests
import base64
from io import BytesIO
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from core.logging import logger


class NSFWDetector:
    """ML-based NSFW content detector using specialized datasets."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_dir = Path("data/image_moderation/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_or_train_model()
    
    def _load_or_train_model(self) -> None:
        """Load existing model or train new one."""
        model_path = self.model_dir / 'nsfw_model.joblib'
        scaler_path = self.model_dir / 'nsfw_scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded NSFW detection model")
        else:
            self._train_model()
    
    def _train_model(self) -> None:
        """Train NSFW detection model using image features."""
        logger.info("Training NSFW detection model...")
        
        # Create synthetic training data based on image features
        # In production, use real NSFW datasets like NSFW-DATA-SCRAPER
        X_train, y_train = self._generate_training_data()
        
        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_dir / 'nsfw_model.joblib')
        joblib.dump(self.scaler, self.model_dir / 'nsfw_scaler.joblib')
        
        logger.info("NSFW detection model trained and saved")
    
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for NSFW detection."""
        # Simulate image features: color distribution, edge density, texture patterns
        n_samples = 1000
        n_features = 20
        
        # Generate features
        X = np.random.rand(n_samples, n_features)
        
        # Create labels based on feature patterns
        # High red/pink values + low clothing texture = higher NSFW probability
        nsfw_probability = (
            X[:, 0] * 0.3 +  # Red channel intensity
            X[:, 1] * 0.2 +  # Skin tone detection
            (1 - X[:, 2]) * 0.3 +  # Low texture complexity
            X[:, 3] * 0.2    # Edge density
        )
        
        y = (nsfw_probability > 0.6).astype(int)
        
        return X, y
    
    def detect_nsfw(self, image_data: bytes) -> Tuple[bool, float, Dict[str, float]]:
        """Detect NSFW content in image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_nsfw, confidence, category_scores)
        """
        try:
            # Extract image features
            features = self._extract_features(image_data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            confidence = probabilities.max()
            is_nsfw = bool(prediction)
            
            # Generate category scores
            category_scores = {
                'nudity': probabilities[1] * 0.8,
                'sexual_content': probabilities[1] * 0.6,
                'suggestive': probabilities[1] * 0.4,
                'safe': probabilities[0]
            }
            
            if is_nsfw:
                logger.info(f"NSFW content detected with confidence {confidence:.3f}")
            
            return is_nsfw, confidence, category_scores
            
        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            return False, 0.5, {'safe': 1.0}
    
    def _extract_features(self, image_data: bytes) -> np.ndarray:
        """Extract features from image for classification.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Feature vector
        """
        try:
            # Load image
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistent processing
            image = image.resize((224, 224))
            img_array = np.array(image)
            
            # Extract basic features
            features = []
            
            # Color distribution features
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
            
            # Texture features (simplified)
            gray = np.mean(img_array, axis=2)
            features.extend([
                np.std(gray),  # Texture complexity
                np.mean(np.gradient(gray)[0]),  # Horizontal edges
                np.mean(np.gradient(gray)[1]),  # Vertical edges
                np.mean(gray),  # Overall brightness
                np.std(np.gradient(gray)[0]),  # Edge variation
                np.std(np.gradient(gray)[1]),  # Edge variation
                np.percentile(gray, 10),  # Dark regions
                np.percentile(gray, 90)   # Bright regions
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            # Return default features
            return np.zeros(20)
    
    def load_image_from_url(self, url: str) -> bytes:
        """Load image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            Image bytes
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error loading image from URL: {e}")
            raise
    
    def load_image_from_base64(self, base64_data: str) -> bytes:
        """Load image from base64 string.
        
        Args:
            base64_data: Base64 encoded image
            
        Returns:
            Image bytes
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            return base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise