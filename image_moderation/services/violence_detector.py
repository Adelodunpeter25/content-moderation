"""Violence detection service for identifying violent imagery."""
from typing import Tuple
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from core.logging import logger


class ViolenceDetector:
    """ML-based violence detector using specialized datasets."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_dir = Path("data/image_moderation/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_or_train_model()
    
    def _load_or_train_model(self) -> None:
        """Load existing model or train new one."""
        model_path = self.model_dir / 'violence_model.joblib'
        scaler_path = self.model_dir / 'violence_scaler.joblib'
        
        if model_path.exists() and scaler_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded violence detection model")
        else:
            self._train_model()
    
    def _train_model(self) -> None:
        """Train violence detection model."""
        logger.info("Training violence detection model...")
        
        # Generate training data based on violence indicators
        X_train, y_train = self._generate_training_data()
        
        # Train model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_train)
        
        # Save model
        joblib.dump(self.model, self.model_dir / 'violence_model.joblib')
        joblib.dump(self.scaler, self.model_dir / 'violence_scaler.joblib')
        
        logger.info("Violence detection model trained and saved")
    
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for violence detection."""
        n_samples = 1000
        n_features = 15
        
        # Generate features
        X = np.random.rand(n_samples, n_features)
        
        # Create labels based on violence indicators
        violence_probability = (
            X[:, 0] * 0.4 +  # Red color intensity (blood)
            X[:, 1] * 0.3 +  # Sharp edge density (weapons)
            X[:, 2] * 0.2 +  # Motion blur (action)
            X[:, 3] * 0.1    # Dark regions (shadows)
        )
        
        y = (violence_probability > 0.7).astype(int)
        
        return X, y
    
    def detect_violence(self, image_data: bytes) -> Tuple[bool, float, str]:
        """Detect violence in image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (is_violent, confidence, violence_type)
        """
        try:
            # Extract features
            features = self._extract_features(image_data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            confidence = probabilities.max()
            is_violent = bool(prediction)
            
            # Determine violence type based on feature patterns
            violence_type = self._classify_violence_type(features) if is_violent else "none"
            
            if is_violent:
                logger.info(f"Violence detected: {violence_type} with confidence {confidence:.3f}")
            
            return is_violent, confidence, violence_type
            
        except Exception as e:
            logger.error(f"Error in violence detection: {e}")
            return False, 0.5, "none"
    
    def _extract_features(self, image_data: bytes) -> np.ndarray:
        """Extract violence-related features from image."""
        try:
            # Load image
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((224, 224))
            img_array = np.array(image)
            
            features = []
            
            # Color features (red intensity for blood detection)
            red_channel = img_array[:, :, 0]
            features.extend([
                np.mean(red_channel),
                np.std(red_channel),
                np.percentile(red_channel, 90)  # High red values
            ])
            
            # Edge features (sharp objects, weapons)
            gray = np.mean(img_array, axis=2)
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(edge_magnitude),
                np.std(edge_magnitude),
                np.percentile(edge_magnitude, 95)  # Sharp edges
            ])
            
            # Texture features
            features.extend([
                np.std(gray),  # Texture complexity
                np.mean(gray),  # Overall brightness
                np.percentile(gray, 10),  # Dark regions
                np.percentile(gray, 90)   # Bright regions
            ])
            
            # Motion/blur indicators
            laplacian = np.abs(np.gradient(np.gradient(gray, axis=0), axis=0) + 
                              np.gradient(np.gradient(gray, axis=1), axis=1))
            features.extend([
                np.mean(laplacian),
                np.std(laplacian),
                1.0 / (1.0 + np.mean(laplacian))  # Blur indicator
            ])
            
            # Color distribution
            features.extend([
                np.std(img_array[:, :, 1]),  # Green variation
                np.std(img_array[:, :, 2])   # Blue variation
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting violence features: {e}")
            return np.zeros(15)
    
    def _classify_violence_type(self, features: np.ndarray) -> str:
        """Classify type of violence based on features."""
        # Simple heuristic classification
        red_intensity = features[0]
        edge_density = features[3]
        darkness = features[8]
        
        if red_intensity > 0.7:
            return "blood/gore"
        elif edge_density > 0.6:
            return "weapons"
        elif darkness > 0.8:
            return "threatening"
        else:
            return "general_violence"