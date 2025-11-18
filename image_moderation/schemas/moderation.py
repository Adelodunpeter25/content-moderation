"""Image moderation request and response schemas."""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class ImageModerationRequest(BaseModel):
    """Request schema for image moderation."""
    
    image_url: Optional[str] = Field(None, description="URL of image to analyze")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    check_nsfw: bool = Field(True, description="Check for NSFW content")
    check_violence: bool = Field(True, description="Check for violent content")
    check_faces: bool = Field(False, description="Detect faces in image")


class ImageModerationResponse(BaseModel):
    """Response schema for image moderation."""
    
    is_inappropriate: bool = Field(description="Whether image contains inappropriate content")
    confidence: float = Field(description="Overall confidence score")
    categories: List[str] = Field(description="Detected inappropriate categories")
    nsfw_score: float = Field(description="NSFW content probability")
    violence_score: float = Field(description="Violence content probability")
    face_count: int = Field(description="Number of faces detected")
    severity: str = Field(description="Content severity level")
    details: Dict[str, float] = Field(description="Detailed category scores")


class NSFWDetectionResponse(BaseModel):
    """Response schema for NSFW detection."""
    
    is_nsfw: bool = Field(description="Whether image is NSFW")
    confidence: float = Field(description="Detection confidence")
    categories: Dict[str, float] = Field(description="NSFW category scores")


class ViolenceDetectionResponse(BaseModel):
    """Response schema for violence detection."""
    
    is_violent: bool = Field(description="Whether image contains violence")
    confidence: float = Field(description="Detection confidence")
    violence_type: str = Field(description="Type of violence detected")


class FaceDetectionResponse(BaseModel):
    """Response schema for face detection."""
    
    face_count: int = Field(description="Number of faces detected")
    faces: List[Dict[str, float]] = Field(description="Face bounding boxes and confidence")
    has_minors: bool = Field(description="Whether minors detected")