"""Image moderation API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict

from core.logging import logger
from ..schemas.moderation import (
    ImageModerationRequest, 
    ImageModerationResponse,
    NSFWDetectionResponse,
    ViolenceDetectionResponse,
    FaceDetectionResponse
)
from ..services.image_moderator import ImageModerator

router = APIRouter(prefix="/api/v1/image", tags=["image_moderation"])

# Initialize image moderator
image_moderator = ImageModerator()


@router.post("/analyze", response_model=ImageModerationResponse)
async def analyze_image(request: ImageModerationRequest) -> ImageModerationResponse:
    """Comprehensive image analysis for inappropriate content.
    
    Args:
        request: Image moderation request
        
    Returns:
        Comprehensive analysis results
    """
    try:
        logger.info(f"Image analysis request: nsfw={request.check_nsfw}, "
                   f"violence={request.check_violence}, faces={request.check_faces}")
        
        # Perform analysis
        results = image_moderator.analyze_image(
            image_url=request.image_url,
            image_base64=request.image_base64,
            check_nsfw=request.check_nsfw,
            check_violence=request.check_violence,
            check_faces=request.check_faces
        )
        
        return ImageModerationResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/nsfw", response_model=NSFWDetectionResponse)
async def detect_nsfw(request: ImageModerationRequest) -> NSFWDetectionResponse:
    """Detect NSFW content in image.
    
    Args:
        request: Image moderation request
        
    Returns:
        NSFW detection results
    """
    try:
        logger.info("NSFW detection request")
        
        results = image_moderator.detect_nsfw_only(
            image_url=request.image_url,
            image_base64=request.image_base64
        )
        
        return NSFWDetectionResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in NSFW detection: {e}")
        raise HTTPException(status_code=500, detail=f"NSFW detection failed: {str(e)}")


@router.post("/violence", response_model=ViolenceDetectionResponse)
async def detect_violence(request: ImageModerationRequest) -> ViolenceDetectionResponse:
    """Detect violent content in image.
    
    Args:
        request: Image moderation request
        
    Returns:
        Violence detection results
    """
    try:
        logger.info("Violence detection request")
        
        results = image_moderator.detect_violence_only(
            image_url=request.image_url,
            image_base64=request.image_base64
        )
        
        return ViolenceDetectionResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in violence detection: {e}")
        raise HTTPException(status_code=500, detail=f"Violence detection failed: {str(e)}")


@router.post("/faces", response_model=FaceDetectionResponse)
async def detect_faces(request: ImageModerationRequest) -> FaceDetectionResponse:
    """Detect faces in image.
    
    Args:
        request: Image moderation request
        
    Returns:
        Face detection results
    """
    try:
        logger.info("Face detection request")
        
        results = image_moderator.detect_faces_only(
            image_url=request.image_url,
            image_base64=request.image_base64
        )
        
        return FaceDetectionResponse(**results)
        
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")