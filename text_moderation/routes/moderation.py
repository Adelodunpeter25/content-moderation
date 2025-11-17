"""FastAPI routes for text moderation."""
from fastapi import APIRouter, HTTPException

from text_moderation.schemas.moderation import (
    ModerationRequest, ModerationResponse, ToxicityResponse, 
    ProfanityResponse, SentimentResponse
)
from text_moderation.services.text_moderator import TextModerator
from core.logging import logger

router = APIRouter(prefix="/text", tags=["Text-moderation"])
moderator = TextModerator()

@router.post("/moderate", response_model=ModerationResponse)
def moderate_text(request: ModerationRequest) -> ModerationResponse:
    """Perform comprehensive text moderation analysis.
    
    Args:
        request: Text moderation request
        
    Returns:
        Complete moderation analysis results
    """
    try:
        result = moderator.moderate_text(request.text)
        
        return ModerationResponse(
            text=result["text"],
            toxicity=ToxicityResponse(**result["toxicity"]),
            profanity=ProfanityResponse(**result["profanity"]),
            sentiment=SentimentResponse(**result["sentiment"]),
            overall_safe=result["overall_safe"]
        )
    except Exception as e:
        logger.error(f"Text moderation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Text moderation failed")

@router.post("/toxicity", response_model=ToxicityResponse)
def detect_toxicity(request: ModerationRequest) -> ToxicityResponse:
    """Detect toxicity in text.
    
    Args:
        request: Text to analyze
        
    Returns:
        Toxicity detection results
    """
    try:
        is_toxic, confidence, categories = moderator.toxicity_detector.detect_toxicity(request.text)
        
        return ToxicityResponse(
            is_toxic=is_toxic,
            confidence=confidence,
            categories=categories
        )
    except Exception as e:
        logger.error(f"Toxicity detection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Toxicity detection failed")

@router.post("/profanity", response_model=ProfanityResponse)
def filter_profanity(request: ModerationRequest) -> ProfanityResponse:
    """Filter profanity from text.
    
    Args:
        request: Text to filter
        
    Returns:
        Profanity filtering results
    """
    try:
        has_profanity, filtered_text, detected_words = moderator.profanity_filter.filter_profanity(request.text)
        
        return ProfanityResponse(
            has_profanity=has_profanity,
            filtered_text=filtered_text,
            detected_words=detected_words
        )
    except Exception as e:
        logger.error(f"Profanity filtering error: {str(e)}")
        raise HTTPException(status_code=500, detail="Profanity filtering failed")

@router.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: ModerationRequest) -> SentimentResponse:
    """Analyze sentiment of text.
    
    Args:
        request: Text to analyze
        
    Returns:
        Sentiment analysis results
    """
    try:
        sentiment, confidence, score = moderator.sentiment_analyzer.analyze_sentiment(request.text)
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            score=score
        )
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")