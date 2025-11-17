"""FastAPI routes for spam classification."""
from fastapi import APIRouter, HTTPException
from spam_classifier.schemas.spam import TextRequest, SpamResponse
from spam_classifier.schemas.feedback import FeedbackRequest, FeedbackResponse, RetrainRequest, RetrainResponse
from spam_classifier.services.classifier import SpamClassifier
from core.logging import logger

router = APIRouter(prefix="/spam", tags=["Spam"])
classifier = SpamClassifier()

@router.post("/classify", response_model=SpamResponse)
def classify_text(request: TextRequest) -> SpamResponse:
    """Classify text as spam or not spam.
    
    Args:
        request: Text classification request
        
    Returns:
        Classification result with confidence score
    """
    try:
        is_spam, confidence = classifier.predict(request.text)
        logger.info(f"Classified text: spam={is_spam}, confidence={confidence:.3f}")
        return SpamResponse(
            is_spam=is_spam,
            confidence=confidence,
            text=request.text
        )
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification failed")