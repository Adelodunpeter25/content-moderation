"""FastAPI routes for spam classification."""
from fastapi import APIRouter
from spam_classifier.schemas.spam import TextRequest, SpamResponse
from spam_classifier.services.classifier import SpamClassifier

router = APIRouter(prefix="/spam", tags=["spam"])
classifier = SpamClassifier()

@router.post("/classify", response_model=SpamResponse)
def classify_text(request: TextRequest) -> SpamResponse:
    """Classify text as spam or not spam.
    
    Args:
        request: Text classification request
        
    Returns:
        Classification result with confidence score
    """
    is_spam, confidence = classifier.predict(request.text)
    return SpamResponse(
        is_spam=is_spam,
        confidence=confidence,
        text=request.text
    )