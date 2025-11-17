"""FastAPI routes for feedback and retraining."""
from fastapi import APIRouter, HTTPException
from spam_classifier.schemas.feedback import FeedbackRequest, FeedbackResponse, RetrainRequest, RetrainResponse
from spam_classifier.services.classifier import SpamClassifier
from core.logging import logger

router = APIRouter(prefix="/feedback", tags=["Feedback"])
classifier = SpamClassifier()

@router.post("/submit", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Submit user feedback about classification accuracy.
    
    Args:
        request: Feedback about a classification
        
    Returns:
        Confirmation of feedback recording
    """
    try:
        classifier.record_feedback(
            text=request.text,
            predicted_spam=request.predicted_spam,
            actual_spam=request.actual_spam,
            confidence=request.confidence,
            user_id=request.user_id
        )
        
        return FeedbackResponse(
            message="Feedback recorded successfully",
            feedback_recorded=True
        )
    except Exception as e:
        logger.error(f"Feedback recording error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@router.post("/retrain", response_model=RetrainResponse)
def retrain_model(request: RetrainRequest) -> RetrainResponse:
    """Retrain the model with user feedback.
    
    Args:
        request: Retraining parameters
        
    Returns:
        Retraining results and metrics
    """
    try:
        metrics = classifier.retrain_model()
        
        if metrics.get('status') == 'insufficient_feedback':
            return RetrainResponse(
                status="insufficient_feedback",
                message=f"Need more feedback samples. Current: {metrics['samples']}",
                metrics=metrics
            )
        
        return RetrainResponse(
            status="success",
            message="Model retrained successfully",
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail="Retraining failed")