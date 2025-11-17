"""Pydantic schemas for feedback and retraining API."""
from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    """Request schema for user feedback."""
    text: str
    predicted_spam: bool
    actual_spam: bool
    confidence: float
    user_id: Optional[str] = "anonymous"

class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""
    message: str
    feedback_recorded: bool

class RetrainRequest(BaseModel):
    """Request schema for model retraining."""
    min_feedback_samples: Optional[int] = 10

class RetrainResponse(BaseModel):
    """Response schema for retraining results."""
    status: str
    message: str
    metrics: Optional[dict] = None