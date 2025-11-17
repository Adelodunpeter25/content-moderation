"""Pydantic schemas for spam classification API."""
from pydantic import BaseModel

class TextRequest(BaseModel):
    """Request schema for text classification."""
    text: str

class SpamResponse(BaseModel):
    """Response schema for spam classification results."""
    is_spam: bool
    confidence: float
    text: str