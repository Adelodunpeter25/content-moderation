"""Pydantic schemas for text moderation API."""
from pydantic import BaseModel
from typing import List, Optional

class ModerationRequest(BaseModel):
    """Request schema for text moderation."""
    text: str

class ToxicityResponse(BaseModel):
    """Response schema for toxicity detection."""
    is_toxic: bool
    confidence: float
    categories: List[str]

class ProfanityResponse(BaseModel):
    """Response schema for profanity detection."""
    has_profanity: bool
    filtered_text: str
    detected_words: List[str]

class SentimentResponse(BaseModel):
    """Response schema for sentiment analysis."""
    sentiment: str  # positive, negative, neutral
    confidence: float
    score: float  # -1 to 1

class ModerationResponse(BaseModel):
    """Combined moderation response."""
    text: str
    toxicity: ToxicityResponse
    profanity: ProfanityResponse
    sentiment: SentimentResponse
    overall_safe: bool