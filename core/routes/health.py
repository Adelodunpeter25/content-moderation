"""Health check and root endpoints."""
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/")
def root() -> dict[str, str]:
    """Root endpoint returning system information."""
    return {"message": "Content Moderation System"}

@router.get("/health")
def health() -> dict:
    """Health check endpoint with system status."""
    # Check model files
    model_exists = os.path.exists('data/spam_model.joblib')
    vectorizer_exists = os.path.exists('data/spam_vectorizer.joblib')
    
    # Check feedback data
    feedback_dir = Path('data/feedback')
    feedback_count = 0
    if feedback_dir.exists():
        feedback_file = feedback_dir / 'user_feedback.jsonl'
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                feedback_count = sum(1 for _ in f)
    
    # Check logs directory
    logs_exist = os.path.exists('logs')
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "components": {
            "spam_model": "ready" if model_exists else "not_trained",
            "vectorizer": "ready" if vectorizer_exists else "not_trained",
            "feedback_system": "active",
            "logging": "active" if logs_exist else "inactive"
        },
        "stats": {
            "feedback_count": feedback_count
        }
    }