"""Health check and root endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/")
def root() -> dict[str, str]:
    """Root endpoint returning system information."""
    return {"message": "Content Moderation System"}

@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}