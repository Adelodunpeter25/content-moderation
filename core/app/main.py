"""Main FastAPI application for content moderation system."""
from fastapi import FastAPI
from core.logging import logger
from core.routes import health
from spam_classifier.routes import spam

app = FastAPI(
    title="Content Moderation System",
    description="Multi-app content moderation system",
    version="0.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

app.include_router(health.router)
app.include_router(spam.router, prefix="/api/v1")

logger.info("Content Moderation System started")