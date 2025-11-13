# Content Moderation System

A multi-app content moderation system built with FastAPI.

## Project Structure

```
content-moderation/
├── core/
│   ├── app/          # Main FastAPI application
│   ├── models/       # Shared database models
│   ├── schemas/      # Shared Pydantic schemas
│   └── utils/        # Shared utilities
├── spam-classifier/  # Spam detection module
│   ├── models/       # ML models
│   ├── routes/       # API endpoints
│   ├── schemas/      # Request/response schemas
│   └── services/     # Business logic
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn core.app.main:app --reload
```
