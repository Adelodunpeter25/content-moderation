# Content Moderation System

A multi-app content moderation system built with FastAPI.

## Features

### 🛡️ **Spam Classification**
- Advanced ML-based spam detection
- Multi-dataset training (SMS + YouTube comments)
- Text preprocessing and normalization
- User feedback and model retraining

### 🔧 **System Features**
- Comprehensive logging
- Health monitoring
- API documentation
- Modular architecture for easy expansion

## Project Structure

```
content-moderation/
├── core/
│   ├── app/          # Main FastAPI application
│   ├── routes/       # Health and system endpoints
│   └── logging.py    # Central logging configuration
├── spam_classifier/
│   ├── routes/       # Spam classification & feedback APIs
│   ├── schemas/      # Request/response models
│   └── services/     # ML services and business logic
├── data/             # Model files and datasets
├── logs/             # Application logs
└── pyproject.toml    # Dependencies and configuration
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd content-moderation

# Install dependencies
uv sync
```

### Running

```bash
# Development server
make dev

# Production server
make run

# View all commands
make help
```

## API Endpoints

- `GET /health` - System health status
- `POST /api/v1/spam/classify` - Classify text as spam
- `POST /api/v1/feedback/submit` - Submit user feedback
- `POST /api/v1/feedback/retrain` - Retrain model
