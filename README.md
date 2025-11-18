# Content Moderation System

A multi-app content moderation system built with FastAPI.

## Features

### 🛡️ **Spam Classification**
- Advanced ML-based spam detection
- Multi-dataset training
- Text preprocessing and normalization
- User feedback and model retraining

### 🚫 **Text Moderation**
- ML-based toxicity detection using Jigsaw datasets
- Hate speech identification with HatEval data
- Offensive language filtering with OffensEval
- Sentiment analysis using Stanford Sentiment Treebank
- Dynamic severity scoring based on model statistics
- Multi-label content classification

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
├── text_moderation/
│   ├── routes/       # Text moderation APIs
│   ├── schemas/      # Request/response models
│   └── services/     # ML-based moderation services
├── data/             # Model files and datasets
├── tests/            # Comprehensive test suite
├── logs/             # Application logs
└── pyproject.toml    # Dependencies and configuration
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Adelodunpeter25/content-moderation.git
cd content-moderation

# Install dependencies
uv sync
```

### Running

```bash
# Development server
make dev

# View all commands
make help
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/spam_classifier/
uv run pytest tests/text_moderation/

# Run with coverage
uv run pytest --cov=.
```
