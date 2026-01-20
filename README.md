# GRAM Deployment Processing System

Transform field deployment footage into structured intelligence.

## Overview

Pipeline for processing multi-source video recordings into:
- Canonical timelines with synchronized multi-source video
- Unified transcripts with speaker identification
- Semantic analysis (events, action items, insights)
- Full-text search index
- Interactive visualizations and reports

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Create a deployment
deploy create --location "vinci" --date "2025-01-19"

# Add video sources
deploy add-source deploy:20250119_vinci_01 --type gopro --number 1 --files /path/to/*.MP4

# Process the deployment
deploy process deploy:20250119_vinci_01

# Search transcripts
deploy search "battery levels"

# Generate report
deploy report deploy:20250119_vinci_01 --format md --output report.md
```

## Architecture

```
src/gram_deploy/
├── models/          # Pydantic data models
├── services/        # Core processing services
│   ├── deployment_manager.py    # Deployment lifecycle
│   ├── audio_extractor.py       # ffmpeg audio extraction
│   ├── transcription_service.py # ElevenLabs/AssemblyAI/Deepgram
│   ├── time_alignment.py        # Multi-source synchronization
│   ├── speaker_resolution.py    # Speaker identification
│   ├── transcript_merger.py     # Canonical transcript creation
│   ├── semantic_analyzer.py     # LLM-based extraction
│   ├── search_index.py          # SQLite FTS5 search
│   ├── timeline_visualizer.py   # HTML visualizations
│   ├── report_generator.py      # Markdown/HTML/PDF reports
│   └── pipeline.py              # Orchestration
├── utils/           # Utility functions
└── __main__.py      # CLI interface
```

## API Keys Required

| Key | Purpose | Get at |
|-----|---------|--------|
| `ELEVENLABS_API_KEY` | Transcription | https://elevenlabs.io |
| `ANTHROPIC_API_KEY` | Semantic analysis | https://console.anthropic.com |

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Lint
ruff check src/
```

## License

Proprietary - GRAM Corporation
