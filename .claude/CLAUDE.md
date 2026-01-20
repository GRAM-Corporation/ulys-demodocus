# GRAM Deployment Processing System - Claude Code Instructions

## Project Overview

This is the GRAM Deployment Processing System - a pipeline for transforming field deployment footage into structured intelligence. It processes video from multiple sources (GoPros, phones, drones) into:
- Canonical timelines with aligned multi-source video
- Unified transcripts with speaker identification
- Semantic analysis (events, action items, insights)
- Searchable indexes and interactive visualizations
- Professional deployment reports

## Architecture

```
src/gram_deploy/
├── models/          # Pydantic data models
├── services/        # Core processing services
│   ├── deployment_manager.py    # Deployment lifecycle
│   ├── audio_extractor.py       # ffmpeg audio extraction
│   ├── transcription_service.py # ElevenLabs/AssemblyAI API
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

## Key Data Models

- **Deployment**: Top-level container (ID format: `deploy:YYYYMMDD_location_NN`)
- **Source**: Video recording device and files
- **RawTranscript**: Unprocessed transcription output
- **CanonicalUtterance**: Aligned, speaker-resolved speech segment
- **DeploymentEvent**: Significant timeline events
- **ActionItem**: Extracted tasks and follow-ups
- **DeploymentInsight**: Semantic observations

## Implementation Guidelines

1. **All services are in `src/gram_deploy/services/`** - implement the full interface as specified
2. **Use Pydantic models** for all data structures - see `src/gram_deploy/models/`
3. **Tests go in `tests/`** - use pytest with the patterns described in the spec
4. **Data is stored in `deployments/`** directory structure per deployment
5. **Cache intermediate results** to support resumable processing

## API Keys Required

- `ELEVENLABS_API_KEY` - for transcription
- `ANTHROPIC_API_KEY` - for semantic analysis

## Running the CLI

```bash
# Create a deployment
deploy create --location "vinci" --date "2025-01-19"

# Add video source
deploy add-source deploy:20250119_vinci_01 --type gopro --number 1 --files /path/to/*.MP4

# Process
deploy process deploy:20250119_vinci_01

# Search
deploy search "Starlink battery"

# Generate report
deploy report deploy:20250119_vinci_01 --format md --output report.md
```

## Wave-Based Implementation

The system should be implemented in waves as specified:

**Wave 1 (Foundation)**: Models, DeploymentManager, AudioExtractor
**Wave 2 (Transcription)**: TranscriptionService, TimeAlignmentService, SpeakerResolutionService
**Wave 3 (Merging/Analysis)**: TranscriptMerger, SemanticAnalyzer, SearchIndexBuilder
**Wave 4 (Output)**: TimelineVisualizer, ReportGenerator, PipelineOrchestrator, CLI

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Each component should have tests covering:
- Happy path functionality
- Edge cases and error handling
- Integration with dependent components
