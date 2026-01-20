# Wave 1: Foundation

Implement the core data models and foundation services.

## Components

### 1. Models Validation (`src/gram_deploy/models/`)
- Verify all Pydantic models have correct field types and validators
- Add any missing validation logic (ID format patterns, enum constraints)
- Ensure JSON serialization/deserialization works correctly
- Test model instantiation with sample data from `deployments/deploy_20250119_vinci_01/`

### 2. DeploymentManager (`src/gram_deploy/services/deployment_manager.py`)
Implement all methods:
- `create_deployment()` - Generate ID, create directory structure, write deployment.json
- `load_deployment()` - Read and parse deployment.json
- `save_deployment()` - Write updated deployment to disk
- `add_source()` - Create source directory, extract video metadata with ffprobe
- `list_deployments()` - Read deployments/index.json
- `update_status()` - Update deployment status with checkpoint support

### 3. AudioExtractor (`src/gram_deploy/services/audio_extractor.py`)
Implement:
- `extract_audio()` - Use ffmpeg to extract audio from video files
- `get_video_metadata()` - Use ffprobe to get duration, codec, resolution, fps
- Support for GoPro, phone, and drone video formats
- Write extracted audio to `sources/{source}/audio.wav`

## Testing
Create `tests/test_wave1.py` with:
- Model validation tests
- DeploymentManager CRUD operations
- AudioExtractor with sample video (mock ffmpeg if needed)

## Success Criteria
- `deploy create --location test --date 2025-01-20` works
- `deploy add-source` creates proper directory structure
- Audio extraction produces valid WAV files
