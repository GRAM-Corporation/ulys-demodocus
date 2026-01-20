# Wave 2: Transcription Pipeline

Implement transcription and time alignment services.

## Components

### 1. TranscriptionService (`src/gram_deploy/services/transcription_service.py`)
Implement provider integrations:
- `transcribe()` - Main entry point, routes to configured provider
- `_transcribe_elevenlabs()` - ElevenLabs API with diarization
- `_transcribe_assemblyai()` - AssemblyAI API with speaker labels
- `_transcribe_deepgram()` - Deepgram API alternative
- Parse responses into `RawTranscript` model
- Save to `sources/{source}/raw_transcript.json`

Reference existing transcript format: `deployments/deploy_20250119_vinci_01/sources/gopro_01/raw_transcript.json`

### 2. TimeAlignmentService (`src/gram_deploy/services/time_alignment.py`)
Implement multi-source synchronization:
- `align_sources()` - Main alignment orchestrator
- `_align_by_audio_fingerprint()` - Use chromaprint for audio matching
- `_align_by_transcript()` - Match overlapping speech patterns
- `_align_by_metadata()` - Use video timestamps as fallback
- Calculate `canonical_offset_ms` for each source
- Set `alignment_confidence` score (0.0-1.0)

### 3. SpeakerResolutionService (`src/gram_deploy/services/speaker_resolution.py`)
Map raw speaker IDs to known people:
- `resolve_speakers()` - Main resolution pipeline
- `_match_by_voice_embedding()` - Compare with stored voice samples
- `_match_by_context()` - Use name mentions, role references
- `_match_by_pattern()` - Speaker frequency, conversation patterns
- Output `SpeakerMapping` entries with confidence scores
- Reference `deployments/people.json` for known team members

## Testing
Create `tests/test_wave2.py` with:
- Mock API responses for transcription providers
- Alignment algorithm unit tests
- Speaker resolution with sample transcript

## Dependencies
- Requires Wave 1 complete (AudioExtractor for audio files)
- Use existing raw_transcript.json for testing alignment/speaker resolution
