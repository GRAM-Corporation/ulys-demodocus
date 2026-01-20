# TranscriptionService

Implement speech-to-text in `src/gram_deploy/services/transcription_service.py`.

## Tasks

1. `transcribe(audio_path, provider)` -> RawTranscript:
   - Route to appropriate provider
   - Return normalized RawTranscript model

2. `_transcribe_elevenlabs(audio_path)`:
   - POST to ElevenLabs speech-to-text API
   - Request diarization (speaker separation)
   - Parse response into utterances with speaker labels

3. `_transcribe_assemblyai(audio_path)`:
   - Upload audio, start transcription job
   - Poll for completion
   - Parse with speaker_labels enabled

4. `_transcribe_deepgram(audio_path)`:
   - Stream or batch transcription
   - Enable diarization
   - Parse response

5. Normalize output:
   - Convert provider-specific format to `RawTranscript`
   - Each utterance: text, start_ms, end_ms, speaker_id, confidence
   - Save to `{source_path}/raw_transcript.json`

Reference format: `deployments/deploy_20250119_vinci_01/sources/gopro_01/raw_transcript.json`

## Tests
Create `tests/test_transcription_service.py`:
- Mock API responses for each provider
- Test response parsing
- Test error handling (rate limits, failures)

## Files
- `src/gram_deploy/services/transcription_service.py`
- `tests/test_transcription_service.py`
