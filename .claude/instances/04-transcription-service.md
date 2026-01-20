# TranscriptionService

Implement speech-to-text in `src/gram_deploy/services/transcription_service.py`.

## Architecture

Use S3 presigned URLs - ElevenLabs fetches the file directly, no local file handling needed.

```
Google Drive → S3 (via rclone) → Presigned URL → ElevenLabs fetches directly
```

## Tasks

1. `transcribe(s3_key_or_path, provider)` -> RawTranscript:
   - Generate S3 presigned URL for the audio/video file
   - Pass URL to provider API (they fetch it)
   - Return normalized RawTranscript model

2. `_generate_presigned_url(s3_key)` -> str:
   ```python
   import boto3
   s3 = boto3.client('s3')
   url = s3.generate_presigned_url('get_object',
       Params={'Bucket': BUCKET, 'Key': s3_key},
       ExpiresIn=3600)
   return url
   ```

3. `_transcribe_elevenlabs(presigned_url)`:
   - POST to `https://api.elevenlabs.io/v1/speech-to-text`
   - Use `url` parameter (ElevenLabs fetches from S3)
   - Request diarization (speaker separation)
   ```python
   response = requests.post(
       'https://api.elevenlabs.io/v1/speech-to-text',
       headers={'xi-api-key': API_KEY},
       json={'url': presigned_url}
   )
   ```
   - Parse response into utterances with speaker labels

4. `_transcribe_assemblyai(presigned_url)`:
   - Submit URL directly (AssemblyAI also supports URL input)
   - Poll for completion
   - Parse with speaker_labels enabled

5. `_transcribe_deepgram(presigned_url)`:
   - Submit URL to Deepgram API
   - Enable diarization
   - Parse response

6. Normalize output:
   - Convert provider-specific format to `RawTranscript`
   - Each utterance: text, start_ms, end_ms, speaker_id, confidence
   - Save to `{source_path}/raw_transcript.json`

## Environment Variables

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
GRAM_S3_BUCKET=your-bucket
ELEVENLABS_API_KEY=...
```

## Reference

- Sample transcript format: `deployments/deploy_20250119_vinci_01/sources/gopro_01/raw_transcript.json`
- S3 bucket structure mirrors `deployments/` directory

## Tests

Create `tests/test_transcription_service.py`:
- Mock boto3 presigned URL generation
- Mock API responses for each provider
- Test response parsing
- Test error handling (rate limits, failures)

## Files

- `src/gram_deploy/services/transcription_service.py`
- `tests/test_transcription_service.py`
