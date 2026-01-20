# TranscriptionService

Implement speech-to-text in `src/gram_deploy/services/transcription_service.py`.

## Architecture

Pure S3 presigned URL flow - no local audio extraction. ElevenLabs fetches video directly from S3.

```
Google Drive → S3 (rclone sync) → Presigned URL → ElevenLabs fetches & transcribes
```

**No AudioExtractor dependency.** Transcription services accept video files directly.

## Tasks

1. `transcribe(source: Source, provider: str)` -> RawTranscript:
   - Get S3 key from source metadata
   - Generate presigned URL
   - Submit to provider API
   - Poll for completion (if async)
   - Parse and return RawTranscript

2. `_get_s3_key(source: Source)` -> str:
   - Map source to S3 bucket path
   - Convention: `deployments/{deployment_id}/sources/{source_name}/{filename}`

3. `_generate_presigned_url(s3_key: str, expires_in: int = 3600)` -> str:
   ```python
   import boto3
   s3 = boto3.client('s3')
   return s3.generate_presigned_url('get_object',
       Params={'Bucket': settings.s3_bucket, 'Key': s3_key},
       ExpiresIn=expires_in)
   ```

4. `_transcribe_elevenlabs(url: str)` -> RawTranscript:
   ```python
   response = requests.post(
       'https://api.elevenlabs.io/v1/speech-to-text',
       headers={'xi-api-key': settings.elevenlabs_api_key},
       json={
           'url': url,
           'diarization': True,  # speaker separation
           'timestamps': True
       }
   )
   ```
   - Parse response into RawTranscript model
   - Map speaker labels to TranscriptSpeaker entries

5. `_transcribe_assemblyai(url: str)` -> RawTranscript:
   - POST to AssemblyAI with URL
   - Enable `speaker_labels=True`
   - Poll `GET /transcript/{id}` until complete
   - Parse response

6. `_transcribe_deepgram(url: str)` -> RawTranscript:
   - POST to Deepgram with URL
   - Enable diarization
   - Parse response

7. Save output:
   - Write to `{source_path}/raw_transcript.json`
   - Update source.transcript_status = "complete"

## Config

Add to `src/gram_deploy/config.py`:
```python
s3_bucket: str = Field(validation_alias="GRAM_S3_BUCKET")
s3_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
```

## Environment Variables

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
GRAM_S3_BUCKET=gram-deployments
ELEVENLABS_API_KEY=...
```

## Reference

- Sample output: `deployments/deploy_20250119_vinci_01/sources/gopro_01/raw_transcript.json`

## Tests

Create `tests/test_transcription_service.py`:
- Mock boto3 presigned URL generation
- Mock provider API responses
- Test RawTranscript parsing
- Test error handling (expired URLs, rate limits, API failures)

## Files

- `src/gram_deploy/services/transcription_service.py`
- `src/gram_deploy/config.py` (add S3 settings)
- `tests/test_transcription_service.py`
