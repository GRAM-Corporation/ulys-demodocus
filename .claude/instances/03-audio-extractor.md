# AudioExtractor Service

**STATUS: DEPRECATED** - Not used in main pipeline. Transcription uses S3 presigned URLs directly.

The existing implementation in `src/gram_deploy/services/audio_extractor.py` is complete but unused.

## When It Might Be Needed

- Local development/testing without S3
- Future audio fingerprinting for high-confidence alignment (would require S3 download)
- Offline processing scenarios

## No Action Required

This instance spec is deprecated. Skip spawning an instance for this component.
