# PipelineOrchestrator Service

Implement end-to-end processing in `src/gram_deploy/services/pipeline.py`.

## Tasks

1. `process(deployment_id, resume=True)`:
   - Load deployment
   - Execute stages in order
   - Update status after each stage
   - Support resume from checkpoint

2. Pipeline stages:
   ```
   STAGES = [
     ("transcribing", transcribe_sources),    # S3 → ElevenLabs
     ("aligning", align_sources),             # Transcript-based
     ("resolving_speakers", resolve_speakers),
     ("merging", merge_transcripts),
     ("analyzing", analyze_semantics),
     ("indexing", build_search_index),
     ("visualizing", generate_timeline),
     ("reporting", generate_report),
   ]
   ```

   **Note:** No audio extraction stage - transcription uses S3 presigned URLs directly.

3. Checkpoint support:
   - Save checkpoint after each stage
   - On resume, skip completed stages
   - Store in deployment.checkpoint field

4. Progress callbacks:
   - `on_stage_start(stage_name)`
   - `on_stage_complete(stage_name, duration)`
   - `on_error(stage_name, error)`
   - For CLI progress display

5. Parallel processing:
   - Multiple sources can transcribe in parallel
   - Use asyncio or concurrent.futures

6. Error handling:
   - Catch stage errors
   - Update deployment.error_message
   - Set status to "failed"
   - Allow retry from failed stage

## Tests
Create `tests/test_pipeline.py`:
- Test stage sequencing
- Test checkpoint/resume
- Test error handling

## Files
- `src/gram_deploy/services/pipeline.py`
- `tests/test_pipeline.py`
