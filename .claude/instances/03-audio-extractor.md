# AudioExtractor Service

Implement audio extraction in `src/gram_deploy/services/audio_extractor.py`.

## Tasks

1. `extract_audio(video_path, output_path)` -> Path:
   - Use ffmpeg to extract audio track
   - Output format: WAV, 16kHz mono (optimal for speech recognition)
   - Command: `ffmpeg -i {video} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output}`
   - Return path to extracted audio

2. `get_video_metadata(video_path)` -> dict:
   - Use ffprobe to extract:
     - duration_seconds
     - video_codec
     - audio_codec
     - resolution (width x height)
     - fps
     - file_size_bytes
   - Parse JSON output from ffprobe

3. `extract_source_audio(source_path)`:
   - Read source.json to get file list
   - Extract audio from each video file
   - Concatenate if multiple files
   - Write to `{source_path}/audio.wav`

4. Error handling:
   - Check ffmpeg/ffprobe availability
   - Handle corrupt/missing video files
   - Validate output file exists and has content

## Tests
Create `tests/test_audio_extractor.py`:
- Mock subprocess calls to ffmpeg/ffprobe
- Test metadata parsing
- Test error handling

## Files
- `src/gram_deploy/services/audio_extractor.py`
- `tests/test_audio_extractor.py`
