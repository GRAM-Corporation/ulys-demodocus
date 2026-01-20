# Models Validation & Completion

Complete and validate all Pydantic data models in `src/gram_deploy/models/`.

## Tasks

1. Review each model file and ensure:
   - Correct field types and Optional markers
   - ID pattern validators (e.g., `deploy:YYYYMMDD_location_NN`)
   - Enum constraints where applicable
   - Proper datetime handling with timezone awareness

2. Add missing validators:
   - `Deployment.id` pattern validation
   - `Source.id` pattern validation
   - `Person.id` pattern validation
   - Time range validators (start < end)

3. Add model methods:
   - `to_json()` / `from_json()` for file I/O
   - `Deployment.get_source_path(source_id)` helper
   - `Timeline.get_utterances_in_range(start, end)`

4. Test with sample data:
   - Load `deployments/deploy_20250119_vinci_01/deployment.json`
   - Load `deployments/deploy_20250119_vinci_01/sources/gopro_01/source.json`
   - Load `deployments/deploy_20250119_vinci_01/sources/gopro_01/raw_transcript.json`

## Tests
Create `tests/test_models.py` with validation and serialization tests.

## Files
- `src/gram_deploy/models/*.py`
- `tests/test_models.py`
