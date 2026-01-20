# DeploymentManager Service

Implement deployment lifecycle management in `src/gram_deploy/services/deployment_manager.py`.

## Tasks

1. `create_deployment(location, date, team_members, notes)`:
   - Generate ID: `deploy:YYYYMMDD_location_NN` (NN = sequential)
   - Create directory: `deployments/deploy_YYYYMMDD_location_NN/`
   - Create subdirs: `sources/`, `outputs/`
   - Write `deployment.json`
   - Update `deployments/index.json`

2. `load_deployment(deployment_id)` -> Deployment:
   - Parse ID to directory path
   - Read and validate `deployment.json`
   - Return Deployment model

3. `save_deployment(deployment)`:
   - Write updated deployment.json
   - Update `updated_at` timestamp

4. `add_source(deployment_id, device_type, device_number, files)`:
   - Create `sources/{device_type}_{number}/`
   - Extract metadata with ffprobe for each file
   - Write `source.json`
   - Update deployment's sources list

5. `list_deployments()` -> list[str]:
   - Read `deployments/index.json`
   - Return deployment IDs

6. `update_status(deployment_id, status, checkpoint, error)`:
   - Load, update status fields, save

## Tests
Create `tests/test_deployment_manager.py` with CRUD operation tests.

## Files
- `src/gram_deploy/services/deployment_manager.py`
- `tests/test_deployment_manager.py`
