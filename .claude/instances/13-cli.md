# CLI Completion

Complete the CLI in `src/gram_deploy/__main__.py`.

## Tasks

1. Verify existing commands work:
   - `deploy create --location X --date Y`
   - `deploy add-source DEPLOY_ID --type gopro --number 1 --files *.MP4`

2. Implement `deploy process DEPLOY_ID`:
   - Call PipelineOrchestrator.process()
   - Display progress with Rich progress bars
   - Show stage status updates
   - Handle Ctrl+C gracefully

3. Implement `deploy search QUERY`:
   - Call SearchIndexBuilder.search()
   - Format results with Rich tables
   - Show snippets with highlighted matches
   - Include timestamps and speaker

4. Implement `deploy report DEPLOY_ID`:
   - Options: --format (md/html/pdf), --output PATH
   - Call ReportGenerator
   - Open in browser if HTML (optional)

5. Implement `deploy status DEPLOY_ID`:
   - Show current processing state
   - List completed stages
   - Show errors if failed

6. Implement person commands:
   - `person add NAME --role ROLE --aliases A1,A2`
   - `person list`
   - Update `deployments/people.json`

7. Polish:
   - Consistent error messages
   - Help text for all commands
   - --verbose flag for debug output

## Tests
Create `tests/test_cli.py`:
- Test command parsing
- Test output formatting
- Use Click's CliRunner

## Files
- `src/gram_deploy/__main__.py`
- `tests/test_cli.py`
