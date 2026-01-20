# Wave 4: Output & Orchestration

Implement visualization, reporting, pipeline orchestration, and CLI.

## Components

### 1. TimelineVisualizer (`src/gram_deploy/services/timeline_visualizer.py`)
Interactive HTML timeline:
- `generate_timeline()` - Create interactive visualization
- Render canonical timeline with speaker lanes
- Mark events, action items on timeline
- Playback synchronization points for multi-source video
- Color-code by speaker, event type
- Output to `deployment/timeline.html`
- Use Jinja2 templates in `src/gram_deploy/templates/`

### 2. ReportGenerator (`src/gram_deploy/services/report_generator.py`)
Multi-format deployment reports:
- `generate_report()` - Main report generation
- `_generate_markdown()` - Clean markdown output
- `_generate_html()` - Styled HTML with embedded timeline
- `_generate_pdf()` - PDF via WeasyPrint
- Include: summary, key events, action items, full transcript, insights
- Template-based with Jinja2
- Output to `deployment/report.{md,html,pdf}`

### 3. PipelineOrchestrator (`src/gram_deploy/services/pipeline.py`)
End-to-end processing:
- `process()` - Full pipeline execution
- Stage management with checkpoints
- Resume from last successful stage on failure
- Progress callbacks for CLI display
- Parallel processing where possible (multiple sources)
- Error handling with detailed logging

### 4. CLI Completion (`src/gram_deploy/__main__.py`)
Wire up all commands:
- Verify `deploy create`, `deploy add-source` work
- Implement `deploy process` with progress display
- Implement `deploy search` with formatted results
- Implement `deploy report` with format options
- Add `deploy status` to show processing state
- Add `person add`, `person list` commands

## Testing
Create `tests/test_wave4.py` with:
- Timeline HTML validation
- Report generation in all formats
- Full pipeline integration test
- CLI command tests

## Dependencies
- Requires Waves 1-3 complete
- End-to-end test with sample deployment
