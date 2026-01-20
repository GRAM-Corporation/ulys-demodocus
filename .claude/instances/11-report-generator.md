# ReportGenerator Service

Implement deployment reports in `src/gram_deploy/services/report_generator.py`.

## Tasks

1. `generate_report(deployment, format, output_path)`:
   - Load all deployment data
   - Render in requested format
   - Write to output path

2. `_generate_markdown(deployment)` -> str:
   - Clean markdown output
   - Sections: Summary, Key Events, Action Items, Insights, Full Transcript
   - Table formatting for structured data

3. `_generate_html(deployment)` -> str:
   - Styled HTML report
   - Embedded CSS for print-friendly layout
   - Optional embedded timeline
   - Use Jinja2 template

4. `_generate_pdf(deployment, output_path)`:
   - Generate HTML first
   - Convert to PDF with WeasyPrint
   - Handle page breaks appropriately

5. Create templates in `src/gram_deploy/templates/`:
   - `report.md.j2` - Markdown template
   - `report.html.j2` - HTML template with styles

6. Report sections:
   - Header: deployment ID, date, location, team
   - Executive summary
   - Timeline of key events
   - Action items table (assignee, priority, status)
   - Insights and recommendations
   - Full transcript with timestamps

## Tests
Create `tests/test_report_generator.py`:
- Test each format output
- Validate markdown structure
- Validate HTML validity

## Files
- `src/gram_deploy/services/report_generator.py`
- `src/gram_deploy/templates/report.md.j2`
- `src/gram_deploy/templates/report.html.j2`
- `tests/test_report_generator.py`
