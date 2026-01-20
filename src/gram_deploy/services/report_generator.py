"""Report Generator - produces deployment reports in various formats.

Responsible for:
- Generating comprehensive deployment reports
- Rendering to Markdown, HTML, and PDF
- Using templates for consistent formatting
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from gram_deploy.models import (
    ActionItem,
    CanonicalUtterance,
    Deployment,
    DeploymentEvent,
    DeploymentInsight,
    Source,
)


class ReportFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "md"
    HTML = "html"
    PDF = "pdf"


@dataclass
class DeploymentReport:
    """A generated deployment report."""

    deployment: Deployment
    executive_summary: str
    timeline_overview: str
    key_events: list[DeploymentEvent]
    issues: list[DeploymentEvent]
    action_items: list[ActionItem]
    insights: list[DeploymentInsight]
    team_members: list[str]
    metrics: dict
    utterances: list[CanonicalUtterance]
    people_names: dict[str, str]
    generated_at: datetime


def _get_enum_value(val) -> str:
    """Get the string value from an enum or return the string itself."""
    if val is None:
        return ""
    if hasattr(val, 'value'):
        return val.value
    return str(val)


class ReportGenerator:
    """Generates deployment reports in various formats."""

    def __init__(
        self,
        template_dir: Optional[str] = None,
        data_dir: str = "deployments",
    ):
        """Initialize the generator.

        Args:
            template_dir: Optional directory containing report templates.
                         If None, uses package templates.
            data_dir: Directory containing deployment data files.
        """
        self.data_dir = Path(data_dir)

        # Setup Jinja2 environment
        if template_dir:
            self.template_dir = Path(template_dir)
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(['html', 'xml']),
            )
        else:
            # Use default templates from package
            default_template_dir = Path(__file__).parent.parent / "templates"
            self.template_dir = default_template_dir
            if default_template_dir.exists():
                self.jinja_env = Environment(
                    loader=FileSystemLoader(str(default_template_dir)),
                    autoescape=select_autoescape(['html', 'xml']),
                )
            else:
                self.jinja_env = None

        # Register custom filters
        if self.jinja_env:
            self.jinja_env.filters['format_time'] = self._format_time
            self.jinja_env.filters['format_priority'] = lambda p: _get_enum_value(p) if p else "-"
            self.jinja_env.filters['format_severity'] = lambda s: _get_enum_value(s) if s else "N/A"
            self.jinja_env.filters['enum_value'] = _get_enum_value

    def generate(
        self,
        deployment: Union[str, Deployment],
        format: Union[str, ReportFormat] = ReportFormat.MARKDOWN,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a deployment report and optionally write to file.

        This is the main entry point that loads all deployment data,
        generates the report, renders it in the requested format,
        and writes to the output path.

        Args:
            deployment: Deployment object or deployment ID string
            format: Output format (md, html, pdf)
            output_path: Optional path to write the report to

        Returns:
            The rendered report as a string (or path for PDF)
        """
        # Normalize format
        if isinstance(format, str):
            format = ReportFormat(format.lower())

        # Load deployment if string ID provided
        if isinstance(deployment, str):
            deployment = self._load_deployment(deployment)

        # Load all deployment data
        deploy_dir = self._get_deploy_dir(deployment.id)
        sources = self._load_sources(deployment, deploy_dir)
        utterances = self._load_utterances(deployment, deploy_dir)
        events = self._load_events(deployment, deploy_dir)
        action_items = self._load_action_items(deployment, deploy_dir)
        insights = self._load_insights(deployment, deploy_dir)
        people_names = self._load_people_names()
        summary = self._load_summary(deploy_dir)

        # Generate report object
        report = self.generate_report(
            deployment=deployment,
            sources=sources,
            utterances=utterances,
            events=events,
            action_items=action_items,
            insights=insights,
            summary=summary,
            people_names=people_names,
        )

        # Render in requested format
        if format == ReportFormat.MARKDOWN:
            content = self.render_markdown(report)
        elif format == ReportFormat.HTML:
            content = self.render_html(report)
        elif format == ReportFormat.PDF:
            if output_path is None:
                raise ValueError("output_path is required for PDF format")
            pdf_bytes = self.render_pdf(report)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(pdf_bytes)
            return str(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding='utf-8')

        return content

    def generate_report(
        self,
        deployment: Deployment,
        sources: list[Source],
        utterances: list[CanonicalUtterance],
        events: list[DeploymentEvent],
        action_items: list[ActionItem],
        insights: list[DeploymentInsight],
        summary: Optional[str] = None,
        people_names: Optional[dict[str, str]] = None,
    ) -> DeploymentReport:
        """Generate a comprehensive deployment report.

        Args:
            deployment: The Deployment entity
            sources: Sources in the deployment
            utterances: Canonical utterances
            events: Extracted events
            action_items: Extracted action items
            insights: Extracted insights
            summary: Optional pre-generated summary
            people_names: Optional mapping of person_id -> display name

        Returns:
            DeploymentReport object
        """
        people_names = people_names or {}

        # Generate executive summary if not provided
        executive_summary = summary or self._generate_executive_summary(
            deployment, events, action_items
        )

        # Generate timeline overview
        timeline_overview = self._generate_timeline_overview(deployment, sources, events)

        # Separate issues from other events
        issues = [e for e in events if _get_enum_value(e.event_type) == "issue"]
        key_events = [e for e in events if _get_enum_value(e.event_type) != "issue"]

        # Compute metrics
        metrics = self._compute_metrics(deployment, sources, utterances, people_names)

        # Get team member names
        team_members = list(set(
            people_names.get(u.speaker_id, u.speaker_id or "Unknown")
            for u in utterances
            if u.speaker_id
        ))

        return DeploymentReport(
            deployment=deployment,
            executive_summary=executive_summary,
            timeline_overview=timeline_overview,
            key_events=key_events,
            issues=issues,
            action_items=action_items,
            insights=insights,
            team_members=team_members,
            metrics=metrics,
            utterances=sorted(utterances, key=lambda u: u.canonical_start_ms),
            people_names=people_names,
            generated_at=datetime.utcnow(),
        )

    def render_markdown(self, report: DeploymentReport) -> str:
        """Render report as Markdown.

        Args:
            report: The DeploymentReport to render

        Returns:
            Markdown string
        """
        # Try to use Jinja2 template if available
        if self.jinja_env and self._template_exists('report.md.j2'):
            template = self.jinja_env.get_template('report.md.j2')
            return template.render(report=report, format_time=self._format_time)

        # Fallback to inline rendering
        return self._render_markdown_inline(report)

    def _render_markdown_inline(self, report: DeploymentReport) -> str:
        """Render markdown without template (fallback)."""
        md = f"""# Deployment Report: {report.deployment.location}

**Date:** {report.deployment.date}
**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}

---

## Executive Summary

{report.executive_summary}

---

## Deployment Details

| Metric | Value |
|--------|-------|
| Location | {report.deployment.location} |
| Date | {report.deployment.date} |
| Duration | {report.metrics.get('duration_formatted', 'N/A')} |
| Team Members | {', '.join(report.team_members) if report.team_members else 'N/A'} |
| Total Utterances | {report.metrics.get('utterance_count', 0)} |
| Sources | {report.metrics.get('source_count', 0)} |

---

## Timeline Overview

{report.timeline_overview}

---

## Key Events

"""

        if report.key_events:
            md += "| Time | Type | Event | Description |\n"
            md += "|------|------|-------|-------------|\n"
            for event in report.key_events[:20]:
                time_str = self._format_time(event.canonical_time_ms)
                desc = (event.description or '-').replace('\n', ' ')[:100]
                md += f"| {time_str} | {_get_enum_value(event.event_type)} | {event.title} | {desc} |\n"
        else:
            md += "*No key events recorded.*\n"

        md += "\n---\n\n## Issues Encountered\n\n"

        if report.issues:
            for issue in report.issues:
                time_str = self._format_time(issue.canonical_time_ms)
                md += f"### {issue.title}\n\n"
                md += f"**Time:** {time_str}\n"
                md += f"**Severity:** {_get_enum_value(issue.severity) if issue.severity else 'N/A'}\n\n"
                md += f"{issue.description or 'No description'}\n\n"
        else:
            md += "*No issues recorded.*\n"

        md += "\n---\n\n## Action Items\n\n"

        if report.action_items:
            md += "| # | Action | Mentioned By | Assigned To | Priority |\n"
            md += "|---|--------|--------------|-------------|----------|\n"
            for i, item in enumerate(report.action_items, 1):
                priority = _get_enum_value(item.priority) if item.priority else "-"
                mentioned = report.people_names.get(item.mentioned_by, item.mentioned_by) if item.mentioned_by else "-"
                assigned = report.people_names.get(item.assigned_to, item.assigned_to) if item.assigned_to else "-"
                md += f"| {i} | {item.description} | {mentioned} | {assigned} | {priority} |\n"
        else:
            md += "*No action items recorded.*\n"

        md += "\n---\n\n## Insights & Observations\n\n"

        if report.insights:
            for insight in report.insights:
                md += f"### {_get_enum_value(insight.insight_type).replace('_', ' ').title()}\n\n"
                md += f"{insight.content}\n\n"
        else:
            md += "*No insights recorded.*\n"

        # Full Transcript section
        md += "\n---\n\n## Full Transcript\n\n"

        if report.utterances:
            for utterance in report.utterances:
                time_str = self._format_time(utterance.canonical_start_ms)
                speaker = report.people_names.get(utterance.speaker_id, utterance.speaker_id or "Unknown")
                md += f"**[{time_str}] {speaker}:** {utterance.text}\n\n"
        else:
            md += "*No transcript available.*\n"

        md += "\n---\n\n*Report generated by GRAM Deployment Processing System*\n"

        return md

    def render_html(self, report: DeploymentReport) -> str:
        """Render report as HTML.

        Args:
            report: The DeploymentReport to render

        Returns:
            HTML string
        """
        # Try to use Jinja2 template if available
        if self.jinja_env and self._template_exists('report.html.j2'):
            template = self.jinja_env.get_template('report.html.j2')
            return template.render(report=report, format_time=self._format_time)

        # Fallback to inline rendering
        return self._render_html_inline(report)

    def _render_html_inline(self, report: DeploymentReport) -> str:
        """Render HTML without template (fallback)."""
        # Build key events table rows
        events_rows = ""
        for e in report.key_events[:20]:
            time_str = self._format_time(e.canonical_time_ms)
            desc = (e.description or '-').replace('\n', ' ')[:100]
            events_rows += f"<tr><td>{time_str}</td><td>{_get_enum_value(e.event_type)}</td><td>{e.title}</td><td>{desc}</td></tr>\n"

        if not events_rows:
            events_rows = "<tr><td colspan='4' style='text-align:center;font-style:italic;'>No key events recorded.</td></tr>"

        # Build action items table rows
        action_rows = ""
        for i, item in enumerate(report.action_items, 1):
            priority = _get_enum_value(item.priority) if item.priority else "-"
            mentioned = report.people_names.get(item.mentioned_by, item.mentioned_by) if item.mentioned_by else "-"
            assigned = report.people_names.get(item.assigned_to, item.assigned_to) if item.assigned_to else "-"
            action_rows += f"<tr><td>{i}</td><td>{item.description}</td><td>{mentioned}</td><td>{assigned}</td><td>{priority}</td></tr>\n"

        if not action_rows:
            action_rows = "<tr><td colspan='5' style='text-align:center;font-style:italic;'>No action items recorded.</td></tr>"

        # Build insights section
        insights_html = ""
        if report.insights:
            for insight in report.insights:
                title = _get_enum_value(insight.insight_type).replace('_', ' ').title()
                insights_html += f"<h3>{title}</h3><p>{insight.content}</p>\n"
        else:
            insights_html = "<p style='font-style:italic;'>No insights recorded.</p>"

        # Build transcript section
        transcript_html = ""
        if report.utterances:
            transcript_html = '<div class="transcript">\n'
            for utterance in report.utterances:
                time_str = self._format_time(utterance.canonical_start_ms)
                speaker = report.people_names.get(utterance.speaker_id, utterance.speaker_id or "Unknown")
                transcript_html += f'<p class="utterance"><span class="timestamp">[{time_str}]</span> <span class="speaker">{speaker}:</span> {utterance.text}</p>\n'
            transcript_html += "</div>"
        else:
            transcript_html = "<p style='font-style:italic;'>No transcript available.</p>"

        # Build issues section
        issues_html = ""
        if report.issues:
            for issue in report.issues:
                time_str = self._format_time(issue.canonical_time_ms)
                severity = _get_enum_value(issue.severity) if issue.severity else 'N/A'
                desc = issue.description or 'No description'
                issues_html += f"""
                <div class="issue">
                    <h3>{issue.title}</h3>
                    <p><strong>Time:</strong> {time_str} | <strong>Severity:</strong> {severity}</p>
                    <p>{desc}</p>
                </div>
                """
        else:
            issues_html = "<p style='font-style:italic;'>No issues recorded.</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deployment Report: {report.deployment.location}</title>
    <style>
        :root {{
            --primary-color: #4285F4;
            --text-color: #333;
            --text-muted: #666;
            --border-color: #ddd;
            --bg-light: #f5f5f5;
            --bg-hover: #f9f9f9;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px 40px;
            line-height: 1.6;
            color: var(--text-color);
        }}

        h1 {{
            color: var(--text-color);
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}

        h2 {{
            color: #444;
            margin-top: 40px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}

        h3 {{ color: var(--text-muted); }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}

        th, td {{
            border: 1px solid var(--border-color);
            padding: 12px 15px;
            text-align: left;
        }}

        th {{
            background: var(--bg-light);
            font-weight: 600;
        }}

        tr:hover {{ background: var(--bg-hover); }}

        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 40px 0;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}

        .metric {{
            background: var(--bg-light);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .metric-label {{
            color: var(--text-muted);
            margin-top: 5px;
        }}

        .issue {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}

        .transcript {{
            background: var(--bg-light);
            padding: 20px 25px;
            border-radius: 8px;
            max-height: 600px;
            overflow-y: auto;
        }}

        .utterance {{
            margin: 10px 0;
            line-height: 1.5;
        }}

        .timestamp {{
            color: var(--text-muted);
            font-family: 'Menlo', 'Monaco', monospace;
            font-size: 0.9em;
        }}

        .speaker {{
            font-weight: 600;
            color: var(--primary-color);
        }}

        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9em;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}

        @media print {{
            body {{ padding: 0; max-width: none; }}
            .transcript {{ max-height: none; }}
            h2 {{ page-break-before: auto; }}
            .issue, .metric {{ break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <h1>Deployment Report: {report.deployment.location}</h1>
    <p><strong>Date:</strong> {report.deployment.date} | <strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>

    <h2>Executive Summary</h2>
    <p>{report.executive_summary}</p>

    <h2>Deployment Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{report.metrics.get('duration_formatted', 'N/A')}</div>
            <div class="metric-label">Duration</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(report.team_members)}</div>
            <div class="metric-label">Team Members</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.metrics.get('source_count', 0)}</div>
            <div class="metric-label">Video Sources</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(report.action_items)}</div>
            <div class="metric-label">Action Items</div>
        </div>
    </div>

    <h2>Team Members</h2>
    <p>{', '.join(report.team_members) if report.team_members else 'N/A'}</p>

    <h2>Key Events</h2>
    <table>
        <thead>
            <tr><th>Time</th><th>Type</th><th>Event</th><th>Description</th></tr>
        </thead>
        <tbody>
            {events_rows}
        </tbody>
    </table>

    <h2>Issues Encountered</h2>
    {issues_html}

    <h2>Action Items</h2>
    <table>
        <thead>
            <tr><th>#</th><th>Action</th><th>Mentioned By</th><th>Assigned To</th><th>Priority</th></tr>
        </thead>
        <tbody>
            {action_rows}
        </tbody>
    </table>

    <h2>Insights & Observations</h2>
    {insights_html}

    <h2>Full Transcript</h2>
    {transcript_html}

    <div class="footer">
        <p>Report generated by GRAM Deployment Processing System</p>
    </div>
</body>
</html>"""

        return html

    def render_pdf(self, report: DeploymentReport) -> bytes:
        """Render report as PDF.

        Args:
            report: The DeploymentReport to render

        Returns:
            PDF bytes
        """
        try:
            from weasyprint import HTML

            html = self.render_html(report)
            return HTML(string=html).write_pdf()

        except ImportError:
            raise RuntimeError("weasyprint is required for PDF generation. Install with: pip install weasyprint")

    def _template_exists(self, template_name: str) -> bool:
        """Check if a template file exists."""
        if not self.template_dir or not self.template_dir.exists():
            return False
        return (self.template_dir / template_name).exists()

    def _get_deploy_dir(self, deployment_id: str) -> Path:
        """Get the directory path for a deployment."""
        # deploy:20250119_vinci_01 -> deploy_20250119_vinci_01
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

    def _load_deployment(self, deployment_id: str) -> Deployment:
        """Load a deployment from file."""
        deploy_dir = self._get_deploy_dir(deployment_id)
        deploy_file = deploy_dir / "deployment.json"

        if not deploy_file.exists():
            raise FileNotFoundError(f"Deployment file not found: {deploy_file}")

        data = json.loads(deploy_file.read_text())
        return Deployment.model_validate(data)

    def _load_sources(self, deployment: Deployment, deploy_dir: Path) -> list[Source]:
        """Load sources for a deployment."""
        sources_file = deploy_dir / "sources.json"
        if not sources_file.exists():
            return []

        data = json.loads(sources_file.read_text())
        if isinstance(data, list):
            return [Source.model_validate(s) for s in data]
        elif isinstance(data, dict) and "sources" in data:
            return [Source.model_validate(s) for s in data["sources"]]
        return []

    def _load_utterances(self, deployment: Deployment, deploy_dir: Path) -> list[CanonicalUtterance]:
        """Load canonical utterances for a deployment."""
        transcript_file = deploy_dir / "canonical" / "transcript.json"
        if not transcript_file.exists():
            return []

        data = json.loads(transcript_file.read_text())
        if isinstance(data, list):
            return [CanonicalUtterance.model_validate(u) for u in data]
        elif isinstance(data, dict) and "utterances" in data:
            return [CanonicalUtterance.model_validate(u) for u in data["utterances"]]
        return []

    def _load_events(self, deployment: Deployment, deploy_dir: Path) -> list[DeploymentEvent]:
        """Load events for a deployment."""
        events_file = deploy_dir / "events.json"
        if not events_file.exists():
            return []

        data = json.loads(events_file.read_text())
        if isinstance(data, list):
            return [DeploymentEvent.model_validate(e) for e in data]
        return []

    def _load_action_items(self, deployment: Deployment, deploy_dir: Path) -> list[ActionItem]:
        """Load action items for a deployment."""
        items_file = deploy_dir / "action_items.json"
        if not items_file.exists():
            return []

        data = json.loads(items_file.read_text())
        if isinstance(data, list):
            return [ActionItem.model_validate(a) for a in data]
        return []

    def _load_insights(self, deployment: Deployment, deploy_dir: Path) -> list[DeploymentInsight]:
        """Load insights for a deployment."""
        insights_file = deploy_dir / "insights.json"
        if not insights_file.exists():
            return []

        data = json.loads(insights_file.read_text())
        if isinstance(data, list):
            return [DeploymentInsight.model_validate(i) for i in data]
        return []

    def _load_summary(self, deploy_dir: Path) -> Optional[str]:
        """Load deployment summary if available."""
        summary_file = deploy_dir / "summary.md"
        if summary_file.exists():
            return summary_file.read_text()
        return None

    def _load_people_names(self) -> dict[str, str]:
        """Load people names from registry."""
        people_file = self.data_dir / "people.json"
        if not people_file.exists():
            return {}

        try:
            data = json.loads(people_file.read_text())
            if "people" in data:
                return {p["id"]: p["name"] for p in data["people"]}
            return {}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _generate_executive_summary(
        self,
        deployment: Deployment,
        events: list[DeploymentEvent],
        action_items: list[ActionItem],
    ) -> str:
        """Generate a basic executive summary."""
        milestones = [e for e in events if _get_enum_value(e.event_type) == "milestone"]
        issues = [e for e in events if _get_enum_value(e.event_type) == "issue"]

        summary = f"The deployment at {deployment.location} on {deployment.date} "

        if milestones:
            summary += f"achieved {len(milestones)} milestone(s). "
        if issues:
            summary += f"{len(issues)} issue(s) were encountered during the deployment. "
        if action_items:
            summary += f"{len(action_items)} follow-up action item(s) were identified."

        if not (milestones or issues or action_items):
            summary += "completed successfully."

        return summary

    def _generate_timeline_overview(
        self,
        deployment: Deployment,
        sources: list[Source],
        events: list[DeploymentEvent],
    ) -> str:
        """Generate a timeline overview section."""
        lines = []

        if deployment.canonical_start_time:
            lines.append(f"**Start:** {deployment.canonical_start_time.strftime('%H:%M:%S')}")
        if deployment.canonical_end_time:
            lines.append(f"**End:** {deployment.canonical_end_time.strftime('%H:%M:%S')}")

        lines.append(f"\n**Sources:** {len(sources)} video source(s)")

        for source in sources:
            duration_min = source.total_duration / 60 if source.total_duration else 0
            lines.append(f"- {_get_enum_value(source.device_type)} {source.device_number}: {duration_min:.1f} minutes")

        return "\n".join(lines)

    def _compute_metrics(
        self,
        deployment: Deployment,
        sources: list[Source],
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> dict:
        """Compute deployment metrics."""
        metrics = {
            "source_count": len(sources),
            "utterance_count": len(utterances),
        }

        # Duration
        if deployment.canonical_start_time and deployment.canonical_end_time:
            duration = deployment.canonical_end_time - deployment.canonical_start_time
            hours = int(duration.total_seconds() // 3600)
            minutes = int((duration.total_seconds() % 3600) // 60)
            metrics["duration_seconds"] = duration.total_seconds()
            metrics["duration_formatted"] = f"{hours}h {minutes}m"
        else:
            total_duration = sum(s.total_duration for s in sources)
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            metrics["duration_seconds"] = total_duration
            metrics["duration_formatted"] = f"{hours}h {minutes}m"

        # Speaker stats
        speakers = set(u.speaker_id for u in utterances if u.speaker_id)
        metrics["speaker_count"] = len(speakers)

        return metrics

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS."""
        if ms is None:
            return "00:00:00"
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
