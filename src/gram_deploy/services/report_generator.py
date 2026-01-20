"""Report Generator - produces deployment reports in various formats.

Responsible for:
- Generating comprehensive deployment reports
- Rendering to Markdown, HTML, and PDF
- Using templates for consistent formatting
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from gram_deploy.models import (
    ActionItem,
    CanonicalUtterance,
    Deployment,
    DeploymentEvent,
    DeploymentInsight,
    Source,
)


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
    generated_at: datetime


class ReportGenerator:
    """Generates deployment reports in various formats."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the generator.

        Args:
            template_dir: Optional directory containing report templates
        """
        self.template_dir = Path(template_dir) if template_dir else None

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
        issues = [e for e in events if e.event_type.value == "issue"]
        key_events = [e for e in events if e.event_type.value != "issue"]

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
            generated_at=datetime.utcnow(),
        )

    def render_markdown(self, report: DeploymentReport) -> str:
        """Render report as Markdown.

        Args:
            report: The DeploymentReport to render

        Returns:
            Markdown string
        """
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
| Team Members | {', '.join(report.team_members)} |
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
                md += f"| {time_str} | {event.event_type.value} | {event.title} | {event.description or '-'} |\n"
        else:
            md += "*No key events recorded.*\n"

        md += "\n---\n\n## Issues Encountered\n\n"

        if report.issues:
            for issue in report.issues:
                time_str = self._format_time(issue.canonical_time_ms)
                md += f"### {issue.title}\n\n"
                md += f"**Time:** {time_str}\n"
                md += f"**Severity:** {issue.severity.value if issue.severity else 'N/A'}\n\n"
                md += f"{issue.description or 'No description'}\n\n"
        else:
            md += "*No issues recorded.*\n"

        md += "\n---\n\n## Action Items\n\n"

        if report.action_items:
            md += "| # | Action | Mentioned By | Assigned To | Priority |\n"
            md += "|---|--------|--------------|-------------|----------|\n"
            for i, item in enumerate(report.action_items, 1):
                priority = item.priority.value if item.priority else "-"
                md += f"| {i} | {item.description} | {item.mentioned_by or '-'} | {item.assigned_to or '-'} | {priority} |\n"
        else:
            md += "*No action items recorded.*\n"

        md += "\n---\n\n## Insights & Observations\n\n"

        if report.insights:
            for insight in report.insights:
                md += f"### {insight.insight_type.value.replace('_', ' ').title()}\n\n"
                md += f"{insight.content}\n\n"
        else:
            md += "*No insights recorded.*\n"

        md += "\n---\n\n*Report generated by GRAM Deployment Processing System*\n"

        return md

    def render_html(self, report: DeploymentReport) -> str:
        """Render report as HTML.

        Args:
            report: The DeploymentReport to render

        Returns:
            HTML string
        """
        # Convert markdown to basic HTML
        md = self.render_markdown(report)

        # Simple markdown to HTML conversion
        html_content = md
        html_content = html_content.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deployment Report: {report.deployment.location}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #4285F4; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        tr:hover {{ background: #f9f9f9; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #4285F4; }}
        .metric-label {{ color: #666; }}
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
    <p>{', '.join(report.team_members)}</p>

    <h2>Key Events</h2>
    <table>
        <tr><th>Time</th><th>Type</th><th>Event</th><th>Description</th></tr>
        {''.join(f"<tr><td>{self._format_time(e.canonical_time_ms)}</td><td>{e.event_type.value}</td><td>{e.title}</td><td>{e.description or '-'}</td></tr>" for e in report.key_events[:20])}
    </table>

    <h2>Action Items</h2>
    <table>
        <tr><th>#</th><th>Action</th><th>Mentioned By</th><th>Assigned To</th><th>Priority</th></tr>
        {''.join(f"<tr><td>{i}</td><td>{item.description}</td><td>{item.mentioned_by or '-'}</td><td>{item.assigned_to or '-'}</td><td>{item.priority.value if item.priority else '-'}</td></tr>" for i, item in enumerate(report.action_items, 1))}
    </table>

    <h2>Insights</h2>
    {''.join(f"<h3>{insight.insight_type.value.replace('_', ' ').title()}</h3><p>{insight.content}</p>" for insight in report.insights)}

    <hr>
    <p><em>Report generated by GRAM Deployment Processing System</em></p>
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
            raise RuntimeError("weasyprint is required for PDF generation")

    def _generate_executive_summary(
        self,
        deployment: Deployment,
        events: list[DeploymentEvent],
        action_items: list[ActionItem],
    ) -> str:
        """Generate a basic executive summary."""
        milestones = [e for e in events if e.event_type.value == "milestone"]
        issues = [e for e in events if e.event_type.value == "issue"]

        summary = f"The deployment at {deployment.location} on {deployment.date} "

        if milestones:
            summary += f"achieved {len(milestones)} milestone(s). "
        if issues:
            summary += f"{len(issues)} issue(s) were encountered during the deployment. "
        if action_items:
            summary += f"{len(action_items)} follow-up action item(s) were identified."

        return summary or "Deployment completed successfully."

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
            lines.append(f"- {source.device_type.value} {source.device_number}: {source.total_duration / 60:.1f} minutes")

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
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
