"""Tests for the ReportGenerator service."""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models.canonical_utterance import CanonicalUtterance, UtteranceSource
from gram_deploy.models.deployment import Deployment, DeploymentStatus
from gram_deploy.models.event import DeploymentEvent, EventType, Severity, ExtractionMethod
from gram_deploy.models.action_item import ActionItem, ActionItemStatus, Priority
from gram_deploy.models.insight import DeploymentInsight, InsightType
from gram_deploy.models.source import Source, SourceFile, DeviceType

# Import report_generator directly to avoid loading all services
import importlib.util

spec = importlib.util.spec_from_file_location(
    "report_generator",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "report_generator.py",
)
rg_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg_module)
ReportGenerator = rg_module.ReportGenerator
ReportFormat = rg_module.ReportFormat
DeploymentReport = rg_module.DeploymentReport


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def service(temp_data_dir):
    """Create a ReportGenerator with temp directory."""
    return ReportGenerator(data_dir=temp_data_dir)


@pytest.fixture
def sample_deployment():
    """Create a sample deployment for testing."""
    return Deployment(
        id="deploy:20250119_vinci_01",
        location="vinci",
        date="2025-01-19",
        canonical_start_time=datetime(2025, 1, 19, 9, 0, 0, tzinfo=timezone.utc),
        canonical_end_time=datetime(2025, 1, 19, 12, 0, 0, tzinfo=timezone.utc),
        status=DeploymentStatus.COMPLETE,
    )


@pytest.fixture
def sample_sources():
    """Create sample sources for testing."""
    return [
        Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            total_duration_seconds=3600.0,
        ),
        Source(
            id="source:deploy:20250119_vinci_01/phone_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.PHONE,
            device_number=1,
            total_duration_seconds=1800.0,
        ),
    ]


@pytest.fixture
def sample_utterances():
    """Create sample canonical utterances for testing."""
    return [
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/abc123",
            deployment_id="deploy:20250119_vinci_01",
            text="We need to check the battery levels before we start.",
            canonical_start_ms=0,
            canonical_end_ms=5000,
            speaker_id="person:damion",
            speaker_confidence=0.9,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=0.0,
                    local_end_time=5.0,
                )
            ],
        ),
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/def456",
            deployment_id="deploy:20250119_vinci_01",
            text="I'll handle the Starlink setup. We should test connectivity first.",
            canonical_start_ms=6000,
            canonical_end_ms=12000,
            speaker_id="person:chu",
            speaker_confidence=0.85,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=6.0,
                    local_end_time=12.0,
                )
            ],
        ),
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/ghi789",
            deployment_id="deploy:20250119_vinci_01",
            text="Let's document this process for the next deployment.",
            canonical_start_ms=15000,
            canonical_end_ms=20000,
            speaker_id="person:damion",
            speaker_confidence=0.9,
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=15.0,
                    local_end_time=20.0,
                )
            ],
        ),
    ]


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/evt001",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.MILESTONE,
            canonical_time_ms=5000,
            title="Deployment Started",
            description="Team began deployment activities",
            extraction_method=ExtractionMethod.LLM_EXTRACTED,
        ),
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/evt002",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.ISSUE,
            canonical_time_ms=60000,
            title="Battery Low Warning",
            description="GoPro 1 showed low battery warning",
            severity=Severity.WARNING,
            extraction_method=ExtractionMethod.LLM_EXTRACTED,
        ),
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/evt003",
            deployment_id="deploy:20250119_vinci_01",
            event_type=EventType.DECISION,
            canonical_time_ms=120000,
            title="Decided to use Starlink",
            description="Team decided to prioritize Starlink setup",
            extraction_method=ExtractionMethod.LLM_EXTRACTED,
        ),
    ]


@pytest.fixture
def sample_action_items():
    """Create sample action items for testing."""
    return [
        ActionItem(
            id="action:deploy:20250119_vinci_01/act001",
            deployment_id="deploy:20250119_vinci_01",
            description="Replace GoPro 1 battery before next deployment",
            source_utterance_id="utterance:deploy:20250119_vinci_01/abc123",
            canonical_time_ms=5000,
            mentioned_by="person:damion",
            priority=Priority.HIGH,
        ),
        ActionItem(
            id="action:deploy:20250119_vinci_01/act002",
            deployment_id="deploy:20250119_vinci_01",
            description="Document Starlink setup process",
            source_utterance_id="utterance:deploy:20250119_vinci_01/ghi789",
            canonical_time_ms=15000,
            mentioned_by="person:damion",
            assigned_to="person:chu",
            priority=Priority.MEDIUM,
        ),
    ]


@pytest.fixture
def sample_insights():
    """Create sample insights for testing."""
    return [
        DeploymentInsight(
            id="insight:deploy:20250119_vinci_01/ins001",
            deployment_id="deploy:20250119_vinci_01",
            insight_type=InsightType.PROCESS_IMPROVEMENT,
            content="Pre-deployment battery checks improve efficiency and prevent interruptions.",
            confidence=0.85,
        ),
        DeploymentInsight(
            id="insight:deploy:20250119_vinci_01/ins002",
            deployment_id="deploy:20250119_vinci_01",
            insight_type=InsightType.TECHNICAL_OBSERVATION,
            content="Starlink connectivity was stable once positioned correctly.",
            confidence=0.9,
        ),
    ]


@pytest.fixture
def people_names():
    """Create sample people names mapping."""
    return {
        "person:damion": "Damion Shelton",
        "person:chu": "Chu",
    }


class TestServiceInitialization:
    """Tests for service initialization."""

    def test_init_with_default_data_dir(self):
        """Test that service uses default data directory."""
        service = ReportGenerator()
        assert service.data_dir == Path("deployments")

    def test_init_with_custom_data_dir(self, temp_data_dir):
        """Test that service uses custom data directory."""
        service = ReportGenerator(data_dir=temp_data_dir)
        assert service.data_dir == Path(temp_data_dir)

    def test_init_with_custom_template_dir(self, temp_data_dir):
        """Test that service uses custom template directory."""
        template_dir = Path(temp_data_dir) / "templates"
        template_dir.mkdir()

        service = ReportGenerator(template_dir=str(template_dir), data_dir=temp_data_dir)
        assert service.template_dir == template_dir


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_creates_report(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that generate_report creates a DeploymentReport."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        assert isinstance(report, DeploymentReport)
        assert report.deployment == sample_deployment
        assert len(report.key_events) == 2  # milestone and decision
        assert len(report.issues) == 1  # one issue
        assert len(report.action_items) == 2
        assert len(report.insights) == 2
        assert len(report.utterances) == 3

    def test_generate_report_with_summary(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that provided summary is used."""
        custom_summary = "This is a custom summary."
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            summary=custom_summary,
        )

        assert report.executive_summary == custom_summary

    def test_generate_report_auto_summary(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that summary is auto-generated if not provided."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        assert "vinci" in report.executive_summary
        assert "2025-01-19" in report.executive_summary

    def test_generate_report_metrics(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that metrics are computed correctly."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        assert report.metrics["source_count"] == 2
        assert report.metrics["utterance_count"] == 3
        assert report.metrics["duration_formatted"] == "3h 0m"

    def test_generate_report_team_members(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that team members are extracted correctly."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        assert "Damion Shelton" in report.team_members
        assert "Chu" in report.team_members


class TestRenderMarkdown:
    """Tests for render_markdown method."""

    def test_render_markdown_structure(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that markdown has expected sections."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        md = service.render_markdown(report)

        # Check major sections are present
        assert "# Deployment Report: vinci" in md
        assert "## Executive Summary" in md
        assert "## Deployment Details" in md
        assert "## Timeline Overview" in md
        assert "## Key Events" in md
        assert "## Issues Encountered" in md
        assert "## Action Items" in md
        assert "## Insights & Observations" in md
        assert "## Full Transcript" in md

    def test_render_markdown_contains_data(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that markdown contains actual data."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        md = service.render_markdown(report)

        # Check events
        assert "Deployment Started" in md
        assert "Battery Low Warning" in md

        # Check action items
        assert "Replace GoPro 1 battery" in md
        assert "Document Starlink setup" in md

        # Check insights
        assert "Pre-deployment battery checks" in md

        # Check transcript
        assert "We need to check the battery levels" in md
        assert "Damion Shelton" in md

    def test_render_markdown_empty_data(self, service, sample_deployment):
        """Test markdown renders gracefully with no data."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=[],
            utterances=[],
            events=[],
            action_items=[],
            insights=[],
        )

        md = service.render_markdown(report)

        assert "*No key events recorded.*" in md
        assert "*No issues recorded.*" in md
        assert "*No action items recorded.*" in md
        assert "*No insights recorded.*" in md
        assert "*No transcript available.*" in md


class TestRenderHTML:
    """Tests for render_html method."""

    def test_render_html_valid_structure(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that HTML has valid structure."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        html = service.render_html(report)

        # Check valid HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html lang=\"en\">" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_render_html_contains_title(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that HTML contains correct title."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        html = service.render_html(report)

        assert "<title>Deployment Report: vinci</title>" in html
        assert "<h1>Deployment Report: vinci</h1>" in html

    def test_render_html_contains_css(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that HTML contains embedded CSS."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        html = service.render_html(report)

        assert "<style>" in html
        assert "</style>" in html
        assert "--primary-color" in html  # CSS variable

    def test_render_html_contains_data(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names
    ):
        """Test that HTML contains actual data."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        html = service.render_html(report)

        # Check metrics
        assert "3h 0m" in html
        assert "2" in html  # team members

        # Check events table
        assert "Deployment Started" in html
        assert "<table>" in html
        assert "<td>" in html

    def test_render_html_print_styles(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that HTML includes print-friendly styles."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        html = service.render_html(report)

        assert "@media print" in html


class TestRenderPDF:
    """Tests for render_pdf method."""

    def test_render_pdf_requires_weasyprint(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights
    ):
        """Test that PDF generation requires weasyprint."""
        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        # Mock weasyprint not being available
        with patch.dict('sys.modules', {'weasyprint': None}):
            with pytest.raises(RuntimeError, match="weasyprint is required"):
                service.render_pdf(report)


class TestGenerateWithFileIO:
    """Tests for the generate method with file I/O."""

    def test_generate_loads_deployment_by_id(
        self, service, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test that generate method loads deployment from file."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())

        # Write transcript file (empty)
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Generate report
        md = service.generate("deploy:20250119_vinci_01", format="md")

        assert "# Deployment Report: vinci" in md

    def test_generate_writes_output_file(
        self, service, sample_deployment, temp_data_dir
    ):
        """Test that generate method writes to output path."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write deployment file
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Generate report with output path
        output_path = Path(temp_data_dir) / "output" / "report.md"
        service.generate("deploy:20250119_vinci_01", format="md", output_path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Deployment Report: vinci" in content

    def test_generate_html_format(
        self, service, sample_deployment, temp_data_dir
    ):
        """Test generating HTML format."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        html = service.generate("deploy:20250119_vinci_01", format="html")

        assert "<!DOCTYPE html>" in html
        assert "<title>Deployment Report: vinci</title>" in html

    def test_generate_loads_all_data(
        self, service, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names, temp_data_dir
    ):
        """Test that generate method loads all deployment data."""
        # Setup deployment directory
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        # Write all data files
        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())
        (deploy_dir / "sources.json").write_text(
            json.dumps([s.model_dump(mode="json") for s in sample_sources], default=str)
        )
        (deploy_dir / "canonical" / "transcript.json").write_text(
            json.dumps([u.model_dump(mode="json") for u in sample_utterances], default=str)
        )
        (deploy_dir / "events.json").write_text(
            json.dumps([e.model_dump(mode="json") for e in sample_events], default=str)
        )
        (deploy_dir / "action_items.json").write_text(
            json.dumps([a.model_dump(mode="json") for a in sample_action_items], default=str)
        )
        (deploy_dir / "insights.json").write_text(
            json.dumps([i.model_dump(mode="json") for i in sample_insights], default=str)
        )

        # Write people registry
        people_data = {"people": [{"id": k, "name": v} for k, v in people_names.items()]}
        (Path(temp_data_dir) / "people.json").write_text(json.dumps(people_data))

        md = service.generate("deploy:20250119_vinci_01", format="md")

        # Verify all data appears in the report
        assert "Deployment Started" in md
        assert "Replace GoPro 1 battery" in md
        assert "Pre-deployment battery checks" in md
        assert "We need to check the battery levels" in md
        assert "Damion Shelton" in md

    def test_generate_deployment_not_found(self, service, temp_data_dir):
        """Test that generate raises error for missing deployment."""
        with pytest.raises(FileNotFoundError):
            service.generate("deploy:20250119_nonexistent_01")


class TestFormatTime:
    """Tests for time formatting."""

    def test_format_time_zero(self, service):
        """Test formatting zero milliseconds."""
        assert service._format_time(0) == "00:00:00"

    def test_format_time_seconds(self, service):
        """Test formatting seconds."""
        assert service._format_time(1000) == "00:00:01"
        assert service._format_time(30000) == "00:00:30"

    def test_format_time_minutes(self, service):
        """Test formatting minutes."""
        assert service._format_time(60000) == "00:01:00"
        assert service._format_time(90000) == "00:01:30"

    def test_format_time_hours(self, service):
        """Test formatting hours."""
        assert service._format_time(3600000) == "01:00:00"
        assert service._format_time(3661000) == "01:01:01"

    def test_format_time_large_value(self, service):
        """Test formatting large time values."""
        # 10 hours, 30 minutes, 45 seconds
        ms = (10 * 3600 + 30 * 60 + 45) * 1000
        assert service._format_time(ms) == "10:30:45"


class TestJinja2Templates:
    """Tests for Jinja2 template rendering."""

    def test_jinja2_markdown_template(
        self, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names, temp_data_dir
    ):
        """Test that Jinja2 markdown template renders correctly."""
        # Copy templates to temp dir
        template_dir = Path(temp_data_dir) / "templates"
        template_dir.mkdir()

        # Create a simple test template
        md_template = """# Test: {{ report.deployment.location }}
Events: {{ report.key_events | length }}
"""
        (template_dir / "report.md.j2").write_text(md_template)

        service = ReportGenerator(template_dir=str(template_dir), data_dir=temp_data_dir)

        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        md = service.render_markdown(report)

        assert "# Test: vinci" in md
        assert "Events: 2" in md

    def test_jinja2_html_template(
        self, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, people_names, temp_data_dir
    ):
        """Test that Jinja2 HTML template renders correctly."""
        template_dir = Path(temp_data_dir) / "templates"
        template_dir.mkdir()

        html_template = """<!DOCTYPE html>
<html>
<head><title>{{ report.deployment.location }}</title></head>
<body><h1>{{ report.deployment.location }}</h1></body>
</html>"""
        (template_dir / "report.html.j2").write_text(html_template)

        service = ReportGenerator(template_dir=str(template_dir), data_dir=temp_data_dir)

        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
            people_names=people_names,
        )

        html = service.render_html(report)

        assert "<title>vinci</title>" in html
        assert "<h1>vinci</h1>" in html

    def test_fallback_when_no_template(
        self, sample_deployment, sample_sources, sample_utterances,
        sample_events, sample_action_items, sample_insights, temp_data_dir
    ):
        """Test that inline rendering is used when templates don't exist."""
        # Create service with non-existent template dir
        service = ReportGenerator(template_dir="/nonexistent", data_dir=temp_data_dir)

        report = service.generate_report(
            deployment=sample_deployment,
            sources=sample_sources,
            utterances=sample_utterances,
            events=sample_events,
            action_items=sample_action_items,
            insights=sample_insights,
        )

        # Should still render using inline method
        md = service.render_markdown(report)
        assert "# Deployment Report: vinci" in md


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_report_format_values(self):
        """Test ReportFormat enum values."""
        assert ReportFormat.MARKDOWN.value == "md"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.PDF.value == "pdf"

    def test_format_string_conversion(self, service, sample_deployment, temp_data_dir):
        """Test that string format is converted to enum."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)
        (deploy_dir / "canonical").mkdir()

        (deploy_dir / "deployment.json").write_text(sample_deployment.model_dump_json())
        (deploy_dir / "canonical" / "transcript.json").write_text("[]")

        # Should accept string format
        md = service.generate("deploy:20250119_vinci_01", format="md")
        assert "# Deployment Report" in md

        html = service.generate("deploy:20250119_vinci_01", format="html")
        assert "<!DOCTYPE html>" in html


class TestLoadMethods:
    """Tests for data loading methods."""

    def test_load_sources_array_format(self, service, sample_sources, temp_data_dir):
        """Test loading sources from array format."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        sources_data = [s.model_dump(mode="json") for s in sample_sources]
        (deploy_dir / "sources.json").write_text(json.dumps(sources_data, default=str))

        from gram_deploy.models import Deployment
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
        )

        sources = service._load_sources(deployment, deploy_dir)
        assert len(sources) == 2

    def test_load_sources_object_format(self, service, sample_sources, temp_data_dir):
        """Test loading sources from object format."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        sources_data = {"sources": [s.model_dump(mode="json") for s in sample_sources]}
        (deploy_dir / "sources.json").write_text(json.dumps(sources_data, default=str))

        from gram_deploy.models import Deployment
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
        )

        sources = service._load_sources(deployment, deploy_dir)
        assert len(sources) == 2

    def test_load_sources_missing_file(self, service, temp_data_dir):
        """Test loading sources when file doesn't exist."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        from gram_deploy.models import Deployment
        deployment = Deployment(
            id="deploy:20250119_vinci_01",
            location="vinci",
            date="2025-01-19",
        )

        sources = service._load_sources(deployment, deploy_dir)
        assert sources == []

    def test_load_summary(self, service, temp_data_dir):
        """Test loading deployment summary."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True)

        summary_text = "This is a deployment summary."
        (deploy_dir / "summary.md").write_text(summary_text)

        summary = service._load_summary(deploy_dir)
        assert summary == summary_text

    def test_load_people_names(self, service, people_names, temp_data_dir):
        """Test loading people names from registry."""
        people_data = {"people": [{"id": k, "name": v} for k, v in people_names.items()]}
        (Path(temp_data_dir) / "people.json").write_text(json.dumps(people_data))

        names = service._load_people_names()
        assert names == people_names
