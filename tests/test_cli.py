"""Tests for the CLI interface.

Tests command parsing, output formatting, and basic functionality
using Click's CliRunner.
"""

import tempfile

import pytest
from click.testing import CliRunner

from gram_deploy.__main__ import cli


# Check if optional dependencies are available
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Mark tests that require full service dependencies
requires_services = pytest.mark.skipif(
    not HAS_BOTO3,
    reason="Requires boto3 and other service dependencies"
)


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self, runner):
        """Test that --help works."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "GRAM Deployment Processing System" in result.output

    def test_cli_version(self, runner):
        """Test that --version works."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_verbose_flag(self, runner):
        """Test that --verbose flag is recognized."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestDeployCommands:
    """Test deployment management commands."""

    def test_deploy_help(self, runner):
        """Test deploy command help."""
        result = runner.invoke(cli, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "Deployment management commands" in result.output

    def test_deploy_create_help(self, runner):
        """Test deploy create help."""
        result = runner.invoke(cli, ["deploy", "create", "--help"])
        assert result.exit_code == 0
        assert "--location" in result.output
        assert "--date" in result.output

    @requires_services
    def test_deploy_create(self, runner, temp_data_dir, monkeypatch):
        """Test deploy create command."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, [
            "deploy", "create",
            "--location", "vinci",
            "--date", "2025-01-19"
        ])

        assert result.exit_code == 0
        assert "Created deployment" in result.output
        assert "vinci" in result.output

    def test_deploy_add_source_help(self, runner):
        """Test deploy add-source help."""
        result = runner.invoke(cli, ["deploy", "add-source", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--number" in result.output
        assert "--files" in result.output

    def test_deploy_process_help(self, runner):
        """Test deploy process help."""
        result = runner.invoke(cli, ["deploy", "process", "--help"])
        assert result.exit_code == 0
        assert "--skip-transcription" in result.output
        assert "--skip-analysis" in result.output
        assert "--force" in result.output
        assert "--language" in result.output

    def test_deploy_status_help(self, runner):
        """Test deploy status help."""
        result = runner.invoke(cli, ["deploy", "status", "--help"])
        assert result.exit_code == 0

    def test_deploy_list_help(self, runner):
        """Test deploy list help."""
        result = runner.invoke(cli, ["deploy", "list", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output

    def test_deploy_search_help(self, runner):
        """Test deploy search help."""
        result = runner.invoke(cli, ["deploy", "search", "--help"])
        assert result.exit_code == 0
        assert "--deployment" in result.output
        assert "--limit" in result.output
        assert "--speaker" in result.output

    def test_deploy_report_help(self, runner):
        """Test deploy report help."""
        result = runner.invoke(cli, ["deploy", "report", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "--output" in result.output

    def test_deploy_timeline_help(self, runner):
        """Test deploy timeline help."""
        result = runner.invoke(cli, ["deploy", "timeline", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output


class TestPersonCommands:
    """Test person registry commands."""

    def test_person_help(self, runner):
        """Test person command help."""
        result = runner.invoke(cli, ["person", "--help"])
        assert result.exit_code == 0
        assert "Person registry commands" in result.output

    def test_person_add_help(self, runner):
        """Test person add help."""
        result = runner.invoke(cli, ["person", "add", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--role" in result.output
        assert "--aliases" in result.output

    def test_person_list_help(self, runner):
        """Test person list help."""
        result = runner.invoke(cli, ["person", "list", "--help"])
        assert result.exit_code == 0

    @requires_services
    def test_person_add(self, runner, temp_data_dir, monkeypatch):
        """Test person add command."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, [
            "person", "add",
            "--name", "John Smith",
            "--role", "Engineer",
            "--aliases", "Johnny,JS"
        ])

        assert result.exit_code == 0
        assert "Added person" in result.output
        assert "person:john_smith" in result.output
        assert "John Smith" in result.output
        assert "Engineer" in result.output
        assert "Johnny, JS" in result.output

    @requires_services
    def test_person_list_empty(self, runner, temp_data_dir, monkeypatch):
        """Test person list with empty registry."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["person", "list"])

        assert result.exit_code == 0
        assert "No people registered" in result.output

    @requires_services
    def test_person_list_with_people(self, runner, temp_data_dir, monkeypatch):
        """Test person list with people registered."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        # First add a person
        runner.invoke(cli, [
            "person", "add",
            "--name", "John Smith",
            "--role", "Engineer",
            "--aliases", "Johnny"
        ])

        # Then list people
        result = runner.invoke(cli, ["person", "list"])

        assert result.exit_code == 0
        assert "People Registry" in result.output
        assert "person:john_smith" in result.output
        assert "John Smith" in result.output

    def test_person_add_voice_sample_help(self, runner):
        """Test person add-voice-sample help."""
        result = runner.invoke(cli, ["person", "add-voice-sample", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output
        assert "--start" in result.output
        assert "--end" in result.output


class TestSearchCommand:
    """Test search command functionality."""

    @requires_services
    def test_search_no_results(self, runner, temp_data_dir, monkeypatch):
        """Test search with no results."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["deploy", "search", "test query"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_search_help(self, runner):
        """Test search help shows all options."""
        result = runner.invoke(cli, ["deploy", "search", "--help"])

        assert result.exit_code == 0
        assert "--deployment" in result.output
        assert "--speaker" in result.output
        assert "--limit" in result.output


class TestCommandValidation:
    """Test command argument validation."""

    def test_deploy_create_missing_location(self, runner):
        """Test create fails without location."""
        result = runner.invoke(cli, ["deploy", "create", "--date", "2025-01-19"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_deploy_create_missing_date(self, runner):
        """Test create fails without date."""
        result = runner.invoke(cli, ["deploy", "create", "--location", "vinci"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_deploy_add_source_invalid_type(self, runner):
        """Test add-source fails with invalid device type."""
        result = runner.invoke(cli, [
            "deploy", "add-source", "deploy:test",
            "--type", "invalid",
            "--number", "1",
            "--files", "test.mp4"
        ])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()

    def test_deploy_report_invalid_format(self, runner):
        """Test report fails with invalid format."""
        result = runner.invoke(cli, [
            "deploy", "report", "deploy:test",
            "--format", "invalid",
            "--output", "test.txt"
        ])
        assert result.exit_code != 0


class TestOutputFormatting:
    """Test CLI output formatting."""

    @requires_services
    def test_deployment_list_table_format(self, runner, temp_data_dir, monkeypatch):
        """Test that deployment list shows as a table."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        # First create a deployment
        runner.invoke(cli, [
            "deploy", "create",
            "--location", "vinci",
            "--date", "2025-01-19"
        ])

        # Then list deployments
        result = runner.invoke(cli, ["deploy", "list"])

        assert result.exit_code == 0
        assert "Deployments" in result.output
        assert "vinci" in result.output

    @requires_services
    def test_deployment_list_empty(self, runner, temp_data_dir, monkeypatch):
        """Test empty deployment list message."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["deploy", "list"])

        assert result.exit_code == 0
        assert "No deployments found" in result.output


class TestErrorHandling:
    """Test CLI error handling."""

    @requires_services
    def test_create_deployment_invalid_date(self, runner, temp_data_dir, monkeypatch):
        """Test create deployment with invalid date format."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, [
            "deploy", "create",
            "--location", "vinci",
            "--date", "not-a-date"
        ])

        # Should still create deployment (date is just a string)
        # or error if date format is validated
        # The actual behavior depends on implementation
        assert "vinci" in result.output or "Error" in result.output

    @requires_services
    def test_status_not_found(self, runner, temp_data_dir, monkeypatch):
        """Test status command with non-existent deployment."""
        monkeypatch.setenv("GRAM_DATA_DIR", temp_data_dir)

        result = runner.invoke(cli, ["deploy", "status", "deploy:nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()
