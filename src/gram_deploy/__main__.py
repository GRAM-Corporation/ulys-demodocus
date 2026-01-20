"""CLI Runner for GRAM Deployment Processing System.

Usage:
    deploy create --location "vinci" --date "2025-01-19" --notes "First test"
    deploy add-source <deployment_id> --type gopro --number 1 --files *.MP4
    deploy process <deployment_id> [--skip-transcription] [--skip-analysis]
    deploy search <query> [--deployment <id>]
    deploy report <deployment_id> --format md --output report.md
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def get_config() -> dict:
    """Load configuration from environment and config file."""
    config = {
        "data_dir": os.environ.get("GRAM_DATA_DIR", "./deployments"),
        "elevenlabs_api_key": os.environ.get("ELEVENLABS_API_KEY", ""),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "transcription_provider": os.environ.get("GRAM_TRANSCRIPTION_PROVIDER", "elevenlabs"),
    }

    # Try to load from config file
    config_path = Path(config["data_dir"]) / "config.json"
    if config_path.exists():
        file_config = json.loads(config_path.read_text())
        config.update(file_config)

    return config


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """GRAM Deployment Processing System.

    Transform field deployment footage into structured intelligence.
    """
    pass


# ============================================================================
# Deployment Commands
# ============================================================================


@cli.group()
def deploy():
    """Deployment management commands."""
    pass


@deploy.command("create")
@click.option("--location", "-l", required=True, help="Deployment location name")
@click.option("--date", "-d", required=True, help="Deployment date (YYYY-MM-DD)")
@click.option("--notes", "-n", default=None, help="Optional notes")
def create_deployment(location: str, date: str, notes: Optional[str]):
    """Create a new deployment."""
    from gram_deploy.services.deployment_manager import DeploymentManager

    config = get_config()
    manager = DeploymentManager(config["data_dir"])

    try:
        deployment = manager.create_deployment(location, date, notes)
        console.print(f"[green]Created deployment:[/green] {deployment.id}")
        console.print(f"  Location: {deployment.location}")
        console.print(f"  Date: {deployment.date}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@deploy.command("add-source")
@click.argument("deployment_id")
@click.option("--type", "-t", "device_type", required=True,
              type=click.Choice(["gopro", "phone", "fixed", "drone", "other"]))
@click.option("--number", "-n", required=True, type=int, help="Device number")
@click.option("--files", "-f", required=True, multiple=True, help="Video file paths")
@click.option("--model", default=None, help="Device model name")
@click.option("--operator", default=None, help="Person ID of camera operator")
def add_source(deployment_id: str, device_type: str, number: int, files: tuple,
               model: Optional[str], operator: Optional[str]):
    """Add a video source to a deployment."""
    from gram_deploy.services.deployment_manager import DeploymentManager

    config = get_config()
    manager = DeploymentManager(config["data_dir"])

    # Expand glob patterns
    import glob
    expanded_files = []
    for pattern in files:
        matches = glob.glob(pattern)
        if matches:
            expanded_files.extend(sorted(matches))
        else:
            expanded_files.append(pattern)

    try:
        source = manager.add_source(
            deployment_id, device_type, number, expanded_files, model, operator
        )
        console.print(f"[green]Added source:[/green] {source.id}")
        console.print(f"  Files: {len(source.files)}")
        console.print(f"  Total duration: {source.total_duration / 60:.1f} minutes")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@deploy.command("process")
@click.argument("deployment_id")
@click.option("--skip-transcription", is_flag=True, help="Use existing transcripts")
@click.option("--skip-analysis", is_flag=True, help="Skip semantic analysis")
@click.option("--force", is_flag=True, help="Reprocess from beginning")
@click.option("--language", default="en", help="Language code for transcription")
def process_deployment(deployment_id: str, skip_transcription: bool, skip_analysis: bool,
                       force: bool, language: str):
    """Run the processing pipeline on a deployment."""
    from gram_deploy.services.pipeline import PipelineOrchestrator, ProcessingOptions

    config = get_config()
    orchestrator = PipelineOrchestrator(config)

    options = ProcessingOptions(
        skip_transcription=skip_transcription,
        skip_analysis=skip_analysis,
        force_reprocess=force,
        language=language,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing deployment...", total=None)

        result = orchestrator.process_deployment(deployment_id, options)

    if result.success:
        console.print(f"\n[green]Processing complete![/green]")
        console.print(f"  Duration: {result.duration_seconds:.1f}s")
        console.print(f"  Utterances: {result.metrics.get('utterance_count', 0)}")
        console.print(f"  Events: {result.metrics.get('event_count', 0)}")
        console.print(f"  Action items: {result.metrics.get('action_item_count', 0)}")
    else:
        console.print(f"\n[red]Processing failed![/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        sys.exit(1)


@deploy.command("status")
@click.argument("deployment_id")
def show_status(deployment_id: str):
    """Show the current processing status of a deployment."""
    from gram_deploy.services.deployment_manager import DeploymentManager

    config = get_config()
    manager = DeploymentManager(config["data_dir"])

    deployment = manager.get_deployment(deployment_id)
    if not deployment:
        console.print(f"[red]Deployment not found:[/red] {deployment_id}")
        sys.exit(1)

    console.print(f"[bold]{deployment.id}[/bold]")
    console.print(f"  Location: {deployment.location}")
    console.print(f"  Date: {deployment.date}")
    console.print(f"  Status: {deployment.status.value}")
    console.print(f"  Checkpoint: {deployment.checkpoint or 'None'}")
    console.print(f"  Sources: {len(deployment.sources)}")

    if deployment.error_message:
        console.print(f"  [red]Error:[/red] {deployment.error_message}")


@deploy.command("list")
@click.option("--limit", default=20, help="Maximum deployments to show")
def list_deployments(limit: int):
    """List all deployments."""
    from gram_deploy.services.deployment_manager import DeploymentManager

    config = get_config()
    manager = DeploymentManager(config["data_dir"])

    deployments = manager.list_deployments(limit=limit)

    if not deployments:
        console.print("No deployments found.")
        return

    table = Table(title="Deployments")
    table.add_column("ID", style="cyan")
    table.add_column("Location")
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Sources")

    for d in deployments:
        status_style = "green" if d.status.value == "complete" else "yellow"
        table.add_row(
            d.id,
            d.location,
            d.date,
            f"[{status_style}]{d.status.value}[/{status_style}]",
            str(len(d.sources)),
        )

    console.print(table)


@deploy.command("search")
@click.argument("query")
@click.option("--deployment", "-d", default=None, help="Limit to specific deployment")
@click.option("--limit", default=20, help="Maximum results")
def search_transcripts(query: str, deployment: Optional[str], limit: int):
    """Search across deployment transcripts."""
    from gram_deploy.services.search_index import SearchIndexBuilder

    config = get_config()
    index_dir = Path(config["data_dir"]) / "search_index"
    builder = SearchIndexBuilder(str(index_dir))

    if deployment:
        results = builder.search(deployment, query, limit=limit)
    else:
        results = builder.search_across_deployments(query, limit=limit)

    if not results:
        console.print("No results found.")
        return

    for result in results:
        time_str = f"{result.canonical_time_ms // 60000}:{(result.canonical_time_ms // 1000) % 60:02d}"
        console.print(f"\n[cyan]{result.deployment_id}[/cyan] @ {time_str}")
        console.print(f"  [bold]{result.speaker_name or 'Unknown'}:[/bold] {result.snippet}")


@deploy.command("report")
@click.argument("deployment_id")
@click.option("--format", "-f", "output_format", default="md",
              type=click.Choice(["md", "html", "pdf"]))
@click.option("--output", "-o", required=True, help="Output file path")
def generate_report(deployment_id: str, output_format: str, output: str):
    """Generate a deployment report."""
    from gram_deploy.services.deployment_manager import DeploymentManager
    from gram_deploy.services.report_generator import ReportGenerator
    from gram_deploy.services.pipeline import PipelineOrchestrator

    config = get_config()
    manager = DeploymentManager(config["data_dir"])
    orchestrator = PipelineOrchestrator(config)

    deployment = manager.get_deployment(deployment_id)
    if not deployment:
        console.print(f"[red]Deployment not found:[/red] {deployment_id}")
        sys.exit(1)

    # Load data
    sources = manager.get_sources(deployment_id)
    utterances = orchestrator._load_utterances(deployment_id)
    events, action_items, insights = orchestrator._load_analysis(deployment_id)
    people_names = orchestrator._get_people_names()

    # Generate report
    generator = ReportGenerator()
    report = generator.generate_report(
        deployment, sources, utterances, events, action_items, insights,
        people_names=people_names
    )

    # Render and save
    output_path = Path(output)
    if output_format == "md":
        content = generator.render_markdown(report)
        output_path.write_text(content)
    elif output_format == "html":
        content = generator.render_html(report)
        output_path.write_text(content)
    elif output_format == "pdf":
        content = generator.render_pdf(report)
        output_path.write_bytes(content)

    console.print(f"[green]Report saved to:[/green] {output}")


@deploy.command("timeline")
@click.argument("deployment_id")
@click.option("--output", "-o", required=True, help="Output HTML file path")
def generate_timeline(deployment_id: str, output: str):
    """Generate an interactive timeline visualization."""
    from gram_deploy.services.deployment_manager import DeploymentManager
    from gram_deploy.services.timeline_visualizer import TimelineVisualizer
    from gram_deploy.services.pipeline import PipelineOrchestrator

    config = get_config()
    manager = DeploymentManager(config["data_dir"])
    orchestrator = PipelineOrchestrator(config)

    deployment = manager.get_deployment(deployment_id)
    if not deployment:
        console.print(f"[red]Deployment not found:[/red] {deployment_id}")
        sys.exit(1)

    sources = manager.get_sources(deployment_id)
    utterances = orchestrator._load_utterances(deployment_id)
    events, _, _ = orchestrator._load_analysis(deployment_id)
    people_names = orchestrator._get_people_names()

    visualizer = TimelineVisualizer()
    html = visualizer.generate_timeline_html(
        deployment, sources, utterances, events, people_names
    )

    Path(output).write_text(html)
    console.print(f"[green]Timeline saved to:[/green] {output}")


# ============================================================================
# Person Commands
# ============================================================================


@cli.group()
def person():
    """Person registry commands."""
    pass


@person.command("add")
@click.option("--name", "-n", required=True, help="Full name")
@click.option("--role", "-r", default=None, help="Job title or role")
def add_person(name: str, role: Optional[str]):
    """Add a person to the registry."""
    from gram_deploy.models import Person
    from gram_deploy.services.speaker_resolution import SpeakerResolutionService

    config = get_config()
    registry_path = Path(config["data_dir"]) / "people.json"
    service = SpeakerResolutionService(str(registry_path))

    person = Person.from_name(name, role)
    service.add_person(person)

    console.print(f"[green]Added person:[/green] {person.id}")
    console.print(f"  Name: {person.name}")
    if role:
        console.print(f"  Role: {role}")


@person.command("list")
def list_people():
    """List all people in the registry."""
    from gram_deploy.services.speaker_resolution import SpeakerResolutionService

    config = get_config()
    registry_path = Path(config["data_dir"]) / "people.json"
    service = SpeakerResolutionService(str(registry_path))

    people = service.list_people()

    if not people:
        console.print("No people registered.")
        return

    table = Table(title="People Registry")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Voice Samples")

    for p in people:
        table.add_row(
            p.id,
            p.name,
            p.role or "-",
            str(len(p.voice_samples)),
        )

    console.print(table)


@person.command("add-voice-sample")
@click.argument("person_id")
@click.option("--source", "-s", required=True, help="Source ID")
@click.option("--start", required=True, type=float, help="Start time in seconds")
@click.option("--end", required=True, type=float, help="End time in seconds")
def add_voice_sample(person_id: str, source: str, start: float, end: float):
    """Add a verified voice sample for speaker identification."""
    from gram_deploy.services.speaker_resolution import SpeakerResolutionService

    config = get_config()
    registry_path = Path(config["data_dir"]) / "people.json"
    service = SpeakerResolutionService(str(registry_path))

    try:
        service.add_voice_sample(person_id, source, start, end, verified=True)
        console.print(f"[green]Added voice sample for {person_id}[/green]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
