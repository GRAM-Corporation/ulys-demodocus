"""Pipeline Orchestrator - coordinates the full processing pipeline.

Responsible for:
- Running the complete processing pipeline for a deployment
- Managing checkpoints for resumable processing
- Coordinating all service components
- Progress callbacks for CLI display
- Parallel processing for transcription
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from gram_deploy.models import (
    Deployment,
    DeploymentStatus,
    RawTranscript,
    Source,
)


@dataclass
class ProcessingOptions:
    """Options for processing a deployment."""

    skip_transcription: bool = False
    skip_alignment: bool = False
    skip_analysis: bool = False
    force_reprocess: bool = False
    transcription_provider: str = "elevenlabs"
    language: str = "en"
    max_parallel_transcriptions: int = 3


@dataclass
class ProcessingResult:
    """Result of processing a deployment."""

    success: bool
    deployment_id: str
    status: DeploymentStatus
    duration_seconds: float
    checkpoints_completed: list[str]
    errors: list[str]
    metrics: dict


class ProgressCallbacks(Protocol):
    """Protocol for progress callbacks."""

    def on_stage_start(self, stage_name: str) -> None:
        """Called when a stage begins."""
        ...

    def on_stage_complete(self, stage_name: str, duration: float) -> None:
        """Called when a stage completes successfully."""
        ...

    def on_error(self, stage_name: str, error: Exception) -> None:
        """Called when a stage encounters an error."""
        ...


@dataclass
class DefaultProgressCallbacks:
    """Default no-op progress callbacks."""

    def on_stage_start(self, stage_name: str) -> None:
        """Called when a stage begins."""
        pass

    def on_stage_complete(self, stage_name: str, duration: float) -> None:
        """Called when a stage completes successfully."""
        pass

    def on_error(self, stage_name: str, error: Exception) -> None:
        """Called when a stage encounters an error."""
        pass


class PipelineOrchestrator:
    """Coordinates the full deployment processing pipeline.

    Implements the processing stages as specified:
    - transcribing: S3 → ElevenLabs (parallel per source)
    - aligning: Transcript-based time synchronization
    - resolving_speakers: Speaker identification
    - merging: Create canonical transcript
    - analyzing: LLM-based semantic extraction
    - indexing: Build FTS5 search index
    - visualizing: Generate timeline HTML
    - reporting: Generate deployment report

    Note: No audio extraction stage - transcription uses S3 presigned URLs directly.
    """

    # Stage names and their corresponding checkpoint values
    STAGES = [
        "transcribing",
        "aligning",
        "resolving_speakers",
        "merging",
        "analyzing",
        "indexing",
        "visualizing",
        "reporting",
    ]

    # Mapping for legacy checkpoint names
    CHECKPOINT_ALIASES = {
        "audio_extraction": None,  # Skip legacy audio extraction
        "transcription": "transcribing",
        "alignment": "aligning",
        "speaker_resolution": "resolving_speakers",
        "transcript_merge": "merging",
        "semantic_analysis": "analyzing",
        "search_index": "indexing",
        "visualization": "visualizing",
        "report": "reporting",
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize the orchestrator.

        Args:
            config: Configuration dict with paths and API keys
        """
        self.config = config
        self.data_dir = Path(config.get("data_dir", "./deployments"))

        # Initialize services lazily
        self._deployment_manager = None
        self._transcription_service = None
        self._alignment_service = None
        self._speaker_service = None
        self._merger = None
        self._analyzer = None
        self._search_builder = None
        self._visualizer = None
        self._report_generator = None

    @property
    def deployment_manager(self):
        """Get or create DeploymentManager."""
        if self._deployment_manager is None:
            from gram_deploy.services.deployment_manager import DeploymentManager

            self._deployment_manager = DeploymentManager(str(self.data_dir))
        return self._deployment_manager

    @property
    def transcription_service(self):
        """Get or create TranscriptionService."""
        if self._transcription_service is None:
            from gram_deploy.services.transcription_service import TranscriptionService

            provider = self.config.get("transcription_provider", "elevenlabs")
            api_key = self.config.get(f"{provider}_api_key", "")
            self._transcription_service = TranscriptionService(
                provider, api_key, str(self.data_dir)
            )
        return self._transcription_service

    @property
    def alignment_service(self):
        """Get or create TimeAlignmentService."""
        if self._alignment_service is None:
            from gram_deploy.services.time_alignment import TimeAlignmentService

            self._alignment_service = TimeAlignmentService(str(self.data_dir))
        return self._alignment_service

    @property
    def speaker_service(self):
        """Get or create SpeakerResolutionService."""
        if self._speaker_service is None:
            from gram_deploy.services.speaker_resolution import SpeakerResolutionService

            registry_path = self.data_dir / "people.json"
            self._speaker_service = SpeakerResolutionService(str(registry_path))
        return self._speaker_service

    @property
    def merger(self):
        """Get or create TranscriptMerger."""
        if self._merger is None:
            from gram_deploy.services.transcript_merger import TranscriptMerger

            self._merger = TranscriptMerger()
        return self._merger

    @property
    def analyzer(self):
        """Get or create SemanticAnalyzer."""
        if self._analyzer is None:
            from gram_deploy.services.semantic_analyzer import SemanticAnalyzer
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.config.get("anthropic_api_key", "")
            )
            cache_dir = self.data_dir / "cache" / "llm"
            self._analyzer = SemanticAnalyzer(client, str(cache_dir))
        return self._analyzer

    @property
    def search_builder(self):
        """Get or create SearchIndexBuilder."""
        if self._search_builder is None:
            from gram_deploy.services.search_index import SearchIndexBuilder

            index_dir = self.data_dir / "search_index"
            self._search_builder = SearchIndexBuilder(str(index_dir))
        return self._search_builder

    @property
    def visualizer(self):
        """Get or create TimelineVisualizer."""
        if self._visualizer is None:
            from gram_deploy.services.timeline_visualizer import TimelineVisualizer

            self._visualizer = TimelineVisualizer()
        return self._visualizer

    @property
    def report_generator(self):
        """Get or create ReportGenerator."""
        if self._report_generator is None:
            from gram_deploy.services.report_generator import ReportGenerator

            template_dir = self.data_dir / "templates"
            self._report_generator = ReportGenerator(str(template_dir))
        return self._report_generator

    def process(
        self,
        deployment_id: str,
        resume: bool = True,
        options: Optional[ProcessingOptions] = None,
        callbacks: Optional[ProgressCallbacks] = None,
    ) -> ProcessingResult:
        """Run the complete processing pipeline for a deployment.

        This is the main entry point per the spec.

        Args:
            deployment_id: The deployment to process
            resume: If True, resume from last checkpoint (default True)
            options: Processing options
            callbacks: Progress callbacks for CLI display

        Returns:
            ProcessingResult with status and metrics
        """
        return self.process_deployment(deployment_id, options, callbacks, resume=resume)

    def process_deployment(
        self,
        deployment_id: str,
        options: Optional[ProcessingOptions] = None,
        callbacks: Optional[ProgressCallbacks] = None,
        resume: bool = True,
    ) -> ProcessingResult:
        """Run the complete processing pipeline for a deployment.

        Args:
            deployment_id: The deployment to process
            options: Processing options
            callbacks: Progress callbacks for CLI display
            resume: If True, resume from last checkpoint

        Returns:
            ProcessingResult with status and metrics
        """
        options = options or ProcessingOptions()
        callbacks = callbacks or DefaultProgressCallbacks()
        start_time = datetime.utcnow()
        errors: list[str] = []
        completed_checkpoints: list[str] = []

        # Get deployment
        deployment = self.deployment_manager.get_deployment(deployment_id)
        if not deployment:
            return ProcessingResult(
                success=False,
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                duration_seconds=0,
                checkpoints_completed=[],
                errors=[f"Deployment not found: {deployment_id}"],
                metrics={},
            )

        # Determine starting stage based on checkpoint
        start_stage_index = 0
        if resume and not options.force_reprocess and deployment.checkpoint:
            start_stage_index = self._get_resume_stage_index(deployment.checkpoint)

        # Get sources
        sources = self.deployment_manager.get_sources(deployment_id)
        if not sources:
            return ProcessingResult(
                success=False,
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                duration_seconds=0,
                checkpoints_completed=[],
                errors=["No sources found for deployment"],
                metrics={},
            )

        # Initialize data containers
        transcripts: list[RawTranscript] = []
        speaker_mappings = []
        utterances = []
        events = []
        action_items = []
        insights = []
        summary = ""

        try:
            # Stage 1: Transcription
            stage_idx = 0
            if start_stage_index <= stage_idx and not options.skip_transcription:
                stage_start = time.time()
                callbacks.on_stage_start("transcribing")
                self._update_status(deployment_id, "transcribing", None)

                transcripts = self._transcribe_sources_parallel(
                    sources, options.language, options.max_parallel_transcriptions
                )

                # Save transcripts
                for source, transcript in zip(sources, transcripts):
                    self._save_transcript(deployment_id, source.id, transcript)

                self._update_status(deployment_id, "transcribing", "transcribing")
                callbacks.on_stage_complete("transcribing", time.time() - stage_start)
                completed_checkpoints.append("transcribing")
            else:
                # Load existing transcripts
                transcripts = self._load_transcripts(deployment_id, sources)

            # Stage 2: Alignment
            stage_idx = 1
            if start_stage_index <= stage_idx and not options.skip_alignment:
                stage_start = time.time()
                callbacks.on_stage_start("aligning")
                self._update_status(deployment_id, "aligning", "transcribing")

                alignments = self.alignment_service.align_sources(
                    deployment, sources, transcripts
                )

                # Apply alignment to sources
                self.alignment_service.calculate_canonical_timeline(deployment, alignments)
                self.alignment_service.save_alignment_results(deployment, alignments)

                # Save updated deployment
                self.deployment_manager.save_deployment(deployment)

                # Reload sources with updated offsets
                sources = self.deployment_manager.get_sources(deployment_id)

                self._update_status(deployment_id, "aligning", "aligning")
                callbacks.on_stage_complete("aligning", time.time() - stage_start)
                completed_checkpoints.append("aligning")

            # Stage 3: Speaker Resolution
            stage_idx = 2
            if start_stage_index <= stage_idx:
                stage_start = time.time()
                callbacks.on_stage_start("resolving_speakers")
                self._update_status(deployment_id, "aligning", "aligning")

                speaker_mappings = self.speaker_service.resolve_speakers(
                    deployment_id, sources, transcripts
                )
                self._save_speaker_mappings(deployment_id, speaker_mappings)

                self._update_status(deployment_id, "aligning", "resolving_speakers")
                callbacks.on_stage_complete(
                    "resolving_speakers", time.time() - stage_start
                )
                completed_checkpoints.append("resolving_speakers")
            else:
                speaker_mappings = self._load_speaker_mappings(deployment_id)

            # Stage 4: Transcript Merge
            stage_idx = 3
            if start_stage_index <= stage_idx:
                stage_start = time.time()
                callbacks.on_stage_start("merging")
                self._update_status(deployment_id, "analyzing", "resolving_speakers")

                utterances = self.merger.merge(sources, transcripts, speaker_mappings)
                self._save_utterances(deployment_id, utterances)

                self._update_status(deployment_id, "analyzing", "merging")
                callbacks.on_stage_complete("merging", time.time() - stage_start)
                completed_checkpoints.append("merging")
            else:
                utterances = self._load_utterances(deployment_id)

            # Stage 5: Semantic Analysis
            stage_idx = 4
            if start_stage_index <= stage_idx and not options.skip_analysis:
                stage_start = time.time()
                callbacks.on_stage_start("analyzing")
                self._update_status(deployment_id, "analyzing", "merging")

                people_names = self._get_people_names()
                result = self.analyzer.analyze_deployment(
                    deployment, utterances, people_names
                )
                events = result.events
                action_items = result.action_items
                insights = result.insights
                summary = result.summary
                self._save_analysis(deployment_id, events, action_items, insights, summary)

                self._update_status(deployment_id, "analyzing", "analyzing")
                callbacks.on_stage_complete("analyzing", time.time() - stage_start)
                completed_checkpoints.append("analyzing")
            else:
                events, action_items, insights, summary = self._load_analysis(
                    deployment_id
                )

            # Stage 6: Search Index
            stage_idx = 5
            if start_stage_index <= stage_idx:
                stage_start = time.time()
                callbacks.on_stage_start("indexing")
                self._update_status(deployment_id, "analyzing", "analyzing")

                self.search_builder.build_index(deployment)
                self.search_builder.index_utterances(deployment, utterances)
                self.search_builder.index_events(deployment, events)
                self.search_builder.index_insights(deployment, insights)

                self._update_status(deployment_id, "analyzing", "indexing")
                callbacks.on_stage_complete("indexing", time.time() - stage_start)
                completed_checkpoints.append("indexing")

            # Stage 7: Visualization
            stage_idx = 6
            if start_stage_index <= stage_idx:
                stage_start = time.time()
                callbacks.on_stage_start("visualizing")
                self._update_status(deployment_id, "analyzing", "indexing")

                people_names = self._get_people_names()
                timeline_html = self.visualizer.generate_timeline_html(
                    deployment, sources, utterances, events, people_names
                )
                self._save_output(deployment_id, "timeline.html", timeline_html)

                self._update_status(deployment_id, "analyzing", "visualizing")
                callbacks.on_stage_complete("visualizing", time.time() - stage_start)
                completed_checkpoints.append("visualizing")

            # Stage 8: Report Generation
            stage_idx = 7
            if start_stage_index <= stage_idx:
                stage_start = time.time()
                callbacks.on_stage_start("reporting")
                self._update_status(deployment_id, "analyzing", "visualizing")

                people_names = self._get_people_names()

                # Use cached summary if available
                if not summary and not options.skip_analysis:
                    _, _, _, summary = self._load_analysis(deployment_id)

                report = self.report_generator.generate_report(
                    deployment,
                    sources,
                    utterances,
                    events,
                    action_items,
                    insights,
                    summary=summary,
                    people_names=people_names,
                )
                report_md = self.report_generator.render_markdown(report)
                report_html = self.report_generator.render_html(report)
                self._save_output(deployment_id, "report.md", report_md)
                self._save_output(deployment_id, "report.html", report_html)

                self._update_status(deployment_id, "complete", "reporting")
                callbacks.on_stage_complete("reporting", time.time() - stage_start)
                completed_checkpoints.append("reporting")

            # Mark complete
            self._update_status(deployment_id, "complete", "complete")

            duration = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=True,
                deployment_id=deployment_id,
                status=DeploymentStatus.COMPLETE,
                duration_seconds=duration,
                checkpoints_completed=completed_checkpoints,
                errors=errors,
                metrics={
                    "source_count": len(sources),
                    "utterance_count": len(utterances),
                    "event_count": len(events),
                    "action_item_count": len(action_items),
                    "insight_count": len(insights),
                },
            )

        except Exception as e:
            # Determine which stage failed
            current_stage = self.STAGES[min(start_stage_index, len(self.STAGES) - 1)]
            for i, checkpoint in enumerate(self.STAGES):
                if checkpoint not in completed_checkpoints and i >= start_stage_index:
                    current_stage = checkpoint
                    break

            callbacks.on_error(current_stage, e)
            errors.append(f"{current_stage}: {str(e)}")

            # Update deployment with error
            self._update_status(
                deployment_id,
                "failed",
                deployment.checkpoint,  # Keep last successful checkpoint
                str(e),
            )

            duration = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                duration_seconds=duration,
                checkpoints_completed=completed_checkpoints,
                errors=errors,
                metrics={},
            )

    def _get_resume_stage_index(self, checkpoint: str) -> int:
        """Get the stage index to resume from based on checkpoint.

        Args:
            checkpoint: The checkpoint name from deployment

        Returns:
            Index of the next stage to run (0-based)
        """
        # Handle legacy checkpoint names
        if checkpoint in self.CHECKPOINT_ALIASES:
            aliased = self.CHECKPOINT_ALIASES[checkpoint]
            if aliased is None:
                return 0  # Skip legacy audio extraction, start from transcription
            checkpoint = aliased

        # Find stage index
        try:
            idx = self.STAGES.index(checkpoint)
            # Resume from next stage after completed checkpoint
            return idx + 1
        except ValueError:
            # Unknown checkpoint, start from beginning
            return 0

    def _transcribe_sources_parallel(
        self,
        sources: list[Source],
        language: str,
        max_workers: int = 3,
    ) -> list[RawTranscript]:
        """Transcribe multiple sources in parallel.

        Args:
            sources: List of Source entities to transcribe
            language: Language code
            max_workers: Maximum parallel transcription jobs

        Returns:
            List of RawTranscript objects in same order as sources
        """
        transcripts: dict[str, RawTranscript] = {}

        def transcribe_one(source: Source) -> tuple[str, RawTranscript]:
            transcript = self.transcription_service.transcribe(source, language=language)
            return source.id, transcript

        # Use ThreadPoolExecutor for parallel transcription
        with ThreadPoolExecutor(max_workers=min(max_workers, len(sources))) as executor:
            futures = {
                executor.submit(transcribe_one, source): source for source in sources
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    source_id, transcript = future.result()
                    transcripts[source_id] = transcript
                except Exception as e:
                    # Re-raise to be caught by main error handler
                    raise RuntimeError(
                        f"Transcription failed for source {source.id}: {e}"
                    ) from e

        # Return in original order
        return [transcripts[source.id] for source in sources]

    def process_source(self, deployment_id: str, source_id: str) -> None:
        """Process a single source (for incremental processing)."""
        # Implementation for adding sources incrementally
        pass

    def _update_status(
        self,
        deployment_id: str,
        status: str,
        checkpoint: Optional[str],
        error: Optional[str] = None,
    ) -> None:
        """Update deployment status and checkpoint."""
        self.deployment_manager.set_deployment_status(
            deployment_id, status, checkpoint, error
        )

    def _get_deployment_dir(self, deployment_id: str) -> Path:
        """Get the directory for a deployment."""
        dir_name = deployment_id.replace(":", "_")
        return self.data_dir / dir_name

    def _save_transcript(
        self, deployment_id: str, source_id: str, transcript: RawTranscript
    ) -> None:
        """Save a raw transcript."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        # Extract device part from source_id
        device_part = source_id.split("/")[-1]
        transcript_path = deploy_dir / "sources" / device_part / "raw_transcript.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(transcript.model_dump_json(indent=2))

    def _load_transcripts(
        self, deployment_id: str, sources: list[Source]
    ) -> list[RawTranscript]:
        """Load existing transcripts for sources."""
        transcripts = []
        deploy_dir = self._get_deployment_dir(deployment_id)
        for source in sources:
            device_part = source.id.split("/")[-1]
            transcript_path = (
                deploy_dir / "sources" / device_part / "raw_transcript.json"
            )
            if transcript_path.exists():
                data = json.loads(transcript_path.read_text())
                transcripts.append(RawTranscript.model_validate(data))
        return transcripts

    def _save_source(self, deployment_id: str, source: Source) -> None:
        """Save source entity."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        device_part = source.id.split("/")[-1]
        source_path = deploy_dir / "sources" / device_part / "source.json"
        source_path.write_text(source.model_dump_json(indent=2))

    def _save_speaker_mappings(self, deployment_id: str, mappings) -> None:
        """Save speaker mappings."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        mappings_path = deploy_dir / "canonical" / "speaker_mappings.json"
        mappings_path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.model_dump() for m in mappings]
        mappings_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_speaker_mappings(self, deployment_id: str):
        """Load speaker mappings."""
        from gram_deploy.models import SpeakerMapping

        deploy_dir = self._get_deployment_dir(deployment_id)
        mappings_path = deploy_dir / "canonical" / "speaker_mappings.json"
        if mappings_path.exists():
            data = json.loads(mappings_path.read_text())
            return [SpeakerMapping.model_validate(m) for m in data]
        return []

    def _save_utterances(self, deployment_id: str, utterances) -> None:
        """Save canonical utterances."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        utterances_path = deploy_dir / "canonical" / "utterances.json"
        utterances_path.parent.mkdir(parents=True, exist_ok=True)
        data = [u.model_dump() for u in utterances]
        utterances_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_utterances(self, deployment_id: str):
        """Load canonical utterances."""
        from gram_deploy.models import CanonicalUtterance

        deploy_dir = self._get_deployment_dir(deployment_id)
        utterances_path = deploy_dir / "canonical" / "utterances.json"
        if utterances_path.exists():
            data = json.loads(utterances_path.read_text())
            return [CanonicalUtterance.model_validate(u) for u in data]
        return []

    def _save_analysis(
        self,
        deployment_id: str,
        events,
        action_items,
        insights,
        summary: str = "",
    ) -> None:
        """Save analysis results."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        analysis_dir = deploy_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        (analysis_dir / "events.json").write_text(
            json.dumps([e.model_dump() for e in events], indent=2, default=str)
        )
        (analysis_dir / "action_items.json").write_text(
            json.dumps([a.model_dump() for a in action_items], indent=2, default=str)
        )
        (analysis_dir / "insights.json").write_text(
            json.dumps([i.model_dump() for i in insights], indent=2, default=str)
        )
        if summary:
            (analysis_dir / "summary.txt").write_text(summary)

    def _load_analysis(self, deployment_id: str):
        """Load analysis results."""
        from gram_deploy.models import ActionItem, DeploymentEvent, DeploymentInsight

        deploy_dir = self._get_deployment_dir(deployment_id)
        analysis_dir = deploy_dir / "analysis"

        events = []
        action_items = []
        insights = []
        summary = ""

        events_path = analysis_dir / "events.json"
        if events_path.exists():
            data = json.loads(events_path.read_text())
            events = [DeploymentEvent.model_validate(e) for e in data]

        actions_path = analysis_dir / "action_items.json"
        if actions_path.exists():
            data = json.loads(actions_path.read_text())
            action_items = [ActionItem.model_validate(a) for a in data]

        insights_path = analysis_dir / "insights.json"
        if insights_path.exists():
            data = json.loads(insights_path.read_text())
            insights = [DeploymentInsight.model_validate(i) for i in data]

        summary_path = analysis_dir / "summary.txt"
        if summary_path.exists():
            summary = summary_path.read_text()

        return events, action_items, insights, summary

    def _save_output(self, deployment_id: str, filename: str, content: str) -> None:
        """Save an output file."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        output_path = deploy_dir / "outputs" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    def _get_people_names(self) -> dict[str, str]:
        """Get mapping of person IDs to names."""
        people = self.speaker_service.list_people()
        return {p.id: p.name for p in people}
