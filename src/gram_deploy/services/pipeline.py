"""Pipeline Orchestrator - coordinates the full processing pipeline.

Responsible for:
- Running the complete processing pipeline for a deployment
- Managing checkpoints for resumable processing
- Coordinating all service components
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from gram_deploy.models import (
    Deployment,
    DeploymentStatus,
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


class PipelineOrchestrator:
    """Coordinates the full deployment processing pipeline."""

    CHECKPOINTS = [
        "audio_extraction",
        "transcription",
        "alignment",
        "speaker_resolution",
        "transcript_merge",
        "semantic_analysis",
        "search_index",
        "visualization",
        "report",
    ]

    def __init__(self, config: dict[str, Any]):
        """Initialize the orchestrator.

        Args:
            config: Configuration dict with paths and API keys
        """
        self.config = config
        self.data_dir = Path(config.get("data_dir", "./deployments"))

        # Initialize services lazily
        self._deployment_manager = None
        self._audio_extractor = None
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
    def audio_extractor(self):
        """Get or create AudioExtractor."""
        if self._audio_extractor is None:
            from gram_deploy.services.audio_extractor import AudioExtractor
            cache_dir = self.data_dir / "cache" / "audio"
            self._audio_extractor = AudioExtractor(str(cache_dir))
        return self._audio_extractor

    @property
    def transcription_service(self):
        """Get or create TranscriptionService."""
        if self._transcription_service is None:
            from gram_deploy.services.transcription_service import TranscriptionService
            provider = self.config.get("transcription_provider", "elevenlabs")
            api_key = self.config.get(f"{provider}_api_key", "")
            cache_dir = self.data_dir / "cache" / "transcripts"
            self._transcription_service = TranscriptionService(provider, api_key, str(cache_dir))
        return self._transcription_service

    @property
    def alignment_service(self):
        """Get or create TimeAlignmentService."""
        if self._alignment_service is None:
            from gram_deploy.services.time_alignment import TimeAlignmentService
            cache_dir = self.data_dir / "cache" / "alignment"
            self._alignment_service = TimeAlignmentService(str(cache_dir))
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
            client = anthropic.Anthropic(api_key=self.config.get("anthropic_api_key", ""))
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

    def process_deployment(
        self,
        deployment_id: str,
        options: Optional[ProcessingOptions] = None,
    ) -> ProcessingResult:
        """Run the complete processing pipeline for a deployment.

        Args:
            deployment_id: The deployment to process
            options: Processing options

        Returns:
            ProcessingResult with status and metrics
        """
        options = options or ProcessingOptions()
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

        # Determine starting checkpoint
        start_checkpoint = 0
        if not options.force_reprocess and deployment.checkpoint:
            try:
                start_checkpoint = self.CHECKPOINTS.index(deployment.checkpoint) + 1
            except ValueError:
                start_checkpoint = 0

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
        audio_paths: dict[str, list[str]] = {}
        transcripts = []
        speaker_mappings = []
        utterances = []
        events = []
        action_items = []
        insights = []

        try:
            # Step 1: Audio Extraction
            if start_checkpoint <= 0:
                self._update_status(deployment_id, "ingesting", "audio_extraction")
                for source in sources:
                    paths = self.audio_extractor.extract_for_source(source)
                    audio_paths[source.id] = paths
                completed_checkpoints.append("audio_extraction")

            # Step 2: Transcription
            if start_checkpoint <= 1 and not options.skip_transcription:
                self._update_status(deployment_id, "transcribing", "transcription")
                for source in sources:
                    paths = audio_paths.get(source.id, [])
                    if paths:
                        transcript = self.transcription_service.transcribe_source(
                            source, paths, options.language
                        )
                        transcripts.append(transcript)
                        self._save_transcript(deployment_id, source.id, transcript)
                completed_checkpoints.append("transcription")
            else:
                # Load existing transcripts
                transcripts = self._load_transcripts(deployment_id, sources)

            # Step 3: Time Alignment
            if start_checkpoint <= 2 and not options.skip_alignment:
                self._update_status(deployment_id, "aligning", "alignment")

                # Compute fingerprints
                fingerprints = {}
                for source in sources:
                    paths = audio_paths.get(source.id, [])
                    if paths:
                        fp = self.audio_extractor.get_audio_fingerprint(paths[0])
                        fingerprints[source.id] = fp

                alignment = self.alignment_service.compute_alignment(
                    sources, transcripts, fingerprints
                )
                self.alignment_service.apply_alignment(alignment, sources)

                # Update sources with alignment
                for source in sources:
                    self._save_source(deployment_id, source)

                # Save alignment
                self._save_alignment(deployment_id, alignment)
                completed_checkpoints.append("alignment")

            # Step 4: Speaker Resolution
            if start_checkpoint <= 3:
                self._update_status(deployment_id, "aligning", "speaker_resolution")
                speaker_mappings = self.speaker_service.resolve_speakers(
                    deployment_id, sources, transcripts
                )
                self._save_speaker_mappings(deployment_id, speaker_mappings)
                completed_checkpoints.append("speaker_resolution")
            else:
                speaker_mappings = self._load_speaker_mappings(deployment_id)

            # Step 5: Transcript Merge
            if start_checkpoint <= 4:
                self._update_status(deployment_id, "analyzing", "transcript_merge")
                utterances = self.merger.merge(sources, transcripts, speaker_mappings)
                self._save_utterances(deployment_id, utterances)
                completed_checkpoints.append("transcript_merge")
            else:
                utterances = self._load_utterances(deployment_id)

            # Step 6: Semantic Analysis
            if start_checkpoint <= 5 and not options.skip_analysis:
                self._update_status(deployment_id, "analyzing", "semantic_analysis")
                people_names = self._get_people_names()
                result = self.analyzer.analyze_deployment(deployment, utterances, people_names)
                events = result.events
                action_items = result.action_items
                insights = result.insights
                self._save_analysis(deployment_id, events, action_items, insights)
                completed_checkpoints.append("semantic_analysis")
            else:
                events, action_items, insights = self._load_analysis(deployment_id)

            # Step 7: Search Index
            if start_checkpoint <= 6:
                people_names = self._get_people_names()
                self.search_builder.build_index(deployment_id, utterances, people_names)
                completed_checkpoints.append("search_index")

            # Step 8: Visualization
            if start_checkpoint <= 7:
                people_names = self._get_people_names()
                timeline_html = self.visualizer.generate_timeline_html(
                    deployment, sources, utterances, events, people_names
                )
                self._save_output(deployment_id, "timeline.html", timeline_html)
                completed_checkpoints.append("visualization")

            # Step 9: Report
            if start_checkpoint <= 8:
                people_names = self._get_people_names()
                result = self.analyzer.analyze_deployment(deployment, utterances, people_names)
                report = self.report_generator.generate_report(
                    deployment, sources, utterances, events, action_items, insights,
                    summary=result.summary, people_names=people_names
                )
                report_md = self.report_generator.render_markdown(report)
                report_html = self.report_generator.render_html(report)
                self._save_output(deployment_id, "report.md", report_md)
                self._save_output(deployment_id, "report.html", report_html)
                completed_checkpoints.append("report")

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
                },
            )

        except Exception as e:
            errors.append(str(e))
            self._update_status(deployment_id, "failed", deployment.checkpoint, str(e))

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

    def process_source(self, deployment_id: str, source_id: str) -> None:
        """Process a single source (for incremental processing)."""
        # Implementation for adding sources incrementally
        pass

    def _update_status(
        self,
        deployment_id: str,
        status: str,
        checkpoint: str,
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

    def _save_transcript(self, deployment_id: str, source_id: str, transcript) -> None:
        """Save a raw transcript."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        # Extract device part from source_id
        device_part = source_id.split("/")[-1]
        transcript_path = deploy_dir / "sources" / device_part / "raw_transcript.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(transcript.model_dump_json(indent=2))

    def _load_transcripts(self, deployment_id: str, sources):
        """Load existing transcripts for sources."""
        from gram_deploy.models import RawTranscript
        transcripts = []
        deploy_dir = self._get_deployment_dir(deployment_id)
        for source in sources:
            device_part = source.id.split("/")[-1]
            transcript_path = deploy_dir / "sources" / device_part / "raw_transcript.json"
            if transcript_path.exists():
                data = json.loads(transcript_path.read_text())
                transcripts.append(RawTranscript.model_validate(data))
        return transcripts

    def _save_source(self, deployment_id: str, source) -> None:
        """Save source entity."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        device_part = source.id.split("/")[-1]
        source_path = deploy_dir / "sources" / device_part / "source.json"
        source_path.write_text(source.model_dump_json(indent=2))

    def _save_alignment(self, deployment_id: str, alignment) -> None:
        """Save alignment data."""
        deploy_dir = self._get_deployment_dir(deployment_id)
        alignment_path = deploy_dir / "canonical" / "alignment.json"
        alignment_path.parent.mkdir(parents=True, exist_ok=True)
        alignment_path.write_text(alignment.model_dump_json(indent=2))

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

    def _save_analysis(self, deployment_id: str, events, action_items, insights) -> None:
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

    def _load_analysis(self, deployment_id: str):
        """Load analysis results."""
        from gram_deploy.models import ActionItem, DeploymentEvent, DeploymentInsight
        deploy_dir = self._get_deployment_dir(deployment_id)
        analysis_dir = deploy_dir / "analysis"

        events = []
        action_items = []
        insights = []

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

        return events, action_items, insights

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
