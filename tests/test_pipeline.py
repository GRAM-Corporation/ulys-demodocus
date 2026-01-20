"""Tests for the PipelineOrchestrator service."""

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys

# Add src to path to enable direct module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import models directly
from gram_deploy.models import (
    Deployment,
    DeploymentStatus,
    Source,
    DeviceType,
    SourceFile,
    RawTranscript,
    TranscriptSegment,
    TranscriptSpeaker,
    CanonicalUtterance,
    UtteranceSource,
    DeploymentEvent,
    EventType,
    ActionItem,
    Priority,
    DeploymentInsight,
    InsightType,
    SpeakerMapping,
    ResolutionMethod,
)

# Import pipeline module directly to avoid services/__init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "pipeline",
    Path(__file__).parent.parent
    / "src"
    / "gram_deploy"
    / "services"
    / "pipeline.py",
)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)
PipelineOrchestrator = pipeline_module.PipelineOrchestrator
ProcessingOptions = pipeline_module.ProcessingOptions
ProcessingResult = pipeline_module.ProcessingResult
DefaultProgressCallbacks = pipeline_module.DefaultProgressCallbacks


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def config(temp_data_dir):
    """Create a basic config dict."""
    return {
        "data_dir": temp_data_dir,
        "transcription_provider": "elevenlabs",
        "elevenlabs_api_key": "test_key",
        "anthropic_api_key": "test_anthropic_key",
    }


@pytest.fixture
def orchestrator(config):
    """Create a PipelineOrchestrator instance."""
    return PipelineOrchestrator(config)


@pytest.fixture
def sample_deployment(temp_data_dir):
    """Create a sample deployment with directory structure."""
    deployment_id = "deploy:20250119_vinci_01"
    deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"

    # Create directory structure
    (deploy_dir / "sources" / "gopro_01").mkdir(parents=True)
    (deploy_dir / "sources" / "gopro_02").mkdir(parents=True)
    (deploy_dir / "canonical").mkdir(parents=True)
    (deploy_dir / "analysis").mkdir(parents=True)
    (deploy_dir / "outputs").mkdir(parents=True)
    (deploy_dir / "cache").mkdir(parents=True)

    # Create deployment object
    deployment = Deployment(
        id=deployment_id,
        location="vinci",
        date="2025-01-19",
        status=DeploymentStatus.INGESTING,
        sources=[
            "source:deploy:20250119_vinci_01/gopro_01",
            "source:deploy:20250119_vinci_01/gopro_02",
        ],
    )

    # Write deployment.json
    (deploy_dir / "deployment.json").write_text(deployment.model_dump_json(indent=2))

    return deployment


@pytest.fixture
def sample_sources(sample_deployment, temp_data_dir):
    """Create sample source objects."""
    deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"

    sources = [
        Source(
            id="source:deploy:20250119_vinci_01/gopro_01",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=1,
            files=[
                SourceFile(
                    filename="GX010001.MP4",
                    file_path="/path/to/GX010001.MP4",
                    duration_seconds=300.0,
                    start_offset_ms=0,
                )
            ],
        ),
        Source(
            id="source:deploy:20250119_vinci_01/gopro_02",
            deployment_id="deploy:20250119_vinci_01",
            device_type=DeviceType.GOPRO,
            device_number=2,
            files=[
                SourceFile(
                    filename="GX020001.MP4",
                    file_path="/path/to/GX020001.MP4",
                    duration_seconds=350.0,
                    start_offset_ms=0,
                )
            ],
        ),
    ]

    # Write source.json files
    for source in sources:
        source_name = source.id.split("/")[-1]
        source_dir = deploy_dir / "sources" / source_name
        (source_dir / "source.json").write_text(source.model_dump_json(indent=2))

    return sources


@pytest.fixture
def sample_transcripts(sample_sources):
    """Create sample transcript objects."""
    return [
        RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            language_code="en",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Hello, this is a test.",
                    start_time=0.0,
                    end_time=2.5,
                    speaker=TranscriptSpeaker(id="speaker_1"),
                    confidence=0.95,
                ),
                TranscriptSegment(
                    text="Let's check the equipment.",
                    start_time=3.0,
                    end_time=5.5,
                    speaker=TranscriptSpeaker(id="speaker_2"),
                    confidence=0.92,
                ),
            ],
            audio_duration_seconds=300.0,
        ),
        RawTranscript(
            id="transcript:source:deploy:20250119_vinci_01/gopro_02",
            source_id="source:deploy:20250119_vinci_01/gopro_02",
            language_code="en",
            transcription_service="elevenlabs",
            segments=[
                TranscriptSegment(
                    text="Everything looks good.",
                    start_time=5.0,
                    end_time=7.0,
                    speaker=TranscriptSpeaker(id="speaker_1"),
                    confidence=0.91,
                ),
            ],
            audio_duration_seconds=350.0,
        ),
    ]


@pytest.fixture
def sample_speaker_mappings():
    """Create sample speaker mappings."""
    return [
        SpeakerMapping(
            deployment_id="deploy:20250119_vinci_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            raw_speaker_id="speaker_1",
            resolved_person_id="person:alice",
            confidence=0.85,
            method=ResolutionMethod.CONTEXT_INFERENCE,
        ),
        SpeakerMapping(
            deployment_id="deploy:20250119_vinci_01",
            source_id="source:deploy:20250119_vinci_01/gopro_01",
            raw_speaker_id="speaker_2",
            resolved_person_id="person:bob",
            confidence=0.80,
            method=ResolutionMethod.CONTEXT_INFERENCE,
        ),
    ]


@pytest.fixture
def sample_utterances():
    """Create sample canonical utterances."""
    return [
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/0001",
            deployment_id="deploy:20250119_vinci_01",
            text="Hello, this is a test.",
            canonical_start_ms=0,
            canonical_end_ms=2500,
            speaker_id="person:alice",
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=0.0,
                    local_end_time=2.5,
                )
            ],
        ),
        CanonicalUtterance(
            id="utterance:deploy:20250119_vinci_01/0002",
            deployment_id="deploy:20250119_vinci_01",
            text="Let's check the equipment.",
            canonical_start_ms=3000,
            canonical_end_ms=5500,
            speaker_id="person:bob",
            sources=[
                UtteranceSource(
                    source_id="source:deploy:20250119_vinci_01/gopro_01",
                    local_start_time=3.0,
                    local_end_time=5.5,
                )
            ],
        ),
    ]


@pytest.fixture
def sample_events():
    """Create sample events."""
    return [
        DeploymentEvent(
            id="event:deploy:20250119_vinci_01/001",
            deployment_id="deploy:20250119_vinci_01",
            title="Equipment check started",
            event_type=EventType.MILESTONE,
            canonical_time_ms=3000,
            description="Team began equipment verification",
        ),
    ]


@pytest.fixture
def sample_action_items():
    """Create sample action items."""
    return [
        ActionItem(
            id="action:deploy:20250119_vinci_01/001",
            deployment_id="deploy:20250119_vinci_01",
            description="Review equipment checklist",
            source_utterance_id="utterance:deploy:20250119_vinci_01/0002",
            canonical_time_ms=3500,
            assigned_to="person:alice",
            priority=Priority.MEDIUM,
        ),
    ]


@pytest.fixture
def sample_insights():
    """Create sample insights."""
    return [
        DeploymentInsight(
            id="insight:deploy:20250119_vinci_01/001",
            deployment_id="deploy:20250119_vinci_01",
            insight_type=InsightType.TECHNICAL_OBSERVATION,
            content="Equipment setup took less time than expected.",
        ),
    ]


@dataclass
class MockProgressCallbacks:
    """Mock progress callbacks for testing."""

    started_stages: list = field(default_factory=list)
    completed_stages: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def on_stage_start(self, stage_name: str) -> None:
        self.started_stages.append(stage_name)

    def on_stage_complete(self, stage_name: str, duration: float) -> None:
        self.completed_stages.append((stage_name, duration))

    def on_error(self, stage_name: str, error: Exception) -> None:
        self.errors.append((stage_name, error))


class TestPipelineStages:
    """Tests for pipeline stage configuration."""

    def test_stages_list(self, orchestrator):
        """Test that STAGES list matches spec."""
        assert orchestrator.STAGES == [
            "transcribing",
            "aligning",
            "resolving_speakers",
            "merging",
            "analyzing",
            "indexing",
            "visualizing",
            "reporting",
        ]

    def test_no_audio_extraction_stage(self, orchestrator):
        """Test that audio extraction stage is not in STAGES."""
        assert "audio_extraction" not in orchestrator.STAGES

    def test_checkpoint_aliases_defined(self, orchestrator):
        """Test that checkpoint aliases are defined for legacy names."""
        assert "audio_extraction" in orchestrator.CHECKPOINT_ALIASES
        assert "transcription" in orchestrator.CHECKPOINT_ALIASES
        assert orchestrator.CHECKPOINT_ALIASES["audio_extraction"] is None


class TestProcessMethod:
    """Tests for the process() method."""

    def test_process_method_exists(self, orchestrator):
        """Test that process() method exists with correct signature."""
        assert hasattr(orchestrator, "process")
        assert callable(orchestrator.process)

    def test_process_returns_result_for_missing_deployment(self, orchestrator):
        """Test that process() returns failure for missing deployment."""
        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = None
        orchestrator._deployment_manager = mock_dm

        result = orchestrator.process("deploy:20250119_nonexistent_01", resume=True)

        assert result.success is False
        assert "Deployment not found" in result.errors[0]


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    def test_get_resume_stage_index_from_transcribing(self, orchestrator):
        """Test resume index from transcribing checkpoint."""
        index = orchestrator._get_resume_stage_index("transcribing")
        assert index == 1  # Should resume at aligning

    def test_get_resume_stage_index_from_analyzing(self, orchestrator):
        """Test resume index from analyzing checkpoint."""
        index = orchestrator._get_resume_stage_index("analyzing")
        assert index == 5  # Should resume at indexing

    def test_get_resume_stage_index_from_reporting(self, orchestrator):
        """Test resume index from reporting checkpoint."""
        index = orchestrator._get_resume_stage_index("reporting")
        assert index == 8  # Past all stages, nothing to do

    def test_get_resume_stage_index_legacy_audio_extraction(self, orchestrator):
        """Test that legacy audio_extraction checkpoint restarts from beginning."""
        index = orchestrator._get_resume_stage_index("audio_extraction")
        assert index == 0  # Should restart from transcribing

    def test_get_resume_stage_index_legacy_transcription(self, orchestrator):
        """Test that legacy transcription checkpoint maps to transcribing."""
        index = orchestrator._get_resume_stage_index("transcription")
        # transcription -> transcribing, so resume at next (aligning = index 1)
        assert index == 1

    def test_get_resume_stage_index_legacy_alignment(self, orchestrator):
        """Test that legacy alignment checkpoint maps to aligning."""
        index = orchestrator._get_resume_stage_index("alignment")
        assert index == 2  # Resume at resolving_speakers

    def test_get_resume_stage_index_unknown_checkpoint(self, orchestrator):
        """Test that unknown checkpoint starts from beginning."""
        index = orchestrator._get_resume_stage_index("unknown_checkpoint")
        assert index == 0

    def test_resume_skips_completed_stages(
        self, orchestrator, sample_deployment, sample_sources, temp_data_dir, config
    ):
        """Test that resume=True skips completed stages."""
        # Set checkpoint to indicate transcribing is done
        sample_deployment.checkpoint = "transcribing"
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "deployment.json").write_text(
            sample_deployment.model_dump_json(indent=2)
        )

        # Create transcripts file to load
        for source in sample_sources:
            source_name = source.id.split("/")[-1]
            transcript = RawTranscript(
                id=f"transcript:{source.id}",
                source_id=source.id,
                language_code="en",
                transcription_service="elevenlabs",
                segments=[],
                audio_duration_seconds=100.0,
            )
            transcript_path = (
                deploy_dir / "sources" / source_name / "raw_transcript.json"
            )
            transcript_path.write_text(transcript.model_dump_json(indent=2))

        callbacks = MockProgressCallbacks()

        # Set up mock services via private attributes
        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = sample_deployment
        mock_dm.get_sources.return_value = sample_sources
        orchestrator._deployment_manager = mock_dm

        mock_align = MagicMock()
        mock_align.align_sources.return_value = {}
        orchestrator._alignment_service = mock_align

        mock_speaker = MagicMock()
        mock_speaker.resolve_speakers.return_value = []
        mock_speaker.list_people.return_value = []
        orchestrator._speaker_service = mock_speaker

        mock_merger = MagicMock()
        mock_merger.merge.return_value = []
        orchestrator._merger = mock_merger

        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.events = []
        mock_result.action_items = []
        mock_result.insights = []
        mock_result.summary = "Test summary"
        mock_analyzer.analyze_deployment.return_value = mock_result
        orchestrator._analyzer = mock_analyzer

        mock_search = MagicMock()
        orchestrator._search_builder = mock_search

        mock_viz = MagicMock()
        mock_viz.generate_timeline_html.return_value = "<html></html>"
        orchestrator._visualizer = mock_viz

        mock_report = MagicMock()
        mock_report.generate_report.return_value = MagicMock()
        mock_report.render_markdown.return_value = "# Report"
        mock_report.render_html.return_value = "<html>Report</html>"
        orchestrator._report_generator = mock_report

        result = orchestrator.process_deployment(
            sample_deployment.id,
            options=ProcessingOptions(
                skip_transcription=True,
                skip_alignment=True,
                skip_analysis=True,
            ),
            callbacks=callbacks,
            resume=True,
        )

        # Transcribing should NOT be in started stages (was skipped due to checkpoint)
        assert "transcribing" not in callbacks.started_stages


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_deployment_not_found(self, orchestrator):
        """Test error when deployment doesn't exist."""
        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = None
        orchestrator._deployment_manager = mock_dm

        result = orchestrator.process_deployment("deploy:20250119_nonexistent_01")

        assert result.success is False
        assert result.status == DeploymentStatus.FAILED
        assert "Deployment not found" in result.errors[0]

    def test_no_sources_error(self, orchestrator, sample_deployment):
        """Test error when deployment has no sources."""
        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = sample_deployment
        mock_dm.get_sources.return_value = []
        orchestrator._deployment_manager = mock_dm

        result = orchestrator.process_deployment(sample_deployment.id)

        assert result.success is False
        assert result.status == DeploymentStatus.FAILED
        assert "No sources found" in result.errors[0]

    def test_stage_error_calls_callback(
        self, orchestrator, sample_deployment, sample_sources
    ):
        """Test that errors trigger on_error callback."""
        callbacks = MockProgressCallbacks()

        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = sample_deployment
        mock_dm.get_sources.return_value = sample_sources
        orchestrator._deployment_manager = mock_dm

        mock_ts = MagicMock()
        mock_ts.transcribe.side_effect = RuntimeError("API Error")
        orchestrator._transcription_service = mock_ts

        result = orchestrator.process_deployment(
            sample_deployment.id,
            callbacks=callbacks,
        )

        assert result.success is False
        assert len(callbacks.errors) == 1
        assert callbacks.errors[0][0] == "transcribing"

    def test_error_preserves_last_checkpoint(
        self, orchestrator, sample_deployment, sample_sources, sample_transcripts
    ):
        """Test that errors preserve the last successful checkpoint."""
        sample_deployment.checkpoint = "transcribing"

        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = sample_deployment
        mock_dm.get_sources.return_value = sample_sources
        orchestrator._deployment_manager = mock_dm

        mock_align = MagicMock()
        mock_align.align_sources.side_effect = RuntimeError("Alignment failed")
        orchestrator._alignment_service = mock_align

        # Set up a method to return transcripts
        with patch.object(orchestrator, "_load_transcripts", return_value=sample_transcripts):
            result = orchestrator.process_deployment(
                sample_deployment.id,
                options=ProcessingOptions(skip_transcription=True),
            )

        assert result.success is False
        # set_deployment_status should be called with failure status
        mock_dm.set_deployment_status.assert_called()
        last_call = mock_dm.set_deployment_status.call_args_list[-1]
        assert last_call.args[1] == "failed"


class TestProgressCallbacks:
    """Tests for progress callback functionality."""

    def test_default_callbacks_no_op(self):
        """Test that default callbacks do nothing."""
        callbacks = DefaultProgressCallbacks()
        # Should not raise
        callbacks.on_stage_start("test")
        callbacks.on_stage_complete("test", 1.0)
        callbacks.on_error("test", Exception("test"))

    def test_callbacks_on_stage_start(
        self, orchestrator, sample_deployment, sample_sources
    ):
        """Test that on_stage_start is called when a stage starts."""
        callbacks = MockProgressCallbacks()

        mock_dm = MagicMock()
        mock_dm.get_deployment.return_value = sample_deployment
        mock_dm.get_sources.return_value = sample_sources
        orchestrator._deployment_manager = mock_dm

        # Transcription will fail, but we should still see the stage start callback
        mock_ts = MagicMock()
        mock_ts.transcribe.side_effect = RuntimeError("Stop processing here")
        orchestrator._transcription_service = mock_ts

        result = orchestrator.process_deployment(
            sample_deployment.id,
            callbacks=callbacks,
        )

        # Check that transcribing stage was started
        assert "transcribing" in callbacks.started_stages
        # Error callback should have been called
        assert len(callbacks.errors) == 1


class TestParallelTranscription:
    """Tests for parallel transcription functionality."""

    def test_transcribe_sources_parallel_returns_ordered_results(
        self, orchestrator, sample_sources
    ):
        """Test that parallel transcription returns results in source order."""
        transcripts = [
            RawTranscript(
                id=f"transcript:{s.id}",
                source_id=s.id,
                language_code="en",
                transcription_service="elevenlabs",
                segments=[],
                audio_duration_seconds=100.0,
            )
            for s in sample_sources
        ]

        # Create a lookup dict for transcripts
        transcript_map = {t.source_id: t for t in transcripts}

        def mock_transcribe(source, language=None, provider=None):
            return transcript_map[source.id]

        mock_ts = MagicMock()
        mock_ts.transcribe.side_effect = mock_transcribe
        orchestrator._transcription_service = mock_ts

        results = orchestrator._transcribe_sources_parallel(
            sample_sources, "en", max_workers=2
        )

        # Results should be in same order as sources
        assert len(results) == len(sample_sources)
        for i, source in enumerate(sample_sources):
            assert results[i].source_id == source.id

    def test_transcribe_sources_parallel_raises_on_error(
        self, orchestrator, sample_sources
    ):
        """Test that transcription errors are propagated."""
        mock_ts = MagicMock()
        mock_ts.transcribe.side_effect = RuntimeError("API timeout")
        orchestrator._transcription_service = mock_ts

        with pytest.raises(RuntimeError, match="Transcription failed"):
            orchestrator._transcribe_sources_parallel(
                sample_sources, "en", max_workers=2
            )


class TestDataPersistence:
    """Tests for data save/load functionality."""

    def test_save_and_load_transcripts(
        self, orchestrator, sample_deployment, sample_sources, sample_transcripts, temp_data_dir
    ):
        """Test saving and loading transcripts."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Save transcripts
        for source, transcript in zip(sample_sources, sample_transcripts):
            orchestrator._save_transcript(sample_deployment.id, source.id, transcript)

        # Load transcripts
        loaded = orchestrator._load_transcripts(sample_deployment.id, sample_sources)

        assert len(loaded) == len(sample_transcripts)
        for i, transcript in enumerate(sample_transcripts):
            assert loaded[i].source_id == transcript.source_id
            assert len(loaded[i].segments) == len(transcript.segments)

    def test_save_and_load_speaker_mappings(
        self, orchestrator, sample_deployment, sample_speaker_mappings, temp_data_dir
    ):
        """Test saving and loading speaker mappings."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "canonical").mkdir(parents=True, exist_ok=True)

        # Save mappings
        orchestrator._save_speaker_mappings(
            sample_deployment.id, sample_speaker_mappings
        )

        # Load mappings
        loaded = orchestrator._load_speaker_mappings(sample_deployment.id)

        assert len(loaded) == len(sample_speaker_mappings)
        for i, mapping in enumerate(sample_speaker_mappings):
            assert loaded[i].raw_speaker_id == mapping.raw_speaker_id
            assert loaded[i].resolved_person_id == mapping.resolved_person_id

    def test_save_and_load_utterances(
        self, orchestrator, sample_deployment, sample_utterances, temp_data_dir
    ):
        """Test saving and loading canonical utterances."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "canonical").mkdir(parents=True, exist_ok=True)

        # Save utterances
        orchestrator._save_utterances(sample_deployment.id, sample_utterances)

        # Load utterances
        loaded = orchestrator._load_utterances(sample_deployment.id)

        assert len(loaded) == len(sample_utterances)
        for i, utterance in enumerate(sample_utterances):
            assert loaded[i].text == utterance.text
            assert loaded[i].speaker_id == utterance.speaker_id

    def test_save_and_load_analysis(
        self,
        orchestrator,
        sample_deployment,
        sample_events,
        sample_action_items,
        sample_insights,
        temp_data_dir,
    ):
        """Test saving and loading analysis results."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        (deploy_dir / "analysis").mkdir(parents=True, exist_ok=True)

        # Save analysis
        summary = "Test deployment summary"
        orchestrator._save_analysis(
            sample_deployment.id,
            sample_events,
            sample_action_items,
            sample_insights,
            summary,
        )

        # Load analysis
        events, action_items, insights, loaded_summary = orchestrator._load_analysis(
            sample_deployment.id
        )

        assert len(events) == len(sample_events)
        assert len(action_items) == len(sample_action_items)
        assert len(insights) == len(sample_insights)
        assert loaded_summary == summary

    def test_save_output(self, orchestrator, sample_deployment, temp_data_dir):
        """Test saving output files."""
        deploy_dir = Path(temp_data_dir) / "deploy_20250119_vinci_01"
        deploy_dir.mkdir(parents=True, exist_ok=True)

        content = "<html><body>Test</body></html>"
        orchestrator._save_output(sample_deployment.id, "test.html", content)

        output_path = deploy_dir / "outputs" / "test.html"
        assert output_path.exists()
        assert output_path.read_text() == content


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful processing result."""
        result = ProcessingResult(
            success=True,
            deployment_id="deploy:20250119_test_01",
            status=DeploymentStatus.COMPLETE,
            duration_seconds=120.5,
            checkpoints_completed=["transcribing", "aligning", "reporting"],
            errors=[],
            metrics={
                "source_count": 2,
                "utterance_count": 50,
                "event_count": 5,
            },
        )

        assert result.success is True
        assert result.status == DeploymentStatus.COMPLETE
        assert len(result.checkpoints_completed) == 3
        assert result.metrics["source_count"] == 2

    def test_failed_result(self):
        """Test creating a failed processing result."""
        result = ProcessingResult(
            success=False,
            deployment_id="deploy:20250119_test_01",
            status=DeploymentStatus.FAILED,
            duration_seconds=10.0,
            checkpoints_completed=["transcribing"],
            errors=["aligning: API timeout"],
            metrics={},
        )

        assert result.success is False
        assert result.status == DeploymentStatus.FAILED
        assert len(result.errors) == 1
        assert "API timeout" in result.errors[0]


class TestProcessingOptions:
    """Tests for ProcessingOptions dataclass."""

    def test_default_options(self):
        """Test default processing options."""
        options = ProcessingOptions()

        assert options.skip_transcription is False
        assert options.skip_alignment is False
        assert options.skip_analysis is False
        assert options.force_reprocess is False
        assert options.transcription_provider == "elevenlabs"
        assert options.language == "en"
        assert options.max_parallel_transcriptions == 3

    def test_custom_options(self):
        """Test custom processing options."""
        options = ProcessingOptions(
            skip_transcription=True,
            transcription_provider="assemblyai",
            language="es",
            max_parallel_transcriptions=5,
        )

        assert options.skip_transcription is True
        assert options.transcription_provider == "assemblyai"
        assert options.language == "es"
        assert options.max_parallel_transcriptions == 5
