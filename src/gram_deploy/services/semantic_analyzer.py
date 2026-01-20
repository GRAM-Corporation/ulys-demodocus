"""Semantic Analyzer - LLM-based extraction of insights, events, and action items.

Responsible for:
- Segmenting transcripts for analysis
- Extracting events, action items, and insights using LLM
- Merging and deduplicating results across segments
- Caching LLM responses
- Saving analysis results to deployment directory
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from gram_deploy.models import (
    ActionItem,
    ActionItemStatus,
    CanonicalUtterance,
    Deployment,
    DeploymentEvent,
    DeploymentInsight,
    EventType,
    ExtractionMethod,
    InsightType,
    Priority,
    Severity,
    SupportingEvidence,
    TimeRange,
)


@dataclass
class SemanticAnalysisResult:
    """Results from semantic analysis of a deployment."""

    events: list[DeploymentEvent]
    action_items: list[ActionItem]
    insights: list[DeploymentInsight]
    summary: Optional[str] = None


class SemanticAnalyzer:
    """Extracts semantic information from transcripts using LLM."""

    SEGMENT_DURATION_MS = 10 * 60 * 1000  # 10 minutes per segment

    def __init__(
        self,
        llm_client: Any,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
    ):
        """Initialize the analyzer.

        Args:
            llm_client: Client for LLM API (e.g., Anthropic client)
            cache_dir: Directory for caching LLM responses (defaults to data_dir/cache/llm_responses)
            data_dir: Root directory for deployment data (default: "deployments")
        """
        self.llm_client = llm_client
        self.data_dir = Path(data_dir) if data_dir else Path("deployments")

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.data_dir / "cache" / "llm_responses"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        deployment: Union[Deployment, str],
        data_dir: Optional[str] = None,
    ) -> SemanticAnalysisResult:
        """Analyze a deployment, loading transcripts and saving results.

        This is the main entry point for semantic analysis. It:
        1. Loads canonical transcript from deployment directory
        2. Runs all extraction pipelines (events, action items, insights)
        3. Generates a summary
        4. Saves results to deployment directory

        Args:
            deployment: Deployment entity or deployment ID string
            data_dir: Optional override for data directory

        Returns:
            SemanticAnalysisResult with events, action items, insights, and summary
        """
        effective_data_dir = Path(data_dir) if data_dir else self.data_dir

        # Load deployment if ID string provided
        if isinstance(deployment, str):
            deployment = self._load_deployment(deployment, effective_data_dir)

        # Load canonical utterances
        utterances = self._load_canonical_utterances(deployment, effective_data_dir)

        # Load people names for speaker mapping
        people_names = self._load_people_names(effective_data_dir)

        # Run the analysis
        result = self.analyze_deployment(deployment, utterances, people_names)

        # Save results to deployment directory
        self._save_results(result, deployment, effective_data_dir)

        return result

    def _load_deployment(
        self,
        deployment_id: str,
        data_dir: Path,
    ) -> Deployment:
        """Load a deployment from disk.

        Args:
            deployment_id: The deployment ID
            data_dir: Root data directory

        Returns:
            Deployment entity

        Raises:
            FileNotFoundError: If deployment not found
        """
        deploy_dir = self._get_deployment_dir(deployment_id, data_dir)
        deployment_path = deploy_dir / "deployment.json"

        if not deployment_path.exists():
            raise FileNotFoundError(f"Deployment not found: {deployment_id}")

        data = json.loads(deployment_path.read_text())
        return Deployment.model_validate(data)

    def _get_deployment_dir(self, deployment_id: str, data_dir: Path) -> Path:
        """Get the directory path for a deployment.

        Args:
            deployment_id: The deployment ID (e.g., "deploy:20250119_vinci_01")
            data_dir: Root data directory

        Returns:
            Path to deployment directory
        """
        # Convert deploy:20250119_vinci_01 to deploy_20250119_vinci_01
        dir_name = deployment_id.replace(":", "_")
        return data_dir / dir_name

    def _load_canonical_utterances(
        self,
        deployment: Deployment,
        data_dir: Path,
    ) -> list[CanonicalUtterance]:
        """Load canonical utterances from deployment directory.

        Args:
            deployment: The Deployment entity
            data_dir: Root data directory

        Returns:
            List of CanonicalUtterance entities
        """
        deploy_dir = self._get_deployment_dir(deployment.id, data_dir)
        transcript_path = deploy_dir / "canonical" / "transcript.json"

        if not transcript_path.exists():
            return []

        data = json.loads(transcript_path.read_text())

        # Handle both array format and object with utterances key
        if isinstance(data, list):
            utterances_data = data
        else:
            utterances_data = data.get("utterances", [])

        return [CanonicalUtterance.model_validate(u) for u in utterances_data]

    def _load_people_names(self, data_dir: Path) -> dict[str, str]:
        """Load people names mapping from registry.

        Args:
            data_dir: Root data directory

        Returns:
            Dict mapping person_id to display name
        """
        registry_path = data_dir / "people.json"

        if not registry_path.exists():
            return {}

        data = json.loads(registry_path.read_text())
        people_names = {}

        for person_data in data.get("people", []):
            person_id = person_data.get("id")
            name = person_data.get("name")
            if person_id and name:
                people_names[person_id] = name

        return people_names

    def _save_results(
        self,
        result: SemanticAnalysisResult,
        deployment: Deployment,
        data_dir: Path,
    ) -> None:
        """Save analysis results to deployment directory.

        Saves:
        - events.json: List of extracted events
        - action_items.json: List of extracted action items
        - insights.json: List of extracted insights
        - summary.md: Executive summary

        Args:
            result: The analysis results
            deployment: The Deployment entity
            data_dir: Root data directory
        """
        deploy_dir = self._get_deployment_dir(deployment.id, data_dir)

        # Save events
        self._save_events(result.events, deploy_dir)

        # Save action items
        self._save_action_items(result.action_items, deploy_dir)

        # Save insights
        self._save_insights(result.insights, deploy_dir)

        # Save summary
        if result.summary:
            self._save_summary(result.summary, deploy_dir)

    def _save_events(
        self,
        events: list[DeploymentEvent],
        deploy_dir: Path,
    ) -> None:
        """Save events to deployment/events.json.

        Args:
            events: List of DeploymentEvent entities
            deploy_dir: Deployment directory path
        """
        events_path = deploy_dir / "events.json"
        events_data = [e.model_dump(mode="json") for e in events]
        events_path.write_text(json.dumps(events_data, indent=2, default=str))

    def _save_action_items(
        self,
        action_items: list[ActionItem],
        deploy_dir: Path,
    ) -> None:
        """Save action items to deployment/action_items.json.

        Args:
            action_items: List of ActionItem entities
            deploy_dir: Deployment directory path
        """
        items_path = deploy_dir / "action_items.json"
        items_data = [item.model_dump(mode="json") for item in action_items]
        items_path.write_text(json.dumps(items_data, indent=2, default=str))

    def _save_insights(
        self,
        insights: list[DeploymentInsight],
        deploy_dir: Path,
    ) -> None:
        """Save insights to deployment/insights.json.

        Args:
            insights: List of DeploymentInsight entities
            deploy_dir: Deployment directory path
        """
        insights_path = deploy_dir / "insights.json"
        insights_data = [i.model_dump(mode="json") for i in insights]
        insights_path.write_text(json.dumps(insights_data, indent=2, default=str))

    def _save_summary(
        self,
        summary: str,
        deploy_dir: Path,
    ) -> None:
        """Save summary to deployment/summary.md.

        Args:
            summary: The summary text
            deploy_dir: Deployment directory path
        """
        summary_path = deploy_dir / "summary.md"
        summary_path.write_text(summary)

    def analyze_deployment(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
        people_names: Optional[dict[str, str]] = None,
    ) -> SemanticAnalysisResult:
        """Perform comprehensive analysis of a deployment.

        Args:
            deployment: The Deployment entity
            utterances: Canonical utterances to analyze
            people_names: Optional mapping of person_id -> display name

        Returns:
            SemanticAnalysisResult with events, action items, and insights
        """
        if not utterances:
            return SemanticAnalysisResult(events=[], action_items=[], insights=[])

        people_names = people_names or {}

        # Segment transcript for analysis
        segments = self._segment_utterances(utterances)

        all_events: list[DeploymentEvent] = []
        all_action_items: list[ActionItem] = []
        all_insights: list[DeploymentInsight] = []

        for segment_start_ms, segment_utterances in segments:
            # Build transcript text for this segment
            transcript_text = self._build_transcript_text(
                segment_utterances, people_names
            )

            # Get speaker names in this segment
            speaker_names = self._get_speaker_names(segment_utterances, people_names)

            # Extract events
            events = self._extract_events(
                deployment.id,
                segment_start_ms,
                transcript_text,
            )
            all_events.extend(events)

            # Extract action items
            action_items = self._extract_action_items(
                deployment.id,
                segment_start_ms,
                transcript_text,
                speaker_names,
                segment_utterances,
            )
            all_action_items.extend(action_items)

            # Extract insights
            insights = self._extract_insights(
                deployment.id,
                segment_start_ms,
                transcript_text,
            )
            all_insights.extend(insights)

        # Deduplicate and merge results
        events = self._deduplicate_events(all_events)
        action_items = self._deduplicate_action_items(all_action_items)
        insights = self._deduplicate_insights(all_insights)

        # Generate summary
        summary = self._generate_summary(deployment, utterances, events, people_names)

        return SemanticAnalysisResult(
            events=events,
            action_items=action_items,
            insights=insights,
            summary=summary,
        )

    def extract_action_items(
        self,
        deployment_id: str,
        utterances: list[CanonicalUtterance],
        people_names: Optional[dict[str, str]] = None,
    ) -> list[ActionItem]:
        """Extract only action items from utterances.

        Args:
            deployment_id: The deployment ID
            utterances: Utterances to analyze
            people_names: Optional mapping of person_id -> display name

        Returns:
            List of ActionItem entities
        """
        people_names = people_names or {}
        segments = self._segment_utterances(utterances)

        all_action_items: list[ActionItem] = []
        for segment_start_ms, segment_utterances in segments:
            transcript_text = self._build_transcript_text(
                segment_utterances, people_names
            )
            speaker_names = self._get_speaker_names(segment_utterances, people_names)

            action_items = self._extract_action_items(
                deployment_id,
                segment_start_ms,
                transcript_text,
                speaker_names,
                segment_utterances,
            )
            all_action_items.extend(action_items)

        return self._deduplicate_action_items(all_action_items)

    def generate_summary(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
        events: list[DeploymentEvent],
        people_names: Optional[dict[str, str]] = None,
    ) -> str:
        """Generate a narrative summary of the deployment.

        Args:
            deployment: The Deployment entity
            utterances: Canonical utterances
            events: Extracted events
            people_names: Optional mapping of person_id -> display name

        Returns:
            Summary text
        """
        return self._generate_summary(deployment, utterances, events, people_names or {})

    def _segment_utterances(
        self,
        utterances: list[CanonicalUtterance],
    ) -> list[tuple[int, list[CanonicalUtterance]]]:
        """Segment utterances into chunks for analysis.

        Returns:
            List of (segment_start_ms, utterances) tuples
        """
        if not utterances:
            return []

        segments: list[tuple[int, list[CanonicalUtterance]]] = []
        current_segment: list[CanonicalUtterance] = []
        segment_start = utterances[0].canonical_start_ms

        for utterance in utterances:
            # Check if we should start a new segment
            if utterance.canonical_start_ms - segment_start >= self.SEGMENT_DURATION_MS:
                if current_segment:
                    segments.append((segment_start, current_segment))
                current_segment = [utterance]
                segment_start = utterance.canonical_start_ms
            else:
                current_segment.append(utterance)

        if current_segment:
            segments.append((segment_start, current_segment))

        return segments

    def _build_transcript_text(
        self,
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> str:
        """Build transcript text with speaker labels."""
        lines = []
        for utterance in utterances:
            speaker = people_names.get(utterance.speaker_id, "Unknown")
            time_str = self._format_time(utterance.canonical_start_ms)
            lines.append(f"[{time_str}] {speaker}: {utterance.text}")
        return "\n".join(lines)

    def _get_speaker_names(
        self,
        utterances: list[CanonicalUtterance],
        people_names: dict[str, str],
    ) -> list[str]:
        """Get unique speaker names in utterances."""
        speakers = set()
        for utterance in utterances:
            if utterance.speaker_id:
                name = people_names.get(utterance.speaker_id, utterance.speaker_id)
                speakers.add(name)
        return list(speakers)

    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS."""
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _extract_events(
        self,
        deployment_id: str,
        segment_start_ms: int,
        transcript_text: str,
    ) -> list[DeploymentEvent]:
        """Extract events from a transcript segment using LLM."""
        prompt = f"""You are analyzing a transcript from a robotics deployment. Identify significant events that occurred.

Event types:
- milestone: A planned objective was achieved
- issue: A problem was encountered
- decision: An important choice was made
- observation: A noteworthy technical observation

For each event, provide:
- event_type: One of the types above
- title: Brief title (5-10 words)
- description: Fuller description
- time_offset_ms: Approximate time in milliseconds from the start of this segment
- severity: "info", "warning", or "critical"
- confidence: Your confidence this is a significant event (0.0-1.0)

Transcript segment (starts at {self._format_time(segment_start_ms)}):
{transcript_text}

Respond with a JSON array of events. Only include genuinely significant events, not routine activities."""

        response = self._call_llm(prompt)
        events = self._parse_events_response(response, deployment_id, segment_start_ms)
        return events

    def _extract_action_items(
        self,
        deployment_id: str,
        segment_start_ms: int,
        transcript_text: str,
        speaker_names: list[str],
        utterances: list[CanonicalUtterance],
    ) -> list[ActionItem]:
        """Extract action items from a transcript segment using LLM."""
        prompt = f"""You are analyzing a transcript from a robotics deployment. Extract all action items—tasks, follow-ups, or commitments mentioned by the team.

For each action item, provide:
- description: What needs to be done (imperative form)
- mentioned_by: Who mentioned it (use the speaker names provided)
- assigned_to: Who should do it, if stated (null if not clear)
- deadline: When it should be done, if stated (null if not clear)
- priority: "low", "medium", "high", or "critical" based on context
- time_offset_ms: Approximate time in milliseconds from the start of this segment
- confidence: Your confidence this is a real action item (0.0-1.0)

Transcript segment:
{transcript_text}

Speakers in this segment: {', '.join(speaker_names)}

Respond with a JSON array of action items. Only include genuine action items, not general statements about what the team does."""

        response = self._call_llm(prompt)
        action_items = self._parse_action_items_response(
            response, deployment_id, segment_start_ms, utterances
        )
        return action_items

    def _extract_insights(
        self,
        deployment_id: str,
        segment_start_ms: int,
        transcript_text: str,
    ) -> list[DeploymentInsight]:
        """Extract insights from a transcript segment using LLM."""
        prompt = f"""You are analyzing a transcript from a robotics deployment. Extract valuable insights.

Insight types:
- technical_observation: Technical finding or observation
- process_improvement: Suggestion for improving workflow
- risk_identified: Potential risk or concern
- success_factor: What contributed to success
- lesson_learned: Learning from experience

For each insight, provide:
- insight_type: One of the types above
- content: The insight text
- supporting_quote: A direct quote from the transcript
- confidence: Your confidence this is a valuable insight (0.0-1.0)

Transcript segment:
{transcript_text}

Respond with a JSON array of insights. Focus on actionable, specific insights rather than generic observations."""

        response = self._call_llm(prompt)
        insights = self._parse_insights_response(response, deployment_id)
        return insights

    def _generate_summary(
        self,
        deployment: Deployment,
        utterances: list[CanonicalUtterance],
        events: list[DeploymentEvent],
        people_names: dict[str, str],
    ) -> str:
        """Generate a narrative summary of the deployment."""
        # Build context
        event_summaries = [f"- {e.title}: {e.description}" for e in events[:10]]
        speakers = list(set(
            people_names.get(u.speaker_id, "Unknown")
            for u in utterances
            if u.speaker_id
        ))

        prompt = f"""Write a concise executive summary of this deployment.

Deployment: {deployment.location} on {deployment.date}
Duration: {(deployment.canonical_end_time - deployment.canonical_start_time).total_seconds() / 3600:.1f} hours
Team members: {', '.join(speakers)}

Key events:
{chr(10).join(event_summaries)}

Write 2-3 paragraphs summarizing what happened, key accomplishments, and any issues encountered. Be specific and factual."""

        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with caching."""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["response"]

        # Call LLM
        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
        except Exception as e:
            result = f"Error calling LLM: {e}"

        # Cache response
        cache_path.write_text(json.dumps({
            "prompt_hash": cache_key,
            "response": result,
            "timestamp": datetime.utcnow().isoformat(),
        }))

        return result

    def _parse_events_response(
        self,
        response: str,
        deployment_id: str,
        segment_start_ms: int,
    ) -> list[DeploymentEvent]:
        """Parse LLM response into DeploymentEvent entities."""
        try:
            # Extract JSON from response
            data = self._extract_json(response)
            if not isinstance(data, list):
                return []

            events = []
            for item in data:
                event_type = EventType(item.get("event_type", "observation"))
                severity_str = item.get("severity", "info")
                severity = Severity(severity_str) if severity_str else None

                event = DeploymentEvent(
                    id=DeploymentEvent.generate_id(deployment_id),
                    deployment_id=deployment_id,
                    event_type=event_type,
                    canonical_time_ms=segment_start_ms + item.get("time_offset_ms", 0),
                    title=item.get("title", ""),
                    description=item.get("description"),
                    severity=severity,
                    extraction_method=ExtractionMethod.LLM_EXTRACTED,
                    confidence=item.get("confidence", 0.5),
                )
                events.append(event)

            return events

        except Exception:
            return []

    def _parse_action_items_response(
        self,
        response: str,
        deployment_id: str,
        segment_start_ms: int,
        utterances: list[CanonicalUtterance],
    ) -> list[ActionItem]:
        """Parse LLM response into ActionItem entities."""
        try:
            data = self._extract_json(response)
            if not isinstance(data, list):
                return []

            action_items = []
            for item in data:
                priority_str = item.get("priority")
                priority = Priority(priority_str) if priority_str else None

                # Find closest utterance for source
                time_offset = item.get("time_offset_ms", 0)
                canonical_time = segment_start_ms + time_offset
                source_utterance = self._find_closest_utterance(
                    utterances, canonical_time
                )

                action_item = ActionItem(
                    id=ActionItem.generate_id(deployment_id),
                    deployment_id=deployment_id,
                    description=item.get("description", ""),
                    source_utterance_id=source_utterance.id if source_utterance else "",
                    canonical_time_ms=canonical_time,
                    mentioned_by=item.get("mentioned_by"),
                    assigned_to=item.get("assigned_to"),
                    priority=priority,
                    status=ActionItemStatus.EXTRACTED,
                    extraction_confidence=item.get("confidence", 0.5),
                )
                action_items.append(action_item)

            return action_items

        except Exception:
            return []

    def _parse_insights_response(
        self,
        response: str,
        deployment_id: str,
    ) -> list[DeploymentInsight]:
        """Parse LLM response into DeploymentInsight entities."""
        try:
            data = self._extract_json(response)
            if not isinstance(data, list):
                return []

            insights = []
            for item in data:
                insight_type = InsightType(
                    item.get("insight_type", "technical_observation")
                )

                evidence = []
                if item.get("supporting_quote"):
                    evidence.append(SupportingEvidence(
                        utterance_id="",
                        quote=item["supporting_quote"],
                    ))

                insight = DeploymentInsight(
                    id=DeploymentInsight.generate_id(deployment_id),
                    deployment_id=deployment_id,
                    insight_type=insight_type,
                    content=item.get("content", ""),
                    supporting_evidence=evidence,
                    confidence=item.get("confidence", 0.5),
                )
                insights.append(insight)

            return insights

        except Exception:
            return []

    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response text."""
        # Try to find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return []

    def _find_closest_utterance(
        self,
        utterances: list[CanonicalUtterance],
        time_ms: int,
    ) -> Optional[CanonicalUtterance]:
        """Find the utterance closest to a given time."""
        if not utterances:
            return None

        closest = utterances[0]
        min_diff = abs(utterances[0].canonical_start_ms - time_ms)

        for utterance in utterances[1:]:
            diff = abs(utterance.canonical_start_ms - time_ms)
            if diff < min_diff:
                min_diff = diff
                closest = utterance

        return closest

    def _deduplicate_events(
        self,
        events: list[DeploymentEvent],
    ) -> list[DeploymentEvent]:
        """Remove duplicate events."""
        seen: set[str] = set()
        unique = []
        for event in events:
            key = f"{event.event_type}:{event.title}:{event.canonical_time_ms // 60000}"
            if key not in seen:
                seen.add(key)
                unique.append(event)
        return unique

    def _deduplicate_action_items(
        self,
        items: list[ActionItem],
    ) -> list[ActionItem]:
        """Remove duplicate action items."""
        seen: set[str] = set()
        unique = []
        for item in items:
            # Normalize description for comparison
            key = item.description.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _deduplicate_insights(
        self,
        insights: list[DeploymentInsight],
    ) -> list[DeploymentInsight]:
        """Remove duplicate insights."""
        seen: set[str] = set()
        unique = []
        for insight in insights:
            key = insight.content.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(insight)
        return unique
