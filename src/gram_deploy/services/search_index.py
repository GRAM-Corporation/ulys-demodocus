"""Search Index Builder - full-text search over canonical transcripts.

Responsible for:
- Building SQLite FTS5 indexes for transcripts, events, and insights
- Supporting phrase, boolean, and prefix queries
- Cross-deployment search capability
- Filtering by speaker and time range
"""

import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from gram_deploy.models import (
    Deployment,
    CanonicalUtterance,
    DeploymentEvent,
    DeploymentInsight,
)


class SearchResultType(str, Enum):
    """Type of search result."""
    UTTERANCE = "utterance"
    EVENT = "event"
    INSIGHT = "insight"


@dataclass
class SearchResult:
    """A search result with context."""

    result_type: SearchResultType
    result_id: str
    deployment_id: str
    text: str
    snippet: str  # Text with highlighted matches
    canonical_time_ms: Optional[int]
    relevance_score: float
    # Utterance-specific fields
    speaker_id: Optional[str] = None
    source_id: Optional[str] = None
    canonical_end_ms: Optional[int] = None
    # Event-specific fields
    event_type: Optional[str] = None
    # Insight-specific fields
    category: Optional[str] = None


class SearchIndexBuilder:
    """Builds and queries full-text search indexes for transcripts, events, and insights."""

    def __init__(self, index_dir: Optional[str] = None):
        """Initialize the index builder.

        Args:
            index_dir: Directory for storing index files. Defaults to 'deployments'.
        """
        self.index_dir = Path(index_dir) if index_dir else Path("deployments")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.index_dir / "search.db"
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self._db_path)

    def _init_database(self) -> None:
        """Initialize the SQLite database with FTS5 tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create utterances FTS5 virtual table per spec
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS utterances USING fts5(
                deployment_id,
                source_id,
                speaker_id,
                text,
                canonical_start_ms UNINDEXED,
                canonical_end_ms UNINDEXED
            )
        """)

        # Create events FTS5 virtual table per spec
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS events USING fts5(
                deployment_id,
                event_type,
                description,
                canonical_time_ms UNINDEXED
            )
        """)

        # Create insights FTS5 virtual table per spec
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS insights USING fts5(
                deployment_id,
                category,
                content
            )
        """)

        # Create metadata table for tracking indexed data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_metadata (
                deployment_id TEXT PRIMARY KEY,
                utterance_count INTEGER DEFAULT 0,
                event_count INTEGER DEFAULT 0,
                insight_count INTEGER DEFAULT 0,
                last_indexed TEXT
            )
        """)

        conn.commit()
        conn.close()

    def build_index(self, deployment: Union[Deployment, str]) -> None:
        """Create or update index for a deployment.

        This clears any existing data for the deployment and prepares
        for new indexing. Call index_utterances, index_events, and
        index_insights to populate the index.

        Args:
            deployment: Deployment object or deployment ID string
        """
        deployment_id = deployment.id if isinstance(deployment, Deployment) else deployment

        conn = self._get_connection()
        cursor = conn.cursor()

        # Clear existing data for this deployment
        cursor.execute("DELETE FROM utterances WHERE deployment_id = ?", (deployment_id,))
        cursor.execute("DELETE FROM events WHERE deployment_id = ?", (deployment_id,))
        cursor.execute("DELETE FROM insights WHERE deployment_id = ?", (deployment_id,))

        # Initialize metadata
        cursor.execute("""
            INSERT OR REPLACE INTO index_metadata (deployment_id, utterance_count, event_count, insight_count, last_indexed)
            VALUES (?, 0, 0, 0, datetime('now'))
        """, (deployment_id,))

        conn.commit()
        conn.close()

    def index_utterances(
        self,
        deployment: Union[Deployment, str],
        utterances: list[CanonicalUtterance],
    ) -> int:
        """Insert canonical utterances into FTS table.

        Args:
            deployment: Deployment object or deployment ID string
            utterances: List of CanonicalUtterance objects to index

        Returns:
            Number of utterances indexed
        """
        deployment_id = deployment.id if isinstance(deployment, Deployment) else deployment

        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for utterance in utterances:
            # Get primary source_id if available
            source_id = utterance.sources[0].source_id if utterance.sources else None

            cursor.execute("""
                INSERT INTO utterances (deployment_id, source_id, speaker_id, text, canonical_start_ms, canonical_end_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                deployment_id,
                source_id,
                utterance.speaker_id,
                utterance.text,
                str(utterance.canonical_start_ms),
                str(utterance.canonical_end_ms),
            ))
            count += 1

        # Update metadata
        cursor.execute("""
            UPDATE index_metadata SET utterance_count = ?, last_indexed = datetime('now')
            WHERE deployment_id = ?
        """, (count, deployment_id))

        conn.commit()
        conn.close()
        return count

    def index_events(
        self,
        deployment: Union[Deployment, str],
        events: list[DeploymentEvent],
    ) -> int:
        """Insert deployment events into FTS table.

        Args:
            deployment: Deployment object or deployment ID string
            events: List of DeploymentEvent objects to index

        Returns:
            Number of events indexed
        """
        deployment_id = deployment.id if isinstance(deployment, Deployment) else deployment

        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for event in events:
            # Combine title and description for search
            description = event.title
            if event.description:
                description = f"{event.title}: {event.description}"

            # Handle both Enum and string event_type
            event_type_str = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type

            cursor.execute("""
                INSERT INTO events (deployment_id, event_type, description, canonical_time_ms)
                VALUES (?, ?, ?, ?)
            """, (
                deployment_id,
                event_type_str,
                description,
                str(event.canonical_time_ms),
            ))
            count += 1

        # Update metadata
        cursor.execute("""
            UPDATE index_metadata SET event_count = ?, last_indexed = datetime('now')
            WHERE deployment_id = ?
        """, (count, deployment_id))

        conn.commit()
        conn.close()
        return count

    def index_insights(
        self,
        deployment: Union[Deployment, str],
        insights: list[DeploymentInsight],
    ) -> int:
        """Insert deployment insights into FTS table.

        Args:
            deployment: Deployment object or deployment ID string
            insights: List of DeploymentInsight objects to index

        Returns:
            Number of insights indexed
        """
        deployment_id = deployment.id if isinstance(deployment, Deployment) else deployment

        conn = self._get_connection()
        cursor = conn.cursor()

        count = 0
        for insight in insights:
            # Use insight_type as category if no explicit category
            # Handle both Enum and string insight_type
            insight_type_str = insight.insight_type.value if hasattr(insight.insight_type, 'value') else insight.insight_type
            category = insight.category or insight_type_str

            cursor.execute("""
                INSERT INTO insights (deployment_id, category, content)
                VALUES (?, ?, ?)
            """, (
                deployment_id,
                category,
                insight.content,
            ))
            count += 1

        # Update metadata
        cursor.execute("""
            UPDATE index_metadata SET insight_count = ?, last_indexed = datetime('now')
            WHERE deployment_id = ?
        """, (count, deployment_id))

        conn.commit()
        conn.close()
        return count

    def search(
        self,
        query: str,
        deployment_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Full-text search across all tables.

        Args:
            query: Search query (supports FTS5 syntax)
            deployment_id: Optional deployment ID to filter results
            limit: Maximum results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        results: list[SearchResult] = []
        conn = self._get_connection()
        cursor = conn.cursor()

        # Search utterances
        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ? AND deployment_id = ?
                ORDER BY score
                LIMIT ?
            """, (query, deployment_id, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, limit))

        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.UTTERANCE,
                result_id=f"utterance:{row[1]}/{row[0]}",
                deployment_id=row[1],
                source_id=row[2],
                speaker_id=row[3],
                text=row[4],
                canonical_time_ms=int(row[5]) if row[5] else None,
                canonical_end_ms=int(row[6]) if row[6] else None,
                snippet=row[7],
                relevance_score=-row[8],  # BM25 returns negative scores
            ))

        # Search events
        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    event_type,
                    description,
                    canonical_time_ms,
                    snippet(events, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(events) as score
                FROM events
                WHERE events MATCH ? AND deployment_id = ?
                ORDER BY score
                LIMIT ?
            """, (query, deployment_id, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    event_type,
                    description,
                    canonical_time_ms,
                    snippet(events, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(events) as score
                FROM events
                WHERE events MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, limit))

        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.EVENT,
                result_id=f"event:{row[1]}/{row[0]}",
                deployment_id=row[1],
                event_type=row[2],
                text=row[3],
                canonical_time_ms=int(row[4]) if row[4] else None,
                snippet=row[5],
                relevance_score=-row[6],
            ))

        # Search insights
        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    category,
                    content,
                    snippet(insights, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(insights) as score
                FROM insights
                WHERE insights MATCH ? AND deployment_id = ?
                ORDER BY score
                LIMIT ?
            """, (query, deployment_id, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    category,
                    content,
                    snippet(insights, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(insights) as score
                FROM insights
                WHERE insights MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, limit))

        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.INSIGHT,
                result_id=f"insight:{row[1]}/{row[0]}",
                deployment_id=row[1],
                category=row[2],
                text=row[3],
                canonical_time_ms=None,  # Insights don't have a specific time
                snippet=row[4],
                relevance_score=-row[5],
            ))

        conn.close()

        # Sort all results by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def search_by_speaker(
        self,
        query: str,
        person_id: str,
        deployment_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Filter search results by speaker.

        Args:
            query: Search query (supports FTS5 syntax)
            person_id: Person ID to filter by
            deployment_id: Optional deployment ID to filter results
            limit: Maximum results to return

        Returns:
            List of SearchResult objects from the specified speaker
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ? AND speaker_id = ? AND deployment_id = ?
                ORDER BY score
                LIMIT ?
            """, (query, person_id, deployment_id, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ? AND speaker_id = ?
                ORDER BY score
                LIMIT ?
            """, (query, person_id, limit))

        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.UTTERANCE,
                result_id=f"utterance:{row[1]}/{row[0]}",
                deployment_id=row[1],
                source_id=row[2],
                speaker_id=row[3],
                text=row[4],
                canonical_time_ms=int(row[5]) if row[5] else None,
                canonical_end_ms=int(row[6]) if row[6] else None,
                snippet=row[7],
                relevance_score=-row[8],
            ))

        conn.close()
        return results

    def search_by_timerange(
        self,
        query: str,
        start_ms: int,
        end_ms: int,
        deployment_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Filter search results by canonical time range.

        Args:
            query: Search query (supports FTS5 syntax)
            start_ms: Start of time range in canonical milliseconds
            end_ms: End of time range in canonical milliseconds
            deployment_id: Optional deployment ID to filter results
            limit: Maximum results to return

        Returns:
            List of SearchResult objects within the time range
        """
        results: list[SearchResult] = []
        conn = self._get_connection()
        cursor = conn.cursor()

        # Search utterances in time range
        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ?
                    AND deployment_id = ?
                    AND CAST(canonical_start_ms AS INTEGER) >= ?
                    AND CAST(canonical_start_ms AS INTEGER) <= ?
                ORDER BY score
                LIMIT ?
            """, (query, deployment_id, start_ms, end_ms, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    source_id,
                    speaker_id,
                    text,
                    canonical_start_ms,
                    canonical_end_ms,
                    snippet(utterances, 3, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(utterances) as score
                FROM utterances
                WHERE utterances MATCH ?
                    AND CAST(canonical_start_ms AS INTEGER) >= ?
                    AND CAST(canonical_start_ms AS INTEGER) <= ?
                ORDER BY score
                LIMIT ?
            """, (query, start_ms, end_ms, limit))

        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.UTTERANCE,
                result_id=f"utterance:{row[1]}/{row[0]}",
                deployment_id=row[1],
                source_id=row[2],
                speaker_id=row[3],
                text=row[4],
                canonical_time_ms=int(row[5]) if row[5] else None,
                canonical_end_ms=int(row[6]) if row[6] else None,
                snippet=row[7],
                relevance_score=-row[8],
            ))

        # Search events in time range
        if deployment_id:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    event_type,
                    description,
                    canonical_time_ms,
                    snippet(events, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(events) as score
                FROM events
                WHERE events MATCH ?
                    AND deployment_id = ?
                    AND CAST(canonical_time_ms AS INTEGER) >= ?
                    AND CAST(canonical_time_ms AS INTEGER) <= ?
                ORDER BY score
                LIMIT ?
            """, (query, deployment_id, start_ms, end_ms, limit))
        else:
            cursor.execute("""
                SELECT
                    rowid,
                    deployment_id,
                    event_type,
                    description,
                    canonical_time_ms,
                    snippet(events, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    bm25(events) as score
                FROM events
                WHERE events MATCH ?
                    AND CAST(canonical_time_ms AS INTEGER) >= ?
                    AND CAST(canonical_time_ms AS INTEGER) <= ?
                ORDER BY score
                LIMIT ?
            """, (query, start_ms, end_ms, limit))

        for row in cursor.fetchall():
            results.append(SearchResult(
                result_type=SearchResultType.EVENT,
                result_id=f"event:{row[1]}/{row[0]}",
                deployment_id=row[1],
                event_type=row[2],
                text=row[3],
                canonical_time_ms=int(row[4]) if row[4] else None,
                snippet=row[5],
                relevance_score=-row[6],
            ))

        conn.close()

        # Sort all results by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]

    def get_deployment_stats(self, deployment_id: str) -> dict:
        """Get statistics for a deployment's index.

        Args:
            deployment_id: The deployment ID

        Returns:
            Dict with utterance_count, event_count, insight_count, speaker_count
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get counts from metadata table
        cursor.execute("""
            SELECT utterance_count, event_count, insight_count
            FROM index_metadata
            WHERE deployment_id = ?
        """, (deployment_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return {
                "utterance_count": 0,
                "event_count": 0,
                "insight_count": 0,
                "speaker_count": 0,
            }

        # Get unique speaker count
        cursor.execute("""
            SELECT COUNT(DISTINCT speaker_id)
            FROM utterances
            WHERE deployment_id = ? AND speaker_id IS NOT NULL
        """, (deployment_id,))

        speaker_row = cursor.fetchone()
        conn.close()

        return {
            "utterance_count": row[0],
            "event_count": row[1],
            "insight_count": row[2],
            "speaker_count": speaker_row[0] if speaker_row else 0,
        }

    def delete_deployment_index(self, deployment_id: str) -> None:
        """Remove a deployment from the index.

        Args:
            deployment_id: The deployment ID to remove
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM utterances WHERE deployment_id = ?", (deployment_id,))
        cursor.execute("DELETE FROM events WHERE deployment_id = ?", (deployment_id,))
        cursor.execute("DELETE FROM insights WHERE deployment_id = ?", (deployment_id,))
        cursor.execute("DELETE FROM index_metadata WHERE deployment_id = ?", (deployment_id,))

        conn.commit()
        conn.close()

    def list_indexed_deployments(self) -> list[dict]:
        """List all indexed deployments with their statistics.

        Returns:
            List of dicts with deployment_id and counts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT deployment_id, utterance_count, event_count, insight_count, last_indexed
            FROM index_metadata
            ORDER BY last_indexed DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "deployment_id": row[0],
                "utterance_count": row[1],
                "event_count": row[2],
                "insight_count": row[3],
                "last_indexed": row[4],
            })

        conn.close()
        return results
