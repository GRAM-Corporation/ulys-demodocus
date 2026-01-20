"""Search Index Builder - full-text search over canonical transcripts.

Responsible for:
- Building SQLite FTS5 indexes for transcripts
- Supporting phrase, boolean, and prefix queries
- Cross-deployment search capability
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SearchResult:
    """A search result with context."""

    utterance_id: str
    deployment_id: str
    text: str
    snippet: str  # Text with highlighted matches
    canonical_time_ms: int
    speaker_name: Optional[str]
    relevance_score: float


class SearchIndexBuilder:
    """Builds and queries full-text search indexes for transcripts."""

    def __init__(self, index_dir: str):
        """Initialize the index builder.

        Args:
            index_dir: Directory for storing index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.index_dir / "search.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the SQLite database with FTS5 tables."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Create FTS5 virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
                utterance_id,
                deployment_id,
                text,
                speaker_name,
                canonical_time_ms UNINDEXED,
                content='utterances',
                content_rowid='rowid'
            )
        """)

        # Create backing content table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utterances (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                utterance_id TEXT UNIQUE,
                deployment_id TEXT,
                text TEXT,
                speaker_name TEXT,
                canonical_time_ms INTEGER
            )
        """)

        # Create deployment index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_deployment
            ON utterances(deployment_id)
        """)

        conn.commit()
        conn.close()

    def build_index(
        self,
        deployment_id: str,
        utterances: list[dict],
        people_names: Optional[dict[str, str]] = None,
    ) -> None:
        """Build search index for a deployment's transcript.

        Args:
            deployment_id: The deployment ID
            utterances: List of CanonicalUtterance dicts or objects
            people_names: Optional mapping of person_id -> display name
        """
        people_names = people_names or {}
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Remove existing entries for this deployment
        cursor.execute(
            "DELETE FROM utterances WHERE deployment_id = ?",
            (deployment_id,)
        )

        # Insert new entries
        for utterance in utterances:
            # Handle both dict and object
            if hasattr(utterance, "model_dump"):
                u = utterance.model_dump()
            else:
                u = utterance

            speaker_id = u.get("speaker_id")
            speaker_name = people_names.get(speaker_id, speaker_id)

            cursor.execute("""
                INSERT INTO utterances (utterance_id, deployment_id, text, speaker_name, canonical_time_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (
                u["id"],
                deployment_id,
                u["text"],
                speaker_name,
                u["canonical_start_ms"],
            ))

        # Rebuild FTS index
        cursor.execute("INSERT INTO utterances_fts(utterances_fts) VALUES('rebuild')")

        conn.commit()
        conn.close()

    def search(
        self,
        deployment_id: str,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SearchResult]:
        """Search the index for a specific deployment.

        Args:
            deployment_id: The deployment ID to search
            query: Search query (supports FTS5 syntax)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of SearchResult objects
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Use FTS5 MATCH query
        cursor.execute("""
            SELECT
                u.utterance_id,
                u.deployment_id,
                u.text,
                snippet(utterances_fts, 2, '<mark>', '</mark>', '...', 32) as snippet,
                u.canonical_time_ms,
                u.speaker_name,
                bm25(utterances_fts) as score
            FROM utterances_fts
            JOIN utterances u ON utterances_fts.rowid = u.rowid
            WHERE utterances_fts MATCH ? AND u.deployment_id = ?
            ORDER BY score
            LIMIT ? OFFSET ?
        """, (query, deployment_id, limit, offset))

        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                utterance_id=row[0],
                deployment_id=row[1],
                text=row[2],
                snippet=row[3],
                canonical_time_ms=row[4],
                speaker_name=row[5],
                relevance_score=-row[6],  # BM25 returns negative scores
            ))

        conn.close()
        return results

    def search_across_deployments(
        self,
        query: str,
        deployment_ids: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SearchResult]:
        """Search across multiple deployments.

        Args:
            query: Search query
            deployment_ids: Optional list of deployment IDs (None = all)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of SearchResult objects
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        if deployment_ids:
            placeholders = ",".join("?" * len(deployment_ids))
            cursor.execute(f"""
                SELECT
                    u.utterance_id,
                    u.deployment_id,
                    u.text,
                    snippet(utterances_fts, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    u.canonical_time_ms,
                    u.speaker_name,
                    bm25(utterances_fts) as score
                FROM utterances_fts
                JOIN utterances u ON utterances_fts.rowid = u.rowid
                WHERE utterances_fts MATCH ? AND u.deployment_id IN ({placeholders})
                ORDER BY score
                LIMIT ? OFFSET ?
            """, (query, *deployment_ids, limit, offset))
        else:
            cursor.execute("""
                SELECT
                    u.utterance_id,
                    u.deployment_id,
                    u.text,
                    snippet(utterances_fts, 2, '<mark>', '</mark>', '...', 32) as snippet,
                    u.canonical_time_ms,
                    u.speaker_name,
                    bm25(utterances_fts) as score
                FROM utterances_fts
                JOIN utterances u ON utterances_fts.rowid = u.rowid
                WHERE utterances_fts MATCH ?
                ORDER BY score
                LIMIT ? OFFSET ?
            """, (query, limit, offset))

        results = []
        for row in cursor.fetchall():
            results.append(SearchResult(
                utterance_id=row[0],
                deployment_id=row[1],
                text=row[2],
                snippet=row[3],
                canonical_time_ms=row[4],
                speaker_name=row[5],
                relevance_score=-row[6],
            ))

        conn.close()
        return results

    def get_deployment_stats(self, deployment_id: str) -> dict:
        """Get statistics for a deployment's index.

        Returns:
            Dict with utterance_count, word_count, speaker_count
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as utterance_count,
                COUNT(DISTINCT speaker_name) as speaker_count
            FROM utterances
            WHERE deployment_id = ?
        """, (deployment_id,))

        row = cursor.fetchone()
        conn.close()

        return {
            "utterance_count": row[0],
            "speaker_count": row[1],
        }

    def delete_deployment_index(self, deployment_id: str) -> None:
        """Remove a deployment from the index."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM utterances WHERE deployment_id = ?",
            (deployment_id,)
        )
        cursor.execute("INSERT INTO utterances_fts(utterances_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()
