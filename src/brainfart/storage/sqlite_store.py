"""
SQLite metadata store with encryption support.

Stores memory text and metadata with optional content encryption.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ..crypto import MemoryCrypto


@dataclass
class MemoryRecord:
    """A single memory record with metadata."""

    id: int
    content: str
    category: str
    importance: int
    timestamp: float
    session_id: Optional[str] = None
    turn_number: Optional[int] = None


class SqliteStore:
    """
    SQLite store for memory metadata with encrypted content.

    Content is encrypted at rest, decrypted only when read.
    Uses WAL mode for better concurrent read performance.

    Usage:
        store = SqliteStore(db_path=Path("user.db"))
        store.open()
        memory_id = store.add("content", "category", 3)
        record = store.get(memory_id)
        store.close()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the SQLite store.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._next_id: int = 0

    def open(self) -> None:
        """Open database connection and initialize schema."""
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        else:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                importance INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                turn_number INTEGER
            )
        """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        self.conn.commit()

        # Get next ID
        cursor = self.conn.execute("SELECT MAX(id) FROM memories")
        max_id = cursor.fetchone()[0]
        self._next_id = (max_id + 1) if max_id is not None else 0

        logger.debug(f"Opened SQLite store: {self._next_id} memories")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def add(
        self,
        content: str,
        category: str,
        importance: int,
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> int:
        """
        Add a memory to the store.

        Content is encrypted before storage.

        Args:
            content: Memory text
            category: Category (identity, preference, context, relationship, surprise)
            importance: 1-5 scale
            session_id: Optional session ID
            turn_number: Optional turn number

        Returns:
            Memory ID
        """
        if self.conn is None:
            raise RuntimeError("Store not open")

        memory_id = self._next_id
        self._next_id += 1

        encrypted_content = MemoryCrypto.encrypt_string(content)
        timestamp = time.time()

        self.conn.execute(
            """INSERT INTO memories
               (id, content, category, importance, timestamp, session_id, turn_number)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, encrypted_content, category, importance, timestamp, session_id, turn_number),
        )

        return memory_id

    def add_batch(
        self,
        memories: List[dict],
        session_id: Optional[str] = None,
        turn_number: Optional[int] = None,
    ) -> List[int]:
        """
        Add multiple memories in a batch.

        Args:
            memories: List of dicts with keys: content, category, importance
            session_id: Optional session ID
            turn_number: Optional turn number

        Returns:
            List of memory IDs
        """
        if self.conn is None:
            raise RuntimeError("Store not open")

        if not memories:
            return []

        ids = list(range(self._next_id, self._next_id + len(memories)))
        self._next_id += len(memories)

        timestamp = time.time()
        rows = [
            (
                id_,
                MemoryCrypto.encrypt_string(m["content"]),
                m["category"],
                m.get("importance", 3),
                timestamp,
                session_id,
                turn_number,
            )
            for id_, m in zip(ids, memories)
        ]

        self.conn.executemany(
            """INSERT INTO memories
               (id, content, category, importance, timestamp, session_id, turn_number)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

        return ids

    def get(self, memory_id: int) -> Optional[MemoryRecord]:
        """Get a memory by ID."""
        if self.conn is None:
            raise RuntimeError("Store not open")

        cursor = self.conn.execute(
            """SELECT id, content, category, importance, timestamp, session_id, turn_number
               FROM memories WHERE id = ?""",
            (memory_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return MemoryRecord(
            id=row[0],
            content=MemoryCrypto.decrypt_string(row[1]),
            category=row[2],
            importance=row[3],
            timestamp=row[4],
            session_id=row[5],
            turn_number=row[6],
        )

    def get_many(self, memory_ids: List[int]) -> List[MemoryRecord]:
        """Get multiple memories by ID."""
        if self.conn is None:
            raise RuntimeError("Store not open")

        if not memory_ids:
            return []

        placeholders = ",".join("?" * len(memory_ids))
        cursor = self.conn.execute(
            f"""SELECT id, content, category, importance, timestamp, session_id, turn_number
                FROM memories WHERE id IN ({placeholders})""",
            memory_ids,
        )

        records = []
        for row in cursor:
            records.append(
                MemoryRecord(
                    id=row[0],
                    content=MemoryCrypto.decrypt_string(row[1]),
                    category=row[2],
                    importance=row[3],
                    timestamp=row[4],
                    session_id=row[5],
                    turn_number=row[6],
                )
            )

        return records

    def get_by_category(
        self,
        categories: List[str],
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """Get memories by category."""
        if self.conn is None:
            raise RuntimeError("Store not open")

        placeholders = ",".join("?" * len(categories))
        cursor = self.conn.execute(
            f"""SELECT id, content, category, importance, timestamp, session_id, turn_number
                FROM memories WHERE category IN ({placeholders})
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?""",
            (*categories, limit),
        )

        records = []
        for row in cursor:
            records.append(
                MemoryRecord(
                    id=row[0],
                    content=MemoryCrypto.decrypt_string(row[1]),
                    category=row[2],
                    importance=row[3],
                    timestamp=row[4],
                    session_id=row[5],
                    turn_number=row[6],
                )
            )

        return records

    def commit(self) -> None:
        """Commit pending changes."""
        if self.conn:
            self.conn.commit()

    @property
    def size(self) -> int:
        """Number of memories in the store."""
        return self._next_id

    def get_stats(self) -> dict:
        """Get store statistics."""
        if self.conn is None:
            return {"loaded": False}

        cursor = self.conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        )
        by_category = dict(cursor.fetchall())

        return {
            "loaded": True,
            "total_memories": self._next_id,
            "by_category": by_category,
        }
