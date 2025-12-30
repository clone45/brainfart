"""Tests for LocalMemory class."""

import pytest
import tempfile
from pathlib import Path

from brainfart import LocalMemory, MemorySettings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def settings(temp_dir):
    """Create test settings."""
    return MemorySettings(
        data_dir=temp_dir,
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        similarity_threshold=0.5,
    )


@pytest.fixture
async def memory(settings):
    """Create and load a memory instance."""
    mem = LocalMemory(
        settings=settings,
        user_id="test-user",
        agent_id="test-agent",
    )
    await mem.load()
    yield mem
    await mem.close()


@pytest.mark.asyncio
async def test_store_and_retrieve(memory):
    """Test storing and retrieving a memory."""
    # Store a memory
    memory_id = await memory.store(
        content="User lives in San Francisco",
        category="identity",
        importance=5,
    )
    assert memory_id >= 0

    # Retrieve it
    results = await memory.retrieve("Where does the user live?")
    assert len(results) > 0
    assert "San Francisco" in results[0].content


@pytest.mark.asyncio
async def test_store_batch(memory):
    """Test storing multiple memories at once."""
    memories = [
        {"content": "User works at Acme Corp", "category": "identity", "importance": 4},
        {"content": "User likes coffee", "category": "preference", "importance": 2},
        {"content": "User has a dog named Max", "category": "identity", "importance": 3},
    ]

    ids = await memory.store_batch(memories)
    assert len(ids) == 3

    # Check stats
    stats = memory.get_stats()
    assert stats["total_memories"] == 3


@pytest.mark.asyncio
async def test_retrieve_with_category_filter(memory):
    """Test filtering retrieval by category."""
    # Store different categories
    await memory.store("User is a software engineer", "identity", 4)
    await memory.store("User prefers dark mode", "preference", 2)

    # Retrieve only identity
    results = await memory.retrieve(
        "What does the user do?",
        categories=["identity"],
    )

    for r in results:
        assert r.category == "identity"


@pytest.mark.asyncio
async def test_get_identity_memories(memory):
    """Test getting identity/preference memories."""
    await memory.store("User lives in NYC", "identity", 5)
    await memory.store("User likes Python", "preference", 3)
    await memory.store("Working on a project", "context", 2)

    identities = await memory.get_identity_memories()

    # Should only get identity and preference
    categories = {m.category for m in identities}
    assert "context" not in categories


@pytest.mark.asyncio
async def test_persistence(settings, temp_dir):
    """Test that memories persist across instances."""
    # Create and store
    mem1 = LocalMemory(settings=settings, user_id="persist-test", agent_id="test")
    await mem1.load()
    await mem1.store("User's favorite color is blue", "preference", 3)
    await mem1.save()
    await mem1.close()

    # Create new instance and retrieve
    mem2 = LocalMemory(settings=settings, user_id="persist-test", agent_id="test")
    await mem2.load()

    results = await mem2.retrieve("What is the user's favorite color?")
    assert len(results) > 0
    assert "blue" in results[0].content.lower()

    await mem2.close()


@pytest.mark.asyncio
async def test_encryption(temp_dir):
    """Test that encryption works."""
    settings = MemorySettings(
        data_dir=temp_dir,
        encryption_key="test-secret-key",
        similarity_threshold=0.5,
    )

    mem = LocalMemory(settings=settings, user_id="encrypt-test", agent_id="test")
    await mem.load()

    # Store encrypted
    await mem.store("Super secret information", "identity", 5)
    await mem.save()
    await mem.close()

    # Raw file should not contain plaintext
    db_path = temp_dir / "test" / "encrypt-test.db"
    if db_path.exists():
        raw_content = db_path.read_bytes()
        assert b"Super secret information" not in raw_content

    # But we can still retrieve it
    mem2 = LocalMemory(settings=settings, user_id="encrypt-test", agent_id="test")
    await mem2.load()
    results = await mem2.retrieve("secret")
    assert len(results) > 0
    assert "secret" in results[0].content.lower()
    await mem2.close()


@pytest.mark.asyncio
async def test_empty_retrieve(memory):
    """Test retrieval with no stored memories."""
    results = await memory.retrieve("random query")
    assert results == []


@pytest.mark.asyncio
async def test_stats(memory):
    """Test getting statistics."""
    stats = memory.get_stats()
    assert stats["loaded"] is True
    assert stats["user_id"] == "test-user"
    assert stats["agent_id"] == "test-agent"
    assert stats["total_memories"] == 0
