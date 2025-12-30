# brainfart

[![PyPI version](https://img.shields.io/pypi/v/brainfart)](https://pypi.org/project/brainfart)
[![Python](https://img.shields.io/pypi/pyversions/brainfart)](https://pypi.org/project/brainfart)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

**Batteries-included memory for voice bots. One pip install, one env var, done.**

FAISS vector search + SQLite storage + sentence-transformers embeddings + Gemini extraction + at-rest encryption—all bundled and ready to use.

## Quick Start

### Installation

```bash
pip install brainfart
```

### Set your Gemini API key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Add to your Pipecat bot

```python
from pipecat.pipeline.pipeline import Pipeline
from brainfart import MemoryProcessor

# That's it—zero configuration needed
memory = MemoryProcessor(user_id="user123")

pipeline = Pipeline([
    transport.input(),
    stt_service,
    memory,           # <- Add memory here
    llm_service,
    tts_service,
    transport.output(),
])
```

## What you get

- **Vector similarity search** with FAISS (no external database)
- **Persistent storage** in SQLite (survives restarts)
- **Automatic embeddings** using all-MiniLM-L6-v2 (fast, ~90MB)
- **Smart extraction** with Gemini Flash (extracts what's worth remembering)
- **At-rest encryption** with Fernet (crash-safe, optional)

## How it works

The memory system has two modes:

1. **Extraction**: As the user speaks, the processor buffers messages and periodically uses Gemini to extract memorable facts (identity, preferences, relationships, etc.)

2. **Retrieval**: Before each LLM response, relevant memories are retrieved using semantic search and injected into the context.

```
User speaks → Buffer → Extract (Gemini) → Store (FAISS + SQLite)
                                              ↓
                                         Retrieve → Inject → LLM
```

## Configuration

Everything works with defaults, but you can customize:

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Gemini API key for extraction |
| `BRAINFART_TOP_K` | `5` | Memories to retrieve per query |
| `BRAINFART_DATA_DIR` | `~/.cache/brainfart` | Storage location |
| `BRAINFART_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `BRAINFART_ENCRYPTION_KEY` | None | Enable encryption with this key |

### Constructor Parameters

```python
memory = MemoryProcessor(
    user_id="user123",           # Required: user identifier
    agent_id="my-bot",           # Optional: for multi-agent isolation
    gemini_api_key="your-key",   # Optional: override env var
    top_k=10,                    # Optional: more memories
    embedding_model="all-mpnet-base-v2",  # Optional: larger model
    encryption_key="secret",     # Optional: enable encryption
    inject_memories=True,        # Optional: inject into LLM context
    extract_memories=True,       # Optional: extract from conversation
    extraction_interval=5,       # Optional: extract every N messages
)
```

## Encryption

Enable at-rest encryption for security:

```bash
export BRAINFART_ENCRYPTION_KEY="your-secret-passphrase"
```

Or pass directly:

```python
memory = MemoryProcessor(
    user_id="user123",
    encryption_key="your-secret-passphrase",
)
```

Data is encrypted on disk, decrypted only in memory. Crash-safe—if the process dies, data remains encrypted.

## Multi-Agent Isolation

Different agents maintain separate memories for the same user:

```python
# Sales bot memories
sales_memory = MemoryProcessor(user_id="user123", agent_id="sales-bot")

# Support bot memories (separate store)
support_memory = MemoryProcessor(user_id="user123", agent_id="support-bot")
```

Storage structure:
```
~/.cache/brainfart/
├── sales-bot/
│   └── user123.index
│   └── user123.db
└── support-bot/
    └── user123.index
    └── user123.db
```

## Direct API Usage

Use the memory system without Pipecat:

```python
from brainfart import LocalMemory, MemorySettings

settings = MemorySettings(
    data_dir="/path/to/data",
    encryption_key="secret",
)

memory = LocalMemory(settings, user_id="user123")
await memory.load()

# Store
await memory.store("User lives in SF", category="identity", importance=5)

# Retrieve
results = await memory.retrieve("Where does the user live?")
for r in results:
    print(f"[{r.category}] {r.content} (similarity: {r.similarity:.2f})")

await memory.close()
```

## Memory Categories

Memories are categorized for better retrieval:

| Category | Description | Example |
|----------|-------------|---------|
| `identity` | Core facts about who the user is | "User lives in San Francisco" |
| `preference` | Likes, dislikes, communication style | "User prefers concise answers" |
| `context` | Current projects, ongoing situations | "User is working on a Python project" |
| `relationship` | Emotional moments, shared experiences | "User was excited about promotion" |
| `surprise` | Unusual or noteworthy facts | "User has visited 50 countries" |

## Performance

- **First import**: 5-15 seconds (model loading)
- **Subsequent imports**: <1 second (cached)
- **Embedding**: ~7-8ms per text
- **Retrieval**: ~1-2ms (excluding embedding)
- **Install size**: ~500MB (mostly sentence-transformers/PyTorch)

## Docker

Pre-download models during build for faster cold starts:

```dockerfile
FROM python:3.11-slim

RUN pip install brainfart

# Pre-download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Set cache location
ENV HF_HOME=/app/.cache
```

## Full Example

See [examples/bot.py](examples/bot.py) for a complete working bot.

## Known Issues

- **Redundant memories**: The extraction process may create duplicate or near-duplicate memories. Deduplication is not yet implemented.

## License

BSD-2-Clause (same as Pipecat)
