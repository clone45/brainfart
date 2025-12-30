# Configuration Guide

brainfart works out of the box with sensible defaults, but you can customize everything to fit your needs.

## Configuration Methods

You have three ways to configure the memory system:

### 1. Environment Variables (Recommended for Production)

```bash
# Required
export GOOGLE_API_KEY="your-gemini-key"

# Optional - all have sensible defaults
export BRAINFART_TOP_K="5"
export BRAINFART_DATA_DIR="~/.cache/brainfart"
export BRAINFART_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export BRAINFART_ENCRYPTION_KEY="your-secret"
```

### 2. Constructor Parameters (Quick Overrides)

```python
memory = MemoryProcessor(
    user_id="user-123",
    gemini_api_key="your-key",      # Override env var
    top_k=10,                        # More memories
    encryption_key="secret",         # Enable encryption
)
```

### 3. Settings Object (Full Control)

```python
from brainfart import MemoryProcessor, MemorySettings

settings = MemorySettings(
    gemini_api_key="your-key",
    gemini_model="gemini-1.5-flash",
    embedding_model="all-mpnet-base-v2",
    data_dir="/custom/path",
    top_k=10,
    similarity_threshold=0.6,
    encryption_key="secret",
)

memory = MemoryProcessor(
    user_id="user-123",
    settings=settings,
)
```

## All Configuration Options

### API Keys

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `gemini_api_key` | `GOOGLE_API_KEY` or `BRAINFART_GEMINI_API_KEY` | None (required) | Your Gemini API key for memory extraction |

### Models

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `gemini_model` | `BRAINFART_GEMINI_MODEL` | `gemini-2.0-flash-lite` | Gemini model for extraction |
| `embedding_model` | `BRAINFART_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

**Embedding Model Options:**

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 90MB | Fast | Good | Most use cases |
| `all-mpnet-base-v2` | 420MB | Medium | Better | Higher accuracy needs |
| `all-MiniLM-L12-v2` | 120MB | Fast | Good+ | Balance of speed/quality |

### Storage

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `data_dir` | `BRAINFART_DATA_DIR` | `~/.cache/brainfart` | Where to store memory files |
| `encryption_key` | `BRAINFART_ENCRYPTION_KEY` | None | Enable encryption with this passphrase |

### Retrieval

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `top_k` | `BRAINFART_TOP_K` | `5` | How many memories to retrieve |
| `similarity_threshold` | `BRAINFART_SIMILARITY_THRESHOLD` | `0.7` | Minimum relevance (0.0-1.0) |

**Understanding similarity_threshold:**
- `0.9` - Very strict, only highly relevant memories
- `0.7` - Balanced (recommended)
- `0.5` - Loose, includes somewhat related memories
- `0.3` - Very loose, might include tangentially related content

### Extraction

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `extraction_window_size` | `BRAINFART_EXTRACTION_WINDOW_SIZE` | `10` | Messages to analyze at once |
| `extraction_trigger_interval` | `BRAINFART_EXTRACTION_TRIGGER_INTERVAL` | `5` | Extract every N messages |

## Common Configurations

### Development (Fast Iteration)

```python
memory = MemoryProcessor(
    user_id="dev-user",
    extraction_interval=2,  # Extract more often for testing
)
```

### Production (Balanced)

```bash
export GOOGLE_API_KEY="your-key"
export BRAINFART_ENCRYPTION_KEY="your-secret"
export BRAINFART_DATA_DIR="/data/memories"
```

```python
memory = MemoryProcessor(user_id=user_id, agent_id="prod-bot")
```

### High-Quality Memories

```python
settings = MemorySettings(
    embedding_model="all-mpnet-base-v2",  # Better embeddings
    top_k=10,                              # More context
    similarity_threshold=0.8,              # Higher relevance
)

memory = MemoryProcessor(user_id="user-123", settings=settings)
```

### Minimal Resource Usage

```python
settings = MemorySettings(
    top_k=3,                               # Fewer memories
    extraction_trigger_interval=10,        # Less frequent extraction
    extraction_window_size=5,              # Smaller analysis window
)

memory = MemoryProcessor(user_id="user-123", settings=settings)
```

## Using .env Files

Create a `.env` file in your project root:

```env
# .env
GOOGLE_API_KEY=your-gemini-key
BRAINFART_TOP_K=5
BRAINFART_DATA_DIR=/data/memories
BRAINFART_ENCRYPTION_KEY=your-secret-passphrase
```

The settings will be loaded automatically!

## Runtime Configuration

Some settings can be changed after initialization:

```python
# Change how many memories to retrieve
results = await memory.get_memories("query", k=10)

# Disable memory injection temporarily
memory.inject_memories = False

# Disable extraction temporarily
memory.extract_memories_enabled = False
```

## Validating Configuration

Check your current settings:

```python
stats = memory.get_stats()
print(f"Data directory: {memory._settings.data_dir}")
print(f"Encryption enabled: {stats.get('encryption_enabled', False)}")
print(f"Total memories: {stats.get('total_memories', 0)}")
```

---

Next: [Encryption Guide](./encryption.md) | [Multi-Agent Setup](./multi-agent.md)
