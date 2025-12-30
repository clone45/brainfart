# API Reference

Complete reference for all classes, methods, and functions in brainfart.

## Quick Links

- [MemoryProcessor](#memoryprocessor) - Pipecat pipeline integration
- [LocalMemory](#localmemory) - Direct memory access
- [MemorySettings](#memorysettings) - Configuration
- [MemoryResult](#memoryresult) - Search results
- [MemoryCrypto](#memorycrypto) - Encryption utilities
- [EmbeddingService](#embeddingservice) - Text embeddings
- [Extraction Functions](#extraction-functions) - Memory extraction

---

## MemoryProcessor

The main class for Pipecat integration. Inherits from `FrameProcessor`.

```python
from brainfart import MemoryProcessor
```

### Constructor

```python
MemoryProcessor(
    *,
    user_id: str,
    agent_id: str = "default",
    gemini_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    top_k: Optional[int] = None,
    encryption_key: Optional[str] = None,
    settings: Optional[MemorySettings] = None,
    inject_memories: bool = True,
    extract_memories: bool = True,
    extraction_interval: int = 5,
    **kwargs,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str` | Required | Unique user identifier |
| `agent_id` | `str` | `"default"` | Agent identifier for isolation |
| `gemini_api_key` | `str` | None | Gemini API key (or use env var) |
| `embedding_model` | `str` | None | Sentence-transformers model name |
| `top_k` | `int` | None | Number of memories to retrieve |
| `encryption_key` | `str` | None | Enable encryption with this key |
| `settings` | `MemorySettings` | None | Full settings object |
| `inject_memories` | `bool` | `True` | Inject memories into LLM context |
| `extract_memories` | `bool` | `True` | Extract memories from conversation |
| `extraction_interval` | `int` | `5` | Extract every N user messages |

### Methods

#### get_memories()

Retrieve memories using semantic search.

```python
async def get_memories(
    self,
    query: str,
    k: int = None,
    categories: Optional[List[str]] = None,
) -> List[MemoryResult]
```

**Example:**
```python
results = await memory.get_memories("user's job")
for r in results:
    print(f"{r.content} (similarity: {r.similarity:.2f})")
```

#### store_memory()

Store a memory manually.

```python
async def store_memory(
    self,
    content: str,
    category: str = "context",
    importance: int = 3,
) -> int
```

**Example:**
```python
memory_id = await memory.store_memory(
    "User prefers morning meetings",
    category="preference",
    importance=3,
)
```

#### add_assistant_message()

Add an assistant message to the conversation buffer (for extraction context).

```python
def add_assistant_message(self, text: str) -> None
```

**Example:**
```python
# After LLM response
memory.add_assistant_message("Sure, I can help with that!")
```

#### get_stats()

Get memory statistics.

```python
def get_stats(self) -> dict
```

**Returns:**
```python
{
    "loaded": True,
    "agent_id": "my-bot",
    "user_id": "user-123",
    "total_memories": 42,
    "vector_count": 42,
    "by_category": {"identity": 5, "preference": 12, "context": 25},
    "encryption_enabled": True,
}
```

#### cleanup()

Save and cleanup on shutdown.

```python
async def cleanup(self) -> None
```

**Example:**
```python
# When bot session ends
await memory.cleanup()
```

---

## LocalMemory

Direct access to memory storage without Pipecat.

```python
from brainfart import LocalMemory
```

### Constructor

```python
LocalMemory(
    settings: MemorySettings,
    user_id: str,
    agent_id: str = "default",
)
```

### Methods

#### load()

Load storage from disk.

```python
async def load(self) -> float  # Returns load time in ms
```

#### store()

Store a single memory.

```python
async def store(
    self,
    content: str,
    category: str = "context",
    importance: int = 3,
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> int  # Returns memory ID
```

#### store_batch()

Store multiple memories efficiently.

```python
async def store_batch(
    self,
    memories: List[dict],
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> List[int]  # Returns memory IDs
```

**Example:**
```python
memories = [
    {"content": "User lives in NYC", "category": "identity", "importance": 5},
    {"content": "User likes Python", "category": "preference", "importance": 3},
]
ids = await memory.store_batch(memories)
```

#### retrieve()

Search for relevant memories.

```python
async def retrieve(
    self,
    query: str,
    k: int = None,
    categories: Optional[List[str]] = None,
    min_similarity: float = None,
) -> List[MemoryResult]
```

#### get_identity_memories()

Get identity and preference memories (not query-based).

```python
async def get_identity_memories(self, k: int = 10) -> List[MemoryResult]
```

#### save()

Persist to disk.

```python
async def save(self) -> None
```

#### close()

Save and close.

```python
async def close(self) -> None
```

#### get_stats()

Get statistics.

```python
def get_stats(self) -> dict
```

### Full Example

```python
from brainfart import LocalMemory, MemorySettings

settings = MemorySettings(
    data_dir="/path/to/data",
    encryption_key="secret",
)

memory = LocalMemory(settings, user_id="user-123", agent_id="my-bot")

async def main():
    await memory.load()

    # Store
    await memory.store("User works at Acme Corp", "identity", 4)

    # Retrieve
    results = await memory.retrieve("What company?")
    print(results[0].content)  # "User works at Acme Corp"

    await memory.close()
```

---

## MemorySettings

Configuration using pydantic-settings.

```python
from brainfart import MemorySettings
```

### Constructor

```python
MemorySettings(
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.0-flash-lite",
    embedding_model: str = "all-MiniLM-L6-v2",
    data_dir: Path = Path.home() / ".cache" / "brainfart",
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    encryption_key: Optional[str] = None,
    extraction_window_size: int = 10,
    extraction_trigger_interval: int = 5,
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gemini_api_key` | `str` | Gemini API key |
| `gemini_model` | `str` | Model for extraction |
| `embedding_model` | `str` | Sentence-transformers model |
| `data_dir` | `Path` | Storage directory |
| `top_k` | `int` | Memories to retrieve |
| `similarity_threshold` | `float` | Minimum relevance (0-1) |
| `encryption_key` | `str` | Encryption passphrase |
| `extraction_window_size` | `int` | Messages per extraction |
| `extraction_trigger_interval` | `int` | Extract every N messages |

### Environment Variables

Settings automatically read from environment:

```bash
GOOGLE_API_KEY=...                          # gemini_api_key fallback
BRAINFART_GEMINI_API_KEY=...                # gemini_api_key
BRAINFART_GEMINI_MODEL=...                  # gemini_model
BRAINFART_EMBEDDING_MODEL=...               # embedding_model
BRAINFART_DATA_DIR=...                      # data_dir
BRAINFART_TOP_K=...                         # top_k
BRAINFART_SIMILARITY_THRESHOLD=...          # similarity_threshold
BRAINFART_ENCRYPTION_KEY=...                # encryption_key
```

---

## MemoryResult

Search result dataclass.

```python
from brainfart import MemoryResult
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `int` | Memory ID |
| `content` | `str` | Memory text |
| `category` | `str` | Category name |
| `importance` | `int` | Importance (1-5) |
| `timestamp` | `float` | Unix timestamp |
| `similarity` | `float` | Cosine similarity (0-1) |

### Example

```python
results = await memory.retrieve("user's location")
for r in results:
    print(f"ID: {r.id}")
    print(f"Content: {r.content}")
    print(f"Category: {r.category}")
    print(f"Importance: {r.importance}")
    print(f"Similarity: {r.similarity:.2%}")
```

---

## MemoryCrypto

Encryption utilities (singleton).

```python
from brainfart import MemoryCrypto
```

### Class Methods

#### initialize()

Initialize encryption.

```python
@classmethod
def initialize(cls, key: Optional[str] = None) -> bool
```

**Returns:** `True` if encryption enabled, `False` if disabled.

#### is_enabled()

Check if encryption is active.

```python
@classmethod
def is_enabled(cls) -> bool
```

#### encrypt_string() / decrypt_string()

Encrypt/decrypt text.

```python
@classmethod
def encrypt_string(cls, plaintext: str) -> str

@classmethod
def decrypt_string(cls, ciphertext: str) -> str
```

#### encrypt_bytes() / decrypt_bytes()

Encrypt/decrypt binary data.

```python
@classmethod
def encrypt_bytes(cls, plaintext: bytes) -> bytes

@classmethod
def decrypt_bytes(cls, ciphertext: bytes) -> bytes
```

#### reset()

Reset singleton (for testing).

```python
@classmethod
def reset(cls) -> None
```

---

## EmbeddingService

Text embedding service (lazy-loaded singleton).

```python
from brainfart import EmbeddingService
```

### Constructor

```python
EmbeddingService(model_name: str = "all-MiniLM-L6-v2")
```

### Methods

#### embed()

Embed single text (async).

```python
async def embed(self, text: str) -> np.ndarray
```

#### embed_batch()

Embed multiple texts (async, efficient).

```python
async def embed_batch(self, texts: List[str]) -> np.ndarray
```

#### embed_sync() / embed_batch_sync()

Synchronous versions.

```python
def embed_sync(self, text: str) -> np.ndarray
def embed_batch_sync(self, texts: List[str]) -> np.ndarray
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dimension` | `int` | Embedding dimension (e.g., 384) |
| `model` | SentenceTransformer | Underlying model |

### Class Methods

#### is_loaded()

Check if model is loaded.

```python
@classmethod
def is_loaded(cls) -> bool
```

#### get_dimension()

Get dimension (loads model if needed).

```python
@classmethod
def get_dimension(cls) -> int
```

---

## Extraction Functions

Functions for extracting memories from conversations.

```python
from brainfart import extract_memories, extract_and_store
```

### extract_memories()

Extract memories from messages.

```python
async def extract_memories(
    messages: List[dict],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[dict]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `List[dict]` | Messages with `role` and `content` |
| `model_name` | `str` | Gemini model (default from env) |
| `api_key` | `str` | Gemini API key (default from env) |

**Returns:** List of dicts with `content`, `category`, `importance`.

**Example:**
```python
messages = [
    {"role": "user", "content": "I just moved to Austin"},
    {"role": "assistant", "content": "Welcome to Austin!"},
]

memories = await extract_memories(messages)
# [{"content": "User recently moved to Austin", "category": "identity", "importance": 4}]
```

### extract_and_store()

Extract and store in one step.

```python
async def extract_and_store(
    messages: List[dict],
    memory: LocalMemory,
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> int  # Returns count stored
```

**Example:**
```python
count = await extract_and_store(messages, memory, session_id="sess-123")
print(f"Stored {count} memories")
```

---

## Memory Categories

Standard categories used by the extraction system:

| Category | Description | Examples |
|----------|-------------|----------|
| `identity` | Core user facts | Location, job, family |
| `preference` | Likes and dislikes | "Prefers email", "Likes Python" |
| `context` | Current situations | Projects, problems, events |
| `relationship` | Emotional moments | Shared jokes, milestones |
| `surprise` | Unusual facts | Hobbies, travel, achievements |

---

## Importance Scale

| Level | Meaning | Examples |
|-------|---------|----------|
| 5 | Core identity | Where they live, their profession |
| 4 | Important | Family members, major life events |
| 3 | Notable | Preferences, ongoing projects |
| 2 | Interesting | Minor details worth noting |
| 1 | Minor | Small facts, casual mentions |

---

## Error Handling

All async methods may raise:

- `RuntimeError` - Store not loaded/opened
- `Exception` - API errors, file system errors

**Example:**
```python
try:
    await memory.load()
    results = await memory.retrieve("query")
except RuntimeError as e:
    print(f"Memory not ready: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

Back to [Getting Started](./getting-started.md)
