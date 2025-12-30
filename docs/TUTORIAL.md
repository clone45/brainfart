# Building Voice Bots That Remember: A Practical Guide

This guide walks you through adding persistent memory to your voice bot. It's based on a real production implementation, and I'll share what I learned along the way so you can avoid the same headaches.

## The Problem I Was Trying to Solve

I was building a voice AI assistant with Pipecat. The conversations were working great, but there was something fundamentally broken about the experience: every time a user came back, the bot had completely forgotten who they were. They'd say "remember when we talked about my project?" and the bot would have no idea what they meant.

I needed the bot to remember things like where the user lives, what they're working on, and those little details that make a conversation feel continuous rather than a series of disconnected chats.

At first, I thought this would be straightforward. I'd just save the conversation history and load it on the next session. But conversation history grows fast, and it quickly becomes too expensive to stuff everything into the LLM context. What I actually needed was something smarter: a way to extract the important facts from conversations and store them efficiently for later retrieval.

## What the Memory System Actually Does

The core idea is deceptively simple. As the user talks, the system periodically analyzes the conversation and extracts memorable facts. These aren't just keywords—they're contextual observations like "User's brother Mike works at Google" or "User prefers direct answers without too much pleasantry."

These facts get stored in a vector database, which means you can later search for them semantically. When the user says "tell me more about what we discussed regarding my family," the system can find the relevant memories even though the word "brother" wasn't in the query.

The architecture looks like this: conversation flows in, Gemini analyzes a window of recent messages, extracted memories go into FAISS (a fast vector index), and metadata goes into SQLite for persistence. When it's time to respond, relevant memories are retrieved and injected into the LLM context.

## Starting Simple

Let's start with the minimum viable implementation. If you're using Pipecat, this is genuinely all you need:

```python
from pipecat.pipeline.pipeline import Pipeline
from brainfart import MemoryProcessor

memory = MemoryProcessor(user_id="user123")

pipeline = Pipeline([
    transport.input(),
    stt,
    memory,           # This is the only new line
    llm,
    tts,
    transport.output(),
])
```

Set your Gemini API key in the environment:

```bash
export GOOGLE_API_KEY="your-key-here"
```

That's the entire integration. The processor handles buffering messages, calling Gemini for extraction, storing vectors, and injecting relevant memories into the context before each LLM call.

But most real applications need more control than this. Let me show you how to build something production-ready.

## A More Realistic Implementation

In production, you probably have a multi-tenant application where different users talk to different agents. Maybe you have a sales bot and a support bot, and you don't want them sharing memories—it would be weird if the support bot remembered something from a sales conversation.

You also probably want observability. When memories are extracted, you want to log what happened for debugging. And you might want to trigger extraction on your own schedule rather than letting the processor decide.

Here's how I structured this in my own project. First, I use the standalone extraction API rather than the processor:

```python
from brainfart import extract_memories, LocalMemory, MemorySettings, ExtractionResult

async def handle_extraction(result: ExtractionResult):
    """Called after each extraction attempt with full metadata."""
    print(f"Extraction completed in {result.duration_ms}ms")
    print(f"Status: {result.status}")
    print(f"Tool called: {result.tool_called}")
    print(f"Memories extracted: {len(result.memories)}")

    # You could log this to your observability system
    # await log_to_datadog(result)
    # await save_to_mongodb(result)

async def run_extraction(messages, user_id, agent_id, session_id):
    """Extract memories from a conversation window."""
    memories = await extract_memories(
        messages=messages,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        on_complete=handle_extraction,
    )
    return memories
```

The `on_complete` callback gives you full visibility into what happened during extraction. The `ExtractionResult` contains everything: how long it took, whether Gemini called the tool, what memories were extracted, the raw response text, and any errors. This is invaluable for debugging when something goes wrong.

## Controlling When Extraction Happens

In my application, I don't want extraction running on every single message. Network calls add latency, and most conversation turns don't contain memorable information anyway. Gemini is smart enough to return nothing when there's nothing worth remembering, but I'd rather not make the call at all in those cases.

My approach is to track message counts in the database and trigger extraction every N messages:

```python
MESSAGE_WINDOW = 10  # Extract every 10 messages

async def on_new_message(user_id, agent_id, message):
    """Called whenever a new message is saved."""
    # Save the message to your database
    await save_message(user_id, agent_id, message)

    # Check if it's time for extraction
    message_count = await get_message_count(user_id, agent_id)
    last_extraction_count = await get_last_extraction_count(user_id, agent_id)

    if message_count - last_extraction_count >= MESSAGE_WINDOW:
        # Fetch the conversation window
        messages = await get_recent_messages(user_id, agent_id, limit=MESSAGE_WINDOW)

        # Run extraction in the background
        asyncio.create_task(
            run_extraction_and_store(messages, user_id, agent_id)
        )
```

This gives you complete control over timing. You could extract after every assistant response, only when the user mentions certain keywords, or on a completely custom schedule.

## Storing the Memories

Once you've extracted memories, you need somewhere to put them. The `LocalMemory` class handles all the FAISS and SQLite complexity:

```python
from brainfart import LocalMemory, MemorySettings

async def setup_memory_store(user_id, agent_id):
    """Initialize a memory store for a user/agent pair."""
    settings = MemorySettings(
        data_dir="/var/data/memories",
        similarity_threshold=0.5,
    )

    memory = LocalMemory(settings, user_id=user_id, agent_id=agent_id)
    await memory.load()
    return memory

async def store_memories(memory_store, memories, session_id=None):
    """Store extracted memories."""
    for m in memories:
        await memory_store.store(
            content=m["content"],
            category=m["category"],
            importance=m.get("importance", 3),
            session_id=session_id,
        )
    await memory_store.save()
```

The storage is organized by agent and user, so you get natural isolation:

```
/var/data/memories/
├── sales-bot/
│   ├── user-123.index    # FAISS vectors
│   └── user-123.db       # SQLite metadata
└── support-bot/
    ├── user-123.index
    └── user-123.db
```

## Retrieving Memories for Context

Before the LLM responds, you want to inject relevant memories. The retrieval is semantic, so it finds memories based on meaning rather than keyword matching:

```python
async def get_relevant_memories(memory_store, user_message):
    """Find memories relevant to what the user just said."""
    results = await memory_store.retrieve(
        query=user_message,
        k=5,  # Top 5 most relevant
    )

    if not results:
        return None

    # Format for injection into LLM context
    memory_text = "Relevant things you know about this user:\n"
    for r in results:
        memory_text += f"- {r.content}\n"

    return memory_text
```

You can inject this as a system message or prepend it to the user's message. I typically add it as a separate message right before the user's turn:

```python
messages = [
    {"role": "system", "content": system_prompt},
    # ... conversation history ...
    {"role": "system", "content": memory_context},  # Injected memories
    {"role": "user", "content": user_message},
]
```

## Adding Encryption

If you're storing personal information about users, you should encrypt it at rest. This is straightforward:

```python
settings = MemorySettings(
    data_dir="/var/data/memories",
    encryption_key="your-secret-key",
)
```

With encryption enabled, both the FAISS index and the SQLite content are encrypted on disk. The data is only decrypted in memory when actively being used. If your process crashes, the data remains encrypted.

In production, you'd want to pull this key from a secrets manager rather than hardcoding it:

```python
import os

encryption_key = os.getenv("MEMORY_ENCRYPTION_KEY")
if not encryption_key:
    raise ValueError("MEMORY_ENCRYPTION_KEY must be set in production")
```

## Handling Multiple Conversations

One thing that surprised me early on was how to handle multiple simultaneous conversations. If a user has two browser tabs open talking to the same agent, you don't want them competing for the same memory store.

The solution is to load the memory store once per user session and reuse it:

```python
class SessionManager:
    def __init__(self):
        self.memory_stores = {}

    async def get_memory_store(self, user_id, agent_id):
        key = f"{agent_id}/{user_id}"
        if key not in self.memory_stores:
            self.memory_stores[key] = await setup_memory_store(user_id, agent_id)
        return self.memory_stores[key]

    async def close_session(self, user_id, agent_id):
        key = f"{agent_id}/{user_id}"
        if key in self.memory_stores:
            store = self.memory_stores.pop(key)
            await store.close()
```

The memory store handles file locking internally, so concurrent writes from different processes won't corrupt the data.

## Debugging Extraction Issues

Sometimes the LLM extracts something unexpected, or fails to extract something obvious. The `ExtractionResult` gives you everything you need to debug:

```python
async def debug_extraction(result: ExtractionResult):
    if result.status == "error":
        print(f"Extraction failed: {result.error_message}")
        return

    if not result.tool_called:
        print("Gemini didn't call the tool—nothing memorable in this window")
        print(f"Raw response: {result.raw_response_text}")
        return

    print(f"Extracted {len(result.memories)} memories:")
    for m in result.memories:
        print(f"  [{m['category']}] {m['content']} (importance: {m['importance']})")

    print(f"\nPrompt sent to Gemini:\n{result.formatted_prompt}")
```

The `formatted_prompt` field shows you exactly what was sent to Gemini, which is invaluable when the extraction isn't behaving as expected. You can copy this into the Gemini console to experiment with prompt changes.

## What I Learned

After running this in production for a while, here are the things I wish I'd known from the start.

First, most conversation turns don't contain memorable information, and that's fine. Gemini is trained to only call the extraction tool when there's something worth remembering. Don't worry if 70-80% of your extractions return empty lists.

Second, the extraction window size matters. Too small and you miss context. Too large and you're paying for tokens you don't need. I settled on 10-12 messages as a good balance.

Third, extraction adds latency, so do it asynchronously. Don't make the user wait for extraction to complete before you respond. Fire it off in the background and let it complete on its own time.

Fourth, log everything during development. The `on_complete` callback is your friend. Once you trust the system, you can dial back the logging, but early on you want full visibility.

Finally, start simple. The basic `MemoryProcessor` integration is genuinely useful out of the box. Add complexity only when you need it.

## Next Steps

This guide covered the core patterns, but there's more to explore. Check out the main README for the full API reference, including all the configuration options for both the processor and the standalone APIs.

If you run into issues or have questions, the GitHub issues are a good place to start. And if you build something cool with this, I'd love to hear about it.
