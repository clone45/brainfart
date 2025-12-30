# Getting Started with brainfart

Give your voice bot a memory! This guide will have you up and running in under 5 minutes.

## What is this?

When you chat with most voice bots, they forget everything the moment the call ends. brainfart changes that. It remembers things like:

- "I live in San Francisco"
- "I prefer concise answers"
- "My dog's name is Max"
- "I'm working on a Python project"

The next time you talk to the bot, it already knows these things about you.

## Installation

```bash
pip install brainfart
```

That's it! Everything you need is included—no separate database to set up, no external services to configure.

> **Note:** The first time you use it, it downloads a small AI model (~90MB) for understanding text. This happens automatically and is cached for future use.

## Quick Start

### Step 1: Get a Gemini API Key

The memory system uses Google's Gemini to figure out what's worth remembering from conversations. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step 2: Set Your API Key

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Step 3: Add Memory to Your Bot

```python
from brainfart import MemoryProcessor

# Create the memory processor
memory = MemoryProcessor(user_id="user-123")

# Add it to your pipeline (between speech-to-text and your LLM)
pipeline = Pipeline([
    transport.input(),
    stt,              # Speech-to-text
    memory,           # <-- Memory goes here!
    llm,              # Your language model
    tts,              # Text-to-speech
    transport.output(),
])
```

That's the basic setup! The bot will now:
1. Listen to what users say
2. Extract memorable facts automatically
3. Recall relevant memories in future conversations

## How It Works

### The Memory Cycle

```
User: "By the way, I just moved to Austin last month"
         ↓
    [Memory extracts: "User recently moved to Austin"]
         ↓
    [Stored for later]

--- Later that day or next week ---

User: "What's the weather like here?"
         ↓
    [Memory recalls: "User recently moved to Austin"]
         ↓
Bot: "Let me check the weather in Austin for you..."
```

### What Gets Remembered?

The system is smart about what it saves. It looks for:

| Type | Examples |
|------|----------|
| **Identity** | Where you live, your job, family members |
| **Preferences** | "I prefer email over phone", "I like detailed explanations" |
| **Context** | Current projects, ongoing situations |
| **Relationships** | Shared jokes, emotional moments |
| **Surprises** | Unusual facts that stand out |

It ignores:
- Filler words ("yeah", "okay", "um")
- Temporary states ("I'm tired today")
- Things the bot said (only remembers user facts)

## Your First Memory Bot

Here's a complete example you can run:

```python
import asyncio
import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from brainfart import MemoryProcessor

async def main():
    # Your services (STT, LLM, TTS, transport)
    # ... set these up based on your providers ...

    # Create memory for this user
    memory = MemoryProcessor(
        user_id="demo-user",
        agent_id="my-bot",
    )

    # Build the pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        memory,
        llm,
        tts,
        transport.output(),
    ])

    # Run it
    runner = PipelineRunner()
    await runner.run(PipelineTask(pipeline))

if __name__ == "__main__":
    asyncio.run(main())
```

## Checking What's Stored

Want to see what the bot remembers?

```python
# Get statistics
stats = memory.get_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"By category: {stats['by_category']}")

# Search for specific memories
results = await memory.get_memories("work")
for r in results:
    print(f"- {r.content} (similarity: {r.similarity:.0%})")
```

## Where Data is Stored

By default, memories are saved in:
- **macOS/Linux:** `~/.cache/brainfart/`
- **Windows:** `C:\Users\YourName\.cache\brainfart\`

Each user gets their own files:
```
~/.cache/brainfart/
└── my-bot/
    ├── user-123.index    # Vector search index
    └── user-123.db       # Memory content
```

## Next Steps

- [Configuration Guide](./configuration.md) - Customize settings
- [Encryption Guide](./encryption.md) - Secure your data
- [Multi-Agent Setup](./multi-agent.md) - Multiple bots, separate memories
- [API Reference](./api-reference.md) - Full documentation

## Troubleshooting

### "No Gemini API key found"

Make sure you've set the environment variable:
```bash
export GOOGLE_API_KEY="your-key"
```

Or pass it directly:
```python
memory = MemoryProcessor(user_id="user-123", gemini_api_key="your-key")
```

### First startup is slow

That's normal! The first time, it downloads the embedding model (~90MB). After that, it's cached and starts in under a second.

### Memories aren't being extracted

Check that:
1. Your Gemini API key is valid
2. The user is saying memorable things (not just "yes", "okay", etc.)
3. You've had at least 5 messages (extraction runs periodically)

---

Questions? Issues? [Open an issue on GitHub](https://github.com/r1n-ai/brainfart/issues)
