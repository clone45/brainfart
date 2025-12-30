# Multi-Agent Setup

Running multiple bots? Each can have its own separate memories for users, or share memories across bots. This guide covers both scenarios.

## The Problem

Imagine you have two bots:
- **Sales Bot** - helps users buy products
- **Support Bot** - helps users solve problems

Without isolation, confusing things happen:

```
User to Sales Bot: "I'm interested in the Pro plan"
    [Memory stored: "User interested in Pro plan"]

User to Support Bot: "My app keeps crashing"
    Support Bot: "I see you're interested in the Pro plan!"  # Weird!
```

## The Solution: Agent IDs

Each bot gets its own `agent_id`, creating separate memory stores:

```python
# Sales bot
sales_memory = MemoryProcessor(
    user_id="user-123",
    agent_id="sales-bot",
)

# Support bot
support_memory = MemoryProcessor(
    user_id="user-123",
    agent_id="support-bot",
)
```

Now each bot only sees its own memories:

```
~/.cache/brainfart/
├── sales-bot/
│   ├── user-123.index
│   └── user-123.db
└── support-bot/
    ├── user-123.index
    └── user-123.db
```

## Setup Examples

### Multiple Bots, Same Server

```python
from brainfart import MemoryProcessor

class BotFactory:
    def create_sales_bot(self, user_id: str):
        memory = MemoryProcessor(
            user_id=user_id,
            agent_id="sales-bot",
        )
        # ... rest of pipeline
        return pipeline

    def create_support_bot(self, user_id: str):
        memory = MemoryProcessor(
            user_id=user_id,
            agent_id="support-bot",
        )
        # ... rest of pipeline
        return pipeline
```

### Multiple Bots, Shared Storage

If bots run on different servers but share storage (e.g., network drive, cloud storage):

```python
# On Server A (Sales)
memory = MemoryProcessor(
    user_id=user_id,
    agent_id="sales-bot",
    settings=MemorySettings(data_dir="/shared/memories"),
)

# On Server B (Support)
memory = MemoryProcessor(
    user_id=user_id,
    agent_id="support-bot",
    settings=MemorySettings(data_dir="/shared/memories"),
)
```

### Different Versions of Same Bot

Track memories across bot versions:

```python
# Version 1
memory = MemoryProcessor(
    user_id=user_id,
    agent_id="my-bot-v1",
)

# Version 2 (separate memories)
memory = MemoryProcessor(
    user_id=user_id,
    agent_id="my-bot-v2",
)
```

Or share memories between versions:

```python
# Both versions use same agent_id
memory = MemoryProcessor(
    user_id=user_id,
    agent_id="my-bot",  # Same for all versions
)
```

## Sharing Memories Between Bots

Sometimes you *want* bots to share memories. Use the same `agent_id`:

```python
# All bots in the "customer-service" family share memories
SHARED_AGENT_ID = "customer-service"

sales_memory = MemoryProcessor(
    user_id=user_id,
    agent_id=SHARED_AGENT_ID,
)

support_memory = MemoryProcessor(
    user_id=user_id,
    agent_id=SHARED_AGENT_ID,
)

billing_memory = MemoryProcessor(
    user_id=user_id,
    agent_id=SHARED_AGENT_ID,
)
```

Now all three bots see the same memories about each user.

## Architecture Patterns

### Pattern 1: Complete Isolation

Each bot type has completely separate memories.

```
User "alice" talks to Sales Bot → sales-bot/alice.db
User "alice" talks to Support Bot → support-bot/alice.db
User "bob" talks to Sales Bot → sales-bot/bob.db
```

**Pros:** Clean separation, no confusion
**Cons:** No knowledge sharing, user repeats themselves

### Pattern 2: Shared Core + Bot-Specific

Share identity facts, but keep context separate:

```python
class HybridMemory:
    def __init__(self, user_id: str, bot_type: str):
        # Shared identity memories
        self.shared = MemoryProcessor(
            user_id=user_id,
            agent_id="shared",
            inject_memories=True,
            extract_memories=False,  # Don't extract here
        )

        # Bot-specific context
        self.specific = MemoryProcessor(
            user_id=user_id,
            agent_id=bot_type,
            inject_memories=True,
            extract_memories=True,
        )

    async def get_all_memories(self, query: str):
        shared = await self.shared.get_memories(query, categories=["identity"])
        specific = await self.specific.get_memories(query)
        return shared + specific
```

### Pattern 3: Read-Only Shared Store

One "master" bot extracts memories, others only read:

```python
# Master bot (extracts and stores)
master_memory = MemoryProcessor(
    user_id=user_id,
    agent_id="master",
    extract_memories=True,
    inject_memories=True,
)

# Reader bot (only retrieves)
reader_memory = MemoryProcessor(
    user_id=user_id,
    agent_id="master",  # Same agent_id to see same memories
    extract_memories=False,  # Don't extract
    inject_memories=True,  # Do inject
)
```

## Practical Tips

### Naming Conventions

Use clear, consistent agent IDs:

```python
# Good
agent_id="sales-bot-v2"
agent_id="support-agent"
agent_id="onboarding-assistant"

# Less clear
agent_id="bot1"
agent_id="new"
agent_id="test"
```

### Environment-Based IDs

Separate dev/staging/prod:

```python
import os

ENV = os.getenv("ENVIRONMENT", "dev")

memory = MemoryProcessor(
    user_id=user_id,
    agent_id=f"my-bot-{ENV}",  # my-bot-dev, my-bot-staging, my-bot-prod
)
```

### Dynamic Agent Selection

Let the user or context determine which agent:

```python
def get_memory_for_context(user_id: str, context: str) -> MemoryProcessor:
    if "billing" in context:
        agent_id = "billing-agent"
    elif "technical" in context:
        agent_id = "tech-support"
    else:
        agent_id = "general"

    return MemoryProcessor(user_id=user_id, agent_id=agent_id)
```

## Monitoring Multi-Agent Systems

Track which agents have which memories:

```python
import os
from pathlib import Path

def list_all_agents(data_dir: str = "~/.cache/brainfart"):
    data_path = Path(data_dir).expanduser()
    agents = []
    for agent_dir in data_path.iterdir():
        if agent_dir.is_dir():
            user_count = len(list(agent_dir.glob("*.db")))
            agents.append({
                "agent_id": agent_dir.name,
                "user_count": user_count,
            })
    return agents

# Output: [{"agent_id": "sales-bot", "user_count": 150}, ...]
```

## Troubleshooting

### Memories appearing in wrong bot

Check that you're using different `agent_id` values:

```python
print(f"Agent ID: {memory.agent_id}")
print(f"Data path: {memory._settings.data_dir / memory.agent_id}")
```

### Memories not sharing when they should

Ensure both bots use:
1. Same `agent_id`
2. Same `data_dir`
3. Same encryption key (if using encryption)

### File locking issues

If multiple processes access the same memory files:
- SQLite handles concurrent reads well
- For writes, consider a single writer or external locking
- Use WAL mode (enabled by default) for better concurrency

---

Next: [API Reference](./api-reference.md) | Back to [Getting Started](./getting-started.md)
