# Encryption Guide

Protect your users' memories with at-rest encryption. When enabled, all data is encrypted on disk and only decrypted in memory during use.

## Why Encrypt?

Memory data can contain sensitive personal information:
- Home addresses
- Family member names
- Work details
- Personal preferences
- Emotional moments

If someone gains access to your storage, encrypted data is unreadable without the key.

## Quick Setup

### Option 1: Environment Variable

```bash
export BRAINFART_ENCRYPTION_KEY="your-secret-passphrase"
```

### Option 2: Constructor Parameter

```python
memory = MemoryProcessor(
    user_id="user-123",
    encryption_key="your-secret-passphrase",
)
```

That's it! All data is now encrypted automatically.

## How It Works

```
User says: "I live at 123 Main Street"
                ↓
        [Extract memory]
                ↓
        [Encrypt with Fernet]
                ↓
        [Store encrypted bytes to disk]
                ↓
        [On retrieval: decrypt in memory]
                ↓
Bot uses: "I live at 123 Main Street"
```

### What Gets Encrypted

| Component | Encrypted? | Details |
|-----------|------------|---------|
| SQLite content | Yes | Memory text is encrypted before storage |
| SQLite metadata | No | Category, importance, timestamps are readable |
| FAISS index | Yes | Entire index file is encrypted |
| Embeddings in memory | No | Only encrypted on disk |

### Crash Safety

The encryption is "crash-safe":
- Data is **always encrypted on disk**
- Decryption happens **only in memory**
- If the process crashes, disk data remains encrypted
- No plaintext ever touches the filesystem

## Choosing a Strong Key

Your encryption key should be:
- At least 16 characters
- Random or a strong passphrase
- Stored securely (environment variable, secrets manager)

**Good examples:**
```bash
# Random string
export BRAINFART_ENCRYPTION_KEY="aK9#mP2$xL5@nQ8&"

# Passphrase
export BRAINFART_ENCRYPTION_KEY="correct-horse-battery-staple-memory"
```

**Bad examples:**
```bash
# Too short
export BRAINFART_ENCRYPTION_KEY="secret"

# Predictable
export BRAINFART_ENCRYPTION_KEY="password123"
```

## Key Management

### Development

For local development, a simple approach works:

```bash
# .env file (don't commit to git!)
BRAINFART_ENCRYPTION_KEY=dev-secret-key-12345
```

Add `.env` to your `.gitignore`:
```
# .gitignore
.env
```

### Production

Use a secrets manager:

**AWS Secrets Manager:**
```python
import boto3

def get_encryption_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='brainfart-key')
    return response['SecretString']

memory = MemoryProcessor(
    user_id=user_id,
    encryption_key=get_encryption_key(),
)
```

**Docker:**
```yaml
# docker-compose.yml
services:
  bot:
    environment:
      - BRAINFART_ENCRYPTION_KEY_FILE=/run/secrets/memory_key
    secrets:
      - memory_key

secrets:
  memory_key:
    file: ./secrets/memory_key.txt
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: brainfart
type: Opaque
data:
  encryption-key: <base64-encoded-key>
```

## Migrating Existing Data

### Unencrypted → Encrypted

If you have existing unencrypted memories and want to enable encryption:

```python
import asyncio
from brainfart import LocalMemory, MemorySettings

async def migrate_to_encrypted():
    # Load with old settings (no encryption)
    old_settings = MemorySettings(data_dir="/path/to/data")
    old_memory = LocalMemory(old_settings, user_id="user-123")
    await old_memory.load()

    # Get all memories (they're in plaintext)
    all_memories = []
    # ... export logic here ...

    await old_memory.close()

    # Create new encrypted store
    new_settings = MemorySettings(
        data_dir="/path/to/encrypted-data",
        encryption_key="your-secret",
    )
    new_memory = LocalMemory(new_settings, user_id="user-123")
    await new_memory.load()

    # Re-import (will be encrypted)
    for mem in all_memories:
        await new_memory.store(
            content=mem.content,
            category=mem.category,
            importance=mem.importance,
        )

    await new_memory.close()
    print("Migration complete!")

asyncio.run(migrate_to_encrypted())
```

### Changing Keys

To rotate your encryption key:

1. Export all data with the old key
2. Create a new store with the new key
3. Re-import all data
4. Delete the old store

```python
# Similar to migration above, but both have encryption keys
```

## Verifying Encryption

Check that your data is actually encrypted:

```python
# Check stats
stats = memory.get_stats()
print(f"Encryption enabled: {stats['encryption_enabled']}")

# Look at raw file (should be unreadable)
with open(memory._settings.data_dir / "agent" / "user.db", "rb") as f:
    raw = f.read(100)
    print(f"Raw bytes (should look random): {raw}")
```

## Technical Details

### Algorithm

- **Cipher:** Fernet (AES-128-CBC + HMAC-SHA256)
- **Key derivation:** SHA-256 of your passphrase
- **Library:** Python's `cryptography` package

### Performance Impact

Encryption adds minimal overhead:
- ~0.1ms per memory store
- ~0.05ms per memory retrieve
- No impact on embedding or search speed

### Limitations

- **Memory is in plaintext during use** - encryption only protects data at rest
- **Key loss = data loss** - there's no recovery without the key
- **Metadata is visible** - categories, timestamps, importance scores are not encrypted

## Troubleshooting

### "Decryption failed, returning raw value"

This warning appears when:
- You enabled encryption on existing unencrypted data
- The encryption key changed
- The data file is corrupted

Solution: Migrate your data (see above) or start fresh.

### Performance issues with encryption

Encryption overhead is minimal. If you're seeing slowdowns:
- Check disk I/O (encryption increases write size slightly)
- Ensure you're not storing extremely large memories
- Consider batching writes

---

Next: [Multi-Agent Setup](./multi-agent.md) | [API Reference](./api-reference.md)
