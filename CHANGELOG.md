# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-29

### Added
- `ExtractionResult` dataclass for full extraction metadata
- `on_complete` callback parameter to `extract_memories()` for observability
- Identity context parameters: `user_id`, `agent_id`, `session_id`, `trigger_message_count`
- Support for both sync and async callbacks

### Changed
- Environment variable prefix updated to `BRAINFART_` (was `PIPECAT_MEMORY_`)

## [0.1.0] - 2024-12-29

### Added
- Initial release as `brainfart`
- `MemoryProcessor` - Pipecat FrameProcessor for conversational memory
- `LocalMemory` - Core memory class with FAISS + SQLite
- `EmbeddingService` - Lazy-loaded sentence-transformers
- `MemoryCrypto` - Optional at-rest encryption with Fernet
- Memory extraction using Gemini tool calling
- Multi-agent isolation (separate stores per agent)
- Zero-config with sensible defaults
- Environment variable configuration (`BRAINFART_*`)
- Pydantic settings support
