# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-02-04

### Changed
- **BREAKING**: Switched from deprecated `google-generativeai` to `google-genai` SDK
  - Fixes credential conflicts with service account credentials (GOOGLE_CREDENTIALS_JSON)
  - Uses explicit `genai.Client(api_key=...)` for proper credential isolation
  - Uses dictionary-based tool definitions instead of proto objects
  - API unchanged - existing code works without modification

### Added
- Optional `[legacy-genai]` extra for users who need the old google-generativeai SDK

## [0.3.0] - 2024-12-30

### Changed
- **BREAKING**: Switched from `sentence-transformers` to `fastembed` for embeddings
  - ~4x smaller install size (~150MB vs ~600MB with PyTorch)
  - Uses ONNX runtime instead of PyTorch
  - Same `all-MiniLM-L6-v2` model, same 384-dimension embeddings
  - API unchanged - existing code works without modification

### Added
- Model alias support: can use short names like `"all-MiniLM-L6-v2"` or full names like `"sentence-transformers/all-MiniLM-L6-v2"`
- Support for additional models: `bge-small-en-v1.5`, `bge-base-en-v1.5`, `all-mpnet-base-v2`
- Optional `[transformers]` extra for users who prefer sentence-transformers

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
