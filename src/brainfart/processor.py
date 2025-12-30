"""
MemoryProcessor - Pipecat FrameProcessor for conversational memory.

Integrates with Pipecat pipelines to:
1. Store user utterances and extract memorable facts
2. Retrieve relevant memories to enrich LLM context
"""

from typing import Dict, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .config import MemorySettings, get_settings
from .crypto import MemoryCrypto
from .extraction import extract_memories
from .memory import LocalMemory, MemoryResult


class MemoryProcessor(FrameProcessor):
    """
    Pipecat processor that adds conversational memory.

    Features:
    - Stores conversation turns and extracts memorable facts
    - Retrieves relevant memories to enrich LLM context
    - Supports encryption for at-rest security
    - Zero-config with sensible defaults

    Usage:
        # Zero-config (uses environment variables)
        processor = MemoryProcessor(user_id="user123")

        # Or with explicit settings
        processor = MemoryProcessor(
            user_id="user123",
            gemini_api_key="your-key",
            top_k=10,
        )

        # Add to pipeline
        pipeline = Pipeline([
            transport.input(),
            stt_service,
            memory_processor,  # <- Add here
            llm_service,
            tts_service,
            transport.output(),
        ])
    """

    def __init__(
        self,
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
    ):
        """
        Initialize the memory processor.

        Args:
            user_id: User identifier (required)
            agent_id: Agent identifier for multi-agent isolation
            gemini_api_key: Gemini API key for extraction
            embedding_model: Sentence-transformers model name
            top_k: Number of memories to retrieve
            encryption_key: Key for at-rest encryption
            settings: Pre-configured MemorySettings
            inject_memories: Whether to inject memories into LLM context
            extract_memories: Whether to extract memories from conversation
            extraction_interval: Extract every N user messages
        """
        super().__init__(**kwargs)

        self.user_id = user_id
        self.agent_id = agent_id
        self.inject_memories = inject_memories
        self.extract_memories_enabled = extract_memories
        self.extraction_interval = extraction_interval

        # Merge explicit args with settings
        self._settings = settings or get_settings()
        if gemini_api_key:
            self._settings.gemini_api_key = gemini_api_key
        if embedding_model:
            self._settings.embedding_model = embedding_model
        if top_k:
            self._settings.top_k = top_k
        if encryption_key:
            self._settings.encryption_key = encryption_key

        # Initialize encryption
        if self._settings.encryption_key:
            MemoryCrypto.initialize(self._settings.encryption_key)

        # Lazy-initialize memory system
        self._memory: Optional[LocalMemory] = None

        # Conversation buffer for extraction
        self._conversation_buffer: List[Dict[str, str]] = []
        self._message_count = 0

        if not self._settings.gemini_api_key and extract_memories:
            logger.warning(
                "No Gemini API key found. Set PIPECAT_MEMORY_GEMINI_API_KEY or "
                "GOOGLE_API_KEY environment variable, or pass gemini_api_key parameter. "
                "Memory extraction will be disabled."
            )
            self.extract_memories_enabled = False

    @property
    def memory(self) -> LocalMemory:
        """Lazy-load memory system on first access."""
        if self._memory is None:
            self._memory = LocalMemory(
                settings=self._settings,
                user_id=self.user_id,
                agent_id=self.agent_id,
            )
            logger.info(
                f"Initialized local memory for {self.agent_id}/{self.user_id} "
                f"at {self._settings.data_dir}"
            )
        return self._memory

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, storing and retrieving memories."""
        await super().process_frame(frame, direction)

        # Handle transcription (user speech)
        if isinstance(frame, TranscriptionFrame):
            await self._handle_user_message(frame.text)

        # Handle LLM messages - inject memories
        elif isinstance(frame, LLMMessagesFrame) and self.inject_memories:
            frame = await self._inject_memories(frame)

        await self.push_frame(frame, direction)

    async def _handle_user_message(self, text: str) -> None:
        """Process a user message for memory extraction."""
        if not text.strip():
            return

        # Ensure memory is loaded
        await self.memory.load()

        # Add to conversation buffer
        self._conversation_buffer.append({"role": "user", "content": text})
        self._message_count += 1

        # Trigger extraction periodically
        if (
            self.extract_memories_enabled
            and self._message_count % self.extraction_interval == 0
        ):
            await self._extract_from_buffer()

    async def _extract_from_buffer(self) -> None:
        """Extract memories from conversation buffer."""
        if not self._conversation_buffer:
            return

        # Take recent window
        window = self._conversation_buffer[-self._settings.extraction_window_size :]

        memories = await extract_memories(
            messages=window,
            model_name=self._settings.gemini_model,
            api_key=self._settings.gemini_api_key,
        )

        if memories:
            await self.memory.store_batch(memories)
            logger.debug(f"Extracted {len(memories)} memories")

    async def _inject_memories(self, frame: LLMMessagesFrame) -> LLMMessagesFrame:
        """Inject relevant memories into LLM context."""
        await self.memory.load()

        # Find the last user message
        last_user_msg = None
        for msg in reversed(frame.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if not last_user_msg:
            return frame

        # Retrieve relevant memories
        results = await self.memory.retrieve(last_user_msg)

        if not results:
            return frame

        # Format memories
        memory_text = self._format_memories(results)

        # Inject as system message
        memory_message = {
            "role": "system",
            "content": f"Relevant memories about the user:\n{memory_text}",
        }

        # Insert after system prompt, before user messages
        messages = list(frame.messages)
        insert_idx = 1  # After first system message

        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                insert_idx = i
                break

        messages.insert(insert_idx, memory_message)

        logger.debug(f"Injected {len(results)} memories into context")

        return LLMMessagesFrame(messages=messages)

    def _format_memories(self, memories: List[MemoryResult]) -> str:
        """Format memories for injection into prompt."""
        lines = []
        for mem in memories:
            category_hint = f"[{mem.category}]" if mem.category else ""
            lines.append(f"- {category_hint} {mem.content}")
        return "\n".join(lines)

    def add_assistant_message(self, text: str) -> None:
        """
        Add an assistant message to the conversation buffer.

        Call this from your pipeline after the LLM response to include
        assistant turns in the extraction context.
        """
        if text.strip():
            self._conversation_buffer.append({"role": "assistant", "content": text})

    async def get_memories(
        self,
        query: str,
        k: int = None,
        categories: Optional[List[str]] = None,
    ) -> List[MemoryResult]:
        """
        Retrieve memories (public API for external use).

        Args:
            query: Search query
            k: Number of results
            categories: Filter by categories

        Returns:
            List of MemoryResult
        """
        await self.memory.load()
        return await self.memory.retrieve(query, k=k, categories=categories)

    async def store_memory(
        self,
        content: str,
        category: str = "context",
        importance: int = 3,
    ) -> int:
        """
        Store a memory manually (public API for external use).

        Args:
            content: Memory text
            category: Category
            importance: 1-5 scale

        Returns:
            Memory ID
        """
        await self.memory.load()
        return await self.memory.store(content, category=category, importance=importance)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        if self._memory is None:
            return {"loaded": False}
        return self._memory.get_stats()

    async def cleanup(self) -> None:
        """Save and cleanup on shutdown."""
        # Final extraction
        if self.extract_memories_enabled and self._conversation_buffer:
            await self._extract_from_buffer()

        # Save and close
        if self._memory:
            await self._memory.close()

        logger.info(f"Memory processor cleaned up for {self.agent_id}/{self.user_id}")
