"""
Memory extraction using Gemini tool calling.

Extracts memorable facts from conversation windows. Uses structured output
via function calling for reliable parsing.

Expected behavior:
- 70-80% of calls return empty list (nothing memorable)
- Tool only called when genuine facts are present
- Empty result = nothing memorable = common case

Note: Uses the google-genai SDK (not the deprecated google-generativeai).
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Union

from loguru import logger

# Import the new Google GenAI SDK
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai not installed - memory extraction disabled")


@dataclass
class ExtractionResult:
    """
    Complete result from a memory extraction call.

    Includes all metadata needed for logging, debugging, and analytics.
    """

    # Extracted memories (empty list if nothing memorable)
    memories: List[dict] = field(default_factory=list)

    # Status: "success", "no_memories", "error"
    status: str = "no_memories"

    # Timing
    duration_ms: int = 0

    # Model info
    model: str = ""

    # Input context
    messages: List[dict] = field(default_factory=list)
    formatted_prompt: str = ""

    # Response details
    tool_called: bool = False
    raw_response_text: Optional[str] = None
    finish_reason: Optional[str] = None

    # Error info (if status == "error")
    error_message: Optional[str] = None

    # Optional identity context (for logging)
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    trigger_message_count: Optional[int] = None


# Type alias for extraction callbacks
ExtractionCallback = Callable[[ExtractionResult], Union[None, Awaitable[None]]]


EXTRACTION_SYSTEM_PROMPT = """You analyze conversations to extract memorable facts about the user.

Only store facts that are:
- Explicitly stated or strongly implied by the USER (not the assistant)
- Worth remembering for future conversations
- Not just conversational filler ("yeah", "okay", "tell me more")
- NEW information not already obvious from context

DO NOT extract:
- The user's name (already known to the system)
- Temporary states like "user is tired" or "user is busy today"
- Things the assistant said or suggested
- Vague statements with no specific facts

Most conversation windows have NOTHING worth storing. That's normal — just respond without calling the tool.

Categories:
- identity: Location, job, family members, age, background (NOT name)
- preference: Likes, dislikes, communication style, explicit requests
- context: Current projects, problems, life events in progress
- relationship: Shared moments, emotional references, inside jokes
- surprise: Unusual or unexpected facts that stand out

Importance scale (1-5):
- 5: Core identity (where they live, what they do, family)
- 4: Important relationships or major life events
- 3: Notable preferences or ongoing situations
- 2: Interesting but not critical details
- 1: Minor details worth noting
"""


def _get_store_memories_tool_declaration() -> dict:
    """Build the tool declaration dictionary for memory extraction.

    Uses simple dictionary format compatible with google-genai SDK.
    """
    return {
        "name": "store_memories",
        "description": (
            "Store memorable facts about the user. "
            "Only call this if there are facts worth remembering. "
            "Most conversations have nothing memorable — that's fine, "
            "just don't call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "description": "List of memorable facts to store",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "The fact in third person, e.g. "
                                    "'User's brother Mike works at Google'"
                                ),
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "identity",
                                    "preference",
                                    "context",
                                    "relationship",
                                    "surprise",
                                ],
                                "description": (
                                    "identity=core facts, preference=likes/dislikes, "
                                    "context=current projects/problems, "
                                    "relationship=emotional moments, "
                                    "surprise=unusual/noteworthy"
                                ),
                            },
                            "importance": {
                                "type": "integer",
                                "description": "1-5 scale: 5=core identity, 1=minor detail",
                            },
                        },
                        "required": ["content", "category", "importance"],
                    },
                },
            },
            "required": ["memories"],
        },
    }


async def extract_memories(
    messages: List[dict],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    # Identity context for logging
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trigger_message_count: Optional[int] = None,
    # Callback for observability
    on_complete: Optional[ExtractionCallback] = None,
) -> List[dict]:
    """
    Extract memories from a conversation window using Gemini tool calling.

    Uses the google-genai SDK with explicit client creation to avoid
    credential conflicts with other Google services.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_name: Gemini model to use (default: gemini-2.0-flash-lite)
        api_key: Gemini API key (default: from GOOGLE_API_KEY env)
        user_id: Optional user ID for logging/callbacks
        agent_id: Optional agent ID for logging/callbacks
        session_id: Optional session ID for logging/callbacks
        trigger_message_count: Optional message count when extraction was triggered
        on_complete: Optional callback called with ExtractionResult after extraction

    Returns:
        List of memory dicts with keys: content, category, importance
        Empty list if nothing memorable (common case).
    """
    if not GENAI_AVAILABLE:
        logger.warning("google-genai not available for memory extraction")
        return []

    start_time = time.perf_counter()

    if model_name is None:
        model_name = os.getenv("BRAINFART_GEMINI_MODEL", "gemini-2.0-flash-lite")

    # Get API key - explicit priority order
    key = api_key or os.getenv("BRAINFART_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.warning("No Gemini API key found for memory extraction")
        return []

    # Format conversation for the prompt
    conversation = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    # Build the tool configuration
    tool_declaration = _get_store_memories_tool_declaration()
    tools = genai_types.Tool(function_declarations=[tool_declaration])

    # Build generation config with system instruction and tools
    config = genai_types.GenerateContentConfig(
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        tools=[tools],
        temperature=0.3,
    )

    # Track extraction metadata for callback
    memories_result = []
    status = "no_memories"
    tool_called = False
    raw_response_text = None
    finish_reason = None
    error_message = None

    try:
        # Create client with explicit API key - this isolates credentials
        # from any service account credentials that may be in the environment
        client = genai.Client(api_key=key)

        # Make the async generation request
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=f"Analyze this conversation for memorable facts:\n\n{conversation}",
            config=config,
        )

        # Check if model called the tool
        if not response.candidates:
            status = "no_memories"
        else:
            candidate = response.candidates[0]

            # Capture finish reason
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)

            if not candidate.content or not candidate.content.parts:
                status = "no_memories"
            else:
                # Look for function call in response
                for part in candidate.content.parts:
                    # Capture any text response
                    if hasattr(part, "text") and part.text:
                        raw_response_text = part.text

                    # Check for function call
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if fc.name == "store_memories":
                            tool_called = True
                            # Extract structured memories from function call args
                            args = fc.args if hasattr(fc, "args") else {}
                            memories = args.get("memories", [])

                            # Convert to plain dicts
                            for m in memories:
                                if isinstance(m, dict):
                                    memories_result.append({
                                        "content": str(m.get("content", "")),
                                        "category": str(m.get("category", "context")),
                                        "importance": int(m.get("importance", 3)),
                                    })
                                elif hasattr(m, "items"):
                                    memories_result.append(dict(m))

                if memories_result:
                    status = "success"
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"Extracted {len(memories_result)} memories ({elapsed_ms:.0f}ms)")

    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")
        status = "error"
        error_message = str(e)

    # Calculate duration
    duration_ms = int((time.perf_counter() - start_time) * 1000)

    # Call callback if provided
    if on_complete is not None:
        extraction_result = ExtractionResult(
            memories=memories_result,
            status=status,
            duration_ms=duration_ms,
            model=model_name,
            messages=messages,
            formatted_prompt=conversation,
            tool_called=tool_called,
            raw_response_text=raw_response_text,
            finish_reason=finish_reason,
            error_message=error_message,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            trigger_message_count=trigger_message_count,
        )

        # Handle both sync and async callbacks
        try:
            result = on_complete(extraction_result)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Extraction callback failed: {e}")

    return memories_result


async def extract_and_store(
    messages: List[dict],
    memory: "LocalMemory",
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> int:
    """
    Extract memories and store them.

    Convenience function that combines extraction and storage.

    Args:
        messages: Conversation window to analyze
        memory: LocalMemory instance
        session_id: Optional session ID for tracking
        turn_number: Optional turn number for tracking

    Returns:
        Number of memories stored (0 if nothing memorable)
    """
    from .memory import LocalMemory

    memories = await extract_memories(messages)

    if not memories:
        return 0

    await memory.store_batch(
        memories,
        session_id=session_id,
        turn_number=turn_number,
    )

    return len(memories)
