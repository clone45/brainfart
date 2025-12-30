"""
Memory extraction using Gemini tool calling.

Extracts memorable facts from conversation windows. Uses structured output
via function calling for reliable parsing.

Expected behavior:
- 70-80% of calls return empty list (nothing memorable)
- Tool only called when genuine facts are present
- Empty result = nothing memorable = common case
"""

import os
import time
from typing import List, Optional

from loguru import logger


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


def _get_store_memories_tool():
    """Build the tool definition for memory extraction."""
    import google.generativeai as genai

    return genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="store_memories",
                description=(
                    "Store memorable facts about the user. "
                    "Only call this if there are facts worth remembering. "
                    "Most conversations have nothing memorable — that's fine, "
                    "just don't call this tool."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "memories": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(
                                type=genai.protos.Type.OBJECT,
                                properties={
                                    "content": genai.protos.Schema(
                                        type=genai.protos.Type.STRING,
                                        description=(
                                            "The fact in third person, e.g. "
                                            "'User's brother Mike works at Google'"
                                        ),
                                    ),
                                    "category": genai.protos.Schema(
                                        type=genai.protos.Type.STRING,
                                        enum=[
                                            "identity",
                                            "preference",
                                            "context",
                                            "relationship",
                                            "surprise",
                                        ],
                                        description=(
                                            "identity=core facts, preference=likes/dislikes, "
                                            "context=current projects/problems, "
                                            "relationship=emotional moments, "
                                            "surprise=unusual/noteworthy"
                                        ),
                                    ),
                                    "importance": genai.protos.Schema(
                                        type=genai.protos.Type.INTEGER,
                                        description="1-5 scale: 5=core identity, 1=minor detail",
                                    ),
                                },
                                required=["content", "category", "importance"],
                            ),
                        ),
                    },
                    required=["memories"],
                ),
            ),
        ],
    )


async def extract_memories(
    messages: List[dict],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[dict]:
    """
    Extract memories from a conversation window using Gemini tool calling.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_name: Gemini model to use (default: gemini-2.0-flash-lite)
        api_key: Gemini API key (default: from GOOGLE_API_KEY env)

    Returns:
        List of memory dicts with keys: content, category, importance
        Empty list if nothing memorable (common case).
    """
    import google.generativeai as genai

    start_time = time.perf_counter()

    if model_name is None:
        model_name = os.getenv("PIPECAT_MEMORY_GEMINI_MODEL", "gemini-2.0-flash-lite")

    # Configure API key
    key = api_key or os.getenv("PIPECAT_MEMORY_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        genai.configure(api_key=key)
    else:
        logger.warning("No Gemini API key found for memory extraction")
        return []

    # Format conversation
    conversation = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    # Create model with tool
    model = genai.GenerativeModel(
        model_name,
        tools=[_get_store_memories_tool()],
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
    )

    result = []

    try:
        response = await model.generate_content_async(
            f"Analyze this conversation for memorable facts:\n\n{conversation}",
            generation_config=genai.GenerationConfig(temperature=0.3),
        )

        # Check if model called the tool
        if not response.candidates:
            return []

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return []

        # Look for function call in response
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call.name == "store_memories":
                # Extract structured memories from function call args
                args = dict(part.function_call.args)
                memories = args.get("memories", [])

                # Convert proto objects to plain dicts if needed
                for m in memories:
                    if hasattr(m, "items"):
                        result.append(dict(m))
                    else:
                        result.append(
                            {
                                "content": str(m.get("content", "")),
                                "category": str(m.get("category", "context")),
                                "importance": int(m.get("importance", 3)),
                            }
                        )

        if result:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Extracted {len(result)} memories ({elapsed_ms:.0f}ms)")

    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")

    return result


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
