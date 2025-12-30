"""
Example Pipecat bot with brainfart memory.

This demonstrates how to use brainfart in a voice bot.
"""

import asyncio
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.services.daily import DailyParams, DailyTransport

# Import the memory processor
from brainfart import MemoryProcessor


async def main():
    """Run a voice bot with memory."""

    # Get configuration from environment
    daily_api_key = os.getenv("DAILY_API_KEY")
    daily_room_url = os.getenv("DAILY_ROOM_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([daily_api_key, daily_room_url, openai_api_key]):
        print("Missing required environment variables:")
        print("  DAILY_API_KEY - Daily.co API key")
        print("  DAILY_ROOM_URL - Daily.co room URL")
        print("  OPENAI_API_KEY - OpenAI API key")
        print("  GOOGLE_API_KEY - Google API key (for memory extraction)")
        return

    # Create transport
    transport = DailyTransport(
        daily_room_url,
        None,  # No token needed for public rooms
        "Memory Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=True,
        ),
    )

    # Create memory processor - zero config!
    # Uses GOOGLE_API_KEY from environment for Gemini extraction
    memory = MemoryProcessor(
        user_id="demo-user",  # In production, get from auth
        agent_id="demo-bot",
    )

    # Import LLM and TTS services
    from pipecat.services.openai import OpenAILLMService, OpenAITTSService

    llm = OpenAILLMService(
        api_key=openai_api_key,
        model="gpt-4",
    )

    tts = OpenAITTSService(
        api_key=openai_api_key,
        voice="alloy",
    )

    # Build pipeline with memory
    pipeline = Pipeline([
        transport.input(),
        memory,  # Memory processor goes between STT and LLM
        llm,
        tts,
        transport.output(),
    ])

    # Run the pipeline
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant):
        print(f"Participant joined: {participant['id']}")

        # Get any stored memories about this user
        stats = memory.get_stats()
        if stats.get("total_memories", 0) > 0:
            print(f"Found {stats['total_memories']} memories for user")

    @transport.event_handler("on_participant_left")
    async def on_left(transport, participant, reason):
        print(f"Participant left: {participant['id']}")

        # Cleanup and save memories
        await memory.cleanup()

    print("Starting bot with memory...")
    print(f"Memory storage: {memory._settings.data_dir}")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
