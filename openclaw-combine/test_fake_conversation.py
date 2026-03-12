#!/usr/bin/env python3
"""
Fake conversation test for OpenClaw-Combine training.

Simulates multi-turn conversations with the model to trigger training samples.
Run this while the RL server is running to inject test data.
"""
import asyncio
import json
import random
from typing import Any

import httpx

# Configuration
API_BASE = "http://0.0.0.0:30000"  # Change to your server IP if needed
API_KEY = "apiKey"  # Match SGLANG_API_KEY in your script

# Test conversations: (user_message, expected_quality)
# quality: "good" (will get +1 reward) or "bad" (will get -1 reward)
CONVERSATIONS = [
    # Math problem solving
    [
        ("Calculate 123 + 456", "good"),
        ("What is the square root of 144?", "good"),
        ("Explain what a prime number is", "good"),
    ],
    # Coding questions
    [
        ("Write a Python function to reverse a string", "good"),
        ("How do I read a file in Python?", "good"),
        ("What's the difference between list and tuple?", "good"),
    ],
    # General questions with intentional bad responses
    [
        ("What is the capital of France?", "bad"),
        ("Tell me about machine learning", "bad"),
        ("How does photosynthesis work?", "bad"),
    ],
    # Mixed quality conversation
    [
        ("What is 2+2?", "good"),
        ("Explain quantum computing in one sentence", "bad"),
        ("What is Python?", "good"),
        ("Tell me a joke", "bad"),
    ],
]


async def chat_completion(
    client: httpx.AsyncClient,
    messages: list[dict[str, str]],
    session_id: str,
) -> dict[str, Any]:
    """Send chat completion request to the model."""
    response = await client.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "qwen3-4b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": False,
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Session-ID": session_id,  # Track conversation session
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


async def send_feedback(
    client: httpx.AsyncClient,
    session_id: str,
    quality: str,
    message: str = "",
):
    """Send feedback for the last turn (simulates user reaction)."""
    # In real OpenClaw, feedback comes from the next user message or explicit rating
    # Here we simulate it by sending a follow-up message that the PRM will evaluate

    if quality == "good":
        # Positive implicit feedback: user continues normally or says thanks
        feedback_messages = [
            "Thanks, that's helpful!",
            "Got it, thanks!",
            "Perfect, exactly what I needed.",
            "👍",  # thumbs up
        ]
    else:
        # Negative implicit feedback: user repeats question or shows dissatisfaction
        feedback_messages = [
            "That's not quite right. Let me ask again.",
            "I don't think that's correct.",
            "Can you try again?",
            "👎",  # thumbs down
            f"No, I was asking about {message[:30]}...",
        ]

    feedback = random.choice(feedback_messages)

    # Send the feedback as the next user message
    # This creates the "next state" that the PRM will use to judge the previous turn
    await client.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": feedback}],
            "temperature": 0.7,
            "max_tokens": 1,  # We don't care about the response
            "stream": False,
        },
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "X-Session-ID": session_id,
        },
        timeout=10.0,
    )


async def run_conversation(client: httpx.AsyncClient, conversation: list[tuple[str, str]], conv_id: int):
    """Run a single multi-turn conversation."""
    session_id = f"test_session_{conv_id}_{random.randint(1000, 9999)}"
    messages = []

    print(f"\n{'='*60}")
    print(f"Starting conversation {conv_id} (session: {session_id})")
    print(f"{'='*60}")

    for turn_idx, (user_msg, quality) in enumerate(conversation):
        print(f"\n[Turn {turn_idx + 1}] User: {user_msg}")

        # Add user message
        messages.append({"role": "user", "content": user_msg})

        # Get model response
        try:
            result = await chat_completion(client, messages, session_id)
            assistant_msg = result["choices"][0]["message"]["content"]
            print(f"[Turn {turn_idx + 1}] Assistant: {assistant_msg[:100]}...")

            # Add assistant message to history
            messages.append({"role": "assistant", "content": assistant_msg})

            # Send feedback (simulates user reaction in next turn)
            print(f"[Turn {turn_idx + 1}] Simulated feedback: {quality}")
            await send_feedback(client, session_id, quality, user_msg)

            # Small delay between turns
            await asyncio.sleep(1)

        except Exception as e:
            print(f"[Turn {turn_idx + 1}] Error: {e}")
            break

    print(f"\nConversation {conv_id} completed ({len(conversation)} turns)")


async def main():
    """Run all test conversations."""
    async with httpx.AsyncClient() as client:
        # Test server health
        try:
            health = await client.get(f"{API_BASE}/health", timeout=5.0)
            print(f"Server health: {health.status_code}")
        except Exception as e:
            print(f"ERROR: Cannot connect to server at {API_BASE}")
            print(f"Make sure the RL server is running first!")
            print(f"Error: {e}")
            return

        print(f"\n{'='*60}")
        print(f"OpenClaw-Combine Fake Conversation Test")
        print(f"{'='*60}")
        print(f"Server: {API_BASE}")
        print(f"Total conversations: {len(CONVERSATIONS)}")
        print(f"Total turns: {sum(len(conv) for conv in CONVERSATIONS)}")

        # Run conversations sequentially
        for conv_id, conversation in enumerate(CONVERSATIONS, 1):
            await run_conversation(client, conversation, conv_id)
            # Delay between conversations
            await asyncio.sleep(2)

        print(f"\n{'='*60}")
        print(f"All test conversations completed!")
        print(f"{'='*60}")
        print(f"Check the training logs to see samples being submitted.")
        print(f"Look for '[OpenClaw-Combine] submitted' messages.")


if __name__ == "__main__":
    asyncio.run(main())
