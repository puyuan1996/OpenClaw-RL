#!/usr/bin/env python3
"""
Interactive Chat with RL Training Server

Real-time conversation with the training model. Your feedback (👍/👎 or text)
automatically trains the model in the background.

Usage:
    python interactive_chat.py

Commands:
    👍 or +1 or good     - Mark last response as good (positive reward)
    👎 or -1 or bad      - Mark last response as bad (negative reward)
    /new                 - Start new conversation session
    /history             - Show conversation history
    /stats               - Show training statistics
    /quit or /exit       - Exit
"""
import asyncio
import json
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import httpx

# ============================================================================
# Configuration
# ============================================================================
API_BASE = "http://0.0.0.0:30000"
API_KEY = "apiKey"
MODEL = "qwen3-4b"
TIMEOUT = 120.0

# Record file to monitor training progress
RECORD_FILE = Path(__file__).parent / "results" / "qwen3_4b_record.jsonl"
PRM_RECORD_FILE = Path(__file__).parent / "results" / "qwen3_4b_record_prm.jsonl"

# Color codes for terminal output
class Color:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================================
# Session Manager
# ============================================================================
class ChatSession:
    def __init__(self):
        self.session_id = f"interactive_{uuid.uuid4().hex[:8]}"
        self.messages = []
        self.start_time = time.time()
        self.turn_count = 0
        self.feedback_count = {"positive": 0, "negative": 0, "neutral": 0}

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self.turn_count += 1

    def record_feedback(self, feedback_type: str):
        self.feedback_count[feedback_type] = self.feedback_count.get(feedback_type, 0) + 1

    def get_duration(self) -> str:
        duration = int(time.time() - self.start_time)
        return f"{duration // 60}m {duration % 60}s"


# ============================================================================
# Training Statistics Monitor
# ============================================================================
class TrainingMonitor:
    def __init__(self):
        self.total_samples = 0
        self.recent_samples = deque(maxlen=10)

    def check_new_samples(self):
        """Check for new training samples in record files."""
        try:
            if RECORD_FILE.exists():
                with open(RECORD_FILE) as f:
                    lines = f.readlines()
                    new_count = len(lines)
                    if new_count > self.total_samples:
                        new_samples = new_count - self.total_samples
                        self.total_samples = new_count
                        # Parse last few samples
                        for line in lines[-new_samples:]:
                            try:
                                sample = json.loads(line)
                                self.recent_samples.append({
                                    "timestamp": sample.get("timestamp", ""),
                                    "session": sample.get("session_id", "")[:16],
                                    "reward": sample.get("reward", 0),
                                })
                            except:
                                pass
                        return new_samples
        except Exception as e:
            pass
        return 0

    def get_stats(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "recent_samples": list(self.recent_samples),
        }


# ============================================================================
# API Client
# ============================================================================
class ChatClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.session = ChatSession()
        self.monitor = TrainingMonitor()

    async def chat(self, user_input: str, show_thinking: bool = True) -> str:
        """Send chat request and return assistant response."""
        self.session.add_user_message(user_input)

        try:
            response = await self.client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": self.session.messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Session-ID": self.session.session_id,
                    "X-Turn-Type": "main",
                },
            )

            if response.status_code != 200:
                return f"[Error: {response.status_code}] {response.text}"

            result = response.json()
            assistant_msg = result["choices"][0]["message"]["content"]

            self.session.add_assistant_message(assistant_msg)

            # Check for new training samples
            new_samples = self.monitor.check_new_samples()
            if new_samples > 0:
                print(f"\n{Color.GREEN}✓ {new_samples} new training sample(s) submitted{Color.END}")

            return assistant_msg

        except Exception as e:
            return f"[Connection Error] {str(e)}"

    async def send_feedback(self, feedback_type: str, message: str = None):
        """Send feedback signal for the last turn."""
        if not self.session.messages or self.session.messages[-1]["role"] != "assistant":
            print(f"{Color.RED}No assistant response to give feedback on{Color.END}")
            return

        # Record feedback
        self.session.record_feedback(feedback_type)

        # Send implicit feedback as next user message
        if message:
            feedback_msg = message
        elif feedback_type == "positive":
            feedback_msg = "👍 Thanks, that's helpful!"
        elif feedback_type == "negative":
            feedback_msg = "👎 That's not quite right."
        else:
            feedback_msg = "I see."

        print(f"\n{Color.CYAN}→ Feedback: {feedback_msg}{Color.END}")

        # Send feedback message (this creates the "next state" for PRM evaluation)
        try:
            await self.client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": self.session.messages + [{"role": "user", "content": feedback_msg}],
                    "temperature": 0.7,
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Session-ID": self.session.session_id,
                    "X-Turn-Type": "main",
                },
            )

            # Wait a moment for async processing
            await asyncio.sleep(0.5)

            # Check for new samples
            new_samples = self.monitor.check_new_samples()
            if new_samples > 0:
                print(f"{Color.GREEN}✓ Feedback recorded, {new_samples} training sample(s) generated{Color.END}")
            else:
                print(f"{Color.YELLOW}⏳ Feedback sent, waiting for sample generation...{Color.END}")

        except Exception as e:
            print(f"{Color.RED}Failed to send feedback: {e}{Color.END}")

    def new_session(self):
        """Start a new conversation session."""
        self.session = ChatSession()
        print(f"\n{Color.GREEN}New session started: {self.session.session_id}{Color.END}\n")

    def show_history(self):
        """Show conversation history."""
        print(f"\n{Color.BOLD}=== Conversation History ==={Color.END}")
        print(f"Session: {self.session.session_id}")
        print(f"Duration: {self.session.get_duration()}")
        print(f"Turns: {self.session.turn_count}")
        print(f"Feedback: {self.session.feedback_count}")
        print()

        for i, msg in enumerate(self.session.messages):
            role = msg["role"]
            content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")

            if role == "user":
                print(f"{Color.BLUE}[{i+1}] You:{Color.END} {content}")
            else:
                print(f"{Color.GREEN}[{i+1}] Assistant:{Color.END} {content}")
        print()

    def show_stats(self):
        """Show training statistics."""
        stats = self.monitor.get_stats()
        print(f"\n{Color.BOLD}=== Training Statistics ==={Color.END}")
        print(f"Total training samples: {stats['total_samples']}")
        print(f"Recent samples:")

        if stats['recent_samples']:
            for sample in stats['recent_samples']:
                reward_color = Color.GREEN if sample['reward'] > 0 else Color.RED
                print(f"  • {sample['session']} | reward: {reward_color}{sample['reward']:+.1f}{Color.END}")
        else:
            print("  (no samples yet)")

        print()

    async def close(self):
        await self.client.aclose()


# ============================================================================
# Interactive Chat Loop
# ============================================================================
async def interactive_loop():
    """Main interactive chat loop."""
    client = ChatClient()

    # Print welcome message
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}")
    print(f"{Color.BOLD}Interactive Chat with RL Training Server{Color.END}")
    print(f"{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}\n")

    print(f"Server: {Color.YELLOW}{API_BASE}{Color.END}")
    print(f"Model: {Color.YELLOW}{MODEL}{Color.END}")
    print(f"Session: {Color.YELLOW}{client.session.session_id}{Color.END}\n")

    print(f"{Color.BOLD}Commands:{Color.END}")
    print(f"  {Color.GREEN}👍 +1 good{Color.END}  - Positive feedback")
    print(f"  {Color.RED}👎 -1 bad{Color.END}   - Negative feedback")
    print(f"  {Color.CYAN}/new{Color.END}        - New session")
    print(f"  {Color.CYAN}/history{Color.END}    - Show history")
    print(f"  {Color.CYAN}/stats{Color.END}      - Training stats")
    print(f"  {Color.CYAN}/quit{Color.END}       - Exit\n")

    print(f"{Color.MAGENTA}💡 Tip: Your feedback trains the model in real-time!{Color.END}\n")
    print(f"{Color.BOLD}{Color.CYAN}{'='*70}{Color.END}\n")

    # Test connection
    try:
        response = await client.client.get(f"{API_BASE}/health", timeout=5.0)
        if response.status_code == 200:
            print(f"{Color.GREEN}✓ Connected to RL server{Color.END}\n")
        else:
            print(f"{Color.RED}⚠ Server responded with status {response.status_code}{Color.END}\n")
    except Exception as e:
        print(f"{Color.RED}✗ Cannot connect to server: {e}{Color.END}")
        print(f"{Color.YELLOW}Make sure the RL server is running!{Color.END}\n")
        await client.close()
        return

    # Main loop
    try:
        while True:
            # Get user input
            try:
                user_input = input(f"{Color.BLUE}{Color.BOLD}You:{Color.END} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                break

            elif user_input.lower() == "/new":
                client.new_session()
                continue

            elif user_input.lower() in ["/history", "/h"]:
                client.show_history()
                continue

            elif user_input.lower() in ["/stats", "/s"]:
                client.show_stats()
                continue

            # Handle feedback
            elif user_input in ["👍", "+1", "good", "/good"]:
                await client.send_feedback("positive")
                continue

            elif user_input in ["👎", "-1", "bad", "/bad"]:
                await client.send_feedback("negative")
                continue

            # Regular chat
            print(f"\n{Color.GREEN}{Color.BOLD}Assistant:{Color.END} ", end="", flush=True)

            response = await client.chat(user_input)
            print(response)
            print()

            # Prompt for optional feedback
            print(f"{Color.CYAN}(Give feedback: 👍 👎, or press Enter to continue){Color.END}")

    except Exception as e:
        print(f"\n{Color.RED}Error: {e}{Color.END}")

    finally:
        # Cleanup
        print(f"\n{Color.BOLD}Session Summary:{Color.END}")
        print(f"  Turns: {client.session.turn_count}")
        print(f"  Duration: {client.session.get_duration()}")
        print(f"  Feedback: {client.session.feedback_count}")

        stats = client.monitor.get_stats()
        print(f"  Training samples generated: {stats['total_samples']}")

        print(f"\n{Color.GREEN}Thank you for training the model!{Color.END}\n")

        await client.close()


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    try:
        asyncio.run(interactive_loop())
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}Interrupted by user{Color.END}\n")
    except Exception as e:
        print(f"\n{Color.RED}Fatal error: {e}{Color.END}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
