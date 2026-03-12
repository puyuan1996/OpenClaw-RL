#!/usr/bin/env python3
"""
Simple Interactive Chat - Minimal and Stable Version

Chat with your training model. Give feedback with 👍/👎 to train it in real-time.
"""
import json
import sys
import time
import uuid

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

# Config
API_BASE = "http://0.0.0.0:30000"
API_KEY = "apiKey"


class SimpleChat:
    def __init__(self):
        self.client = httpx.Client(timeout=120.0)
        self.session_id = f"chat_{uuid.uuid4().hex[:8]}"
        self.messages = []
        self.turn = 0

    def chat(self, user_input):
        """Send message and get response."""
        self.messages.append({"role": "user", "content": user_input})

        try:
            resp = self.client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3-4b",
                    "messages": self.messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Session-ID": self.session_id,
                },
            )

            if resp.status_code != 200:
                return f"[Error {resp.status_code}]"

            result = resp.json()
            assistant = result["choices"][0]["message"]["content"]
            self.messages.append({"role": "assistant", "content": assistant})
            self.turn += 1
            return assistant

        except Exception as e:
            return f"[Error: {e}]"

    def feedback(self, is_good):
        """Send feedback for last response."""
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return

        msg = "👍 Good!" if is_good else "👎 Not quite right."
        print(f"\n→ {msg}")

        try:
            # Send feedback (creates training signal)
            self.client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3-4b",
                    "messages": self.messages + [{"role": "user", "content": msg}],
                    "max_tokens": 1,
                },
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "X-Session-ID": self.session_id,
                },
            )
            print("✓ Feedback sent\n")
        except:
            print("✗ Failed to send feedback\n")


def main():
    print("\n" + "="*60)
    print("  Interactive Chat with RL Training Model")
    print("="*60)
    print(f"\nServer: {API_BASE}")
    print("\nCommands:")
    print("  👍 or +    → Positive feedback")
    print("  👎 or -    → Negative feedback")
    print("  /new       → New session")
    print("  /quit      → Exit")
    print("\n" + "="*60 + "\n")

    chat = SimpleChat()

    # Test connection
    try:
        httpx.get(f"{API_BASE}/health", timeout=5.0)
        print("✓ Connected\n")
    except:
        print("✗ Cannot connect! Start the RL server first.\n")
        return

    print(f"Session: {chat.session_id}\n")

    while True:
        try:
            inp = input("You: ").strip()

            if not inp:
                continue

            # Commands
            if inp.lower() in ["/quit", "/q", "/exit"]:
                break
            elif inp.lower() == "/new":
                chat = SimpleChat()
                print(f"\n→ New session: {chat.session_id}\n")
                continue
            elif inp in ["👍", "+", "+1", "good"]:
                chat.feedback(True)
                continue
            elif inp in ["👎", "-", "-1", "bad"]:
                chat.feedback(False)
                continue

            # Chat
            print("\nAssistant: ", end="", flush=True)
            response = chat.chat(inp)
            print(response + "\n")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print(f"\n→ Session ended. Turns: {chat.turn}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}\n")
