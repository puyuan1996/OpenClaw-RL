#!/usr/bin/env python3
"""
Simple request test - minimal version to quickly test the API.

This sends individual requests without conversation tracking,
just to verify the server is responding correctly.
"""
import httpx

API_BASE = "http://0.0.0.0:30000"
API_KEY = "apiKey"


def test_simple_chat():
    """Send a single chat request."""
    print(f"Testing {API_BASE}/v1/chat/completions ...")

    with httpx.Client(timeout=60.0) as client:
        try:
            response = client.post(
                f"{API_BASE}/v1/chat/completions",
                json={
                    "model": "qwen3-4b",
                    "messages": [
                        {"role": "user", "content": "Hello! What is 2+2?"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100,
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
            )

            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result['choices'][0]['message']['content']}")
                print("\n✅ API is working!")
            else:
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nMake sure the RL server is running:")
            print("  cd slime")
            print("  bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_pu.sh")


if __name__ == "__main__":
    test_simple_chat()
