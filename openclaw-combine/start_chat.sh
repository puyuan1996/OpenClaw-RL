#!/bin/bash
# Quick start script for interactive chat

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "Starting Interactive Chat"
echo "=================================="
echo ""

# Check if server is running
if ! curl -s http://0.0.0.0:30000/health > /dev/null 2>&1; then
    echo "❌ RL server is not running!"
    echo ""
    echo "Please start it first:"
    echo "  cd $SCRIPT_DIR/../slime"
    echo "  bash $SCRIPT_DIR/run_qwen3_4b_openclaw_combine_pu.sh"
    echo ""
    exit 1
fi

echo "✓ RL server is running"
echo ""

# Install httpx if needed
python3 -c "import httpx" 2>/dev/null || {
    echo "Installing httpx..."
    pip install httpx -q
}

# Run interactive chat
echo "Starting chat..."
echo ""
# python3 "$SCRIPT_DIR/simple_chat.py"
python3 "$SCRIPT_DIR/interactive_chat.py"

