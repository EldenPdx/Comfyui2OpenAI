#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$PROJECT_DIR/apps/api"
SESSION_NAME="ComfyApi"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    echo "API service is already running in tmux session '$SESSION_NAME'."
    exit 0
fi

API_BINARY="$API_DIR/target/release/comfyui-openai-api"
if [[ ! -x "$API_BINARY" ]]; then
    echo "API release binary not found: $API_BINARY" >&2
    echo "Build it with: cd '$API_DIR' && cargo build --release" >&2
    exit 1
fi

for command in tmux; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "$command is not installed or not available in PATH." >&2
        exit 1
    fi
done

tmux new-session -d -s "$SESSION_NAME" -c "$API_DIR" \
    "exec '$API_BINARY'"

echo "API service started in tmux session '$SESSION_NAME'."
echo "Attach with: tmux attach-session -t '$SESSION_NAME'"
