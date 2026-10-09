# ComfyUI OpenAI API Proxy (Rust)

[🇨🇳 中文文档 (Chinese README)](README_CN.md)

A high-performance, asynchronous proxy server written in Rust that exposes an OpenAI-compatible API (`/v1/images/generations`) for your local or remote ComfyUI instance.

This allows you to use ComfyUI's powerful image generation workflows with any client that supports the OpenAI DALL-E API (e.g., ChatGPT web UIs, Open WebUI, LangChain, etc.).

## 🚀 Features

- **OpenAI Compatible**: Drop-in replacement for the `v1/images/generations` endpoint.
- **WebSocket Support**: Real-time task tracking via ComfyUI's WebSocket connection.
- **Dynamic Workflow Mapping**: specific `model` names in API requests map directly to JSON workflow files.
- **Parameter Injection**: Automatically injects prompts, negative prompts, seeds, and dimensions into your ComfyUI workflows.
- **High Performance**: Built with Rust, Axum, and Tokio for low latency and high concurrency.

## 🛠️ Prerequisites

- **Rust**: Ensure you have Rust and Cargo installed ([Install Rust](https://www.rust-lang.org/tools/install)).
- **ComfyUI**: A running instance of ComfyUI (local or remote).

## ⚙️ Configuration

1. **Config File**:
   Ensure `apps/api/config/config.yaml` exists. You can customize the server port and ComfyUI backend address here.

   ```yaml
   log_level: info

   server:
     host: "0.0.0.0"
     port: 8080        # The port this proxy will listen on

   comfyui_backend:
     host: "localhost" # Your ComfyUI IP
     port: 8188        # Your ComfyUI Port
     client_id: "openai-proxy-v1"
     workflows_folder: "./workflows" # Directory to store JSON workflows
     use_ws: true      # Recommended: true for real-time status updates

   routing:
     timeout_seconds: 300
     max_payload_size_mb: 10
   ```

2. **Environment Variable (Optional)**:
   You can specify a custom config path via the `CONFIG_PATH` environment variable.

### Optional response delay

Add this top-level section to `apps/api/config/config.yaml` (or the file selected by `CONFIG_PATH`):

```yaml
delay_layer:
  enabled: true
  min_seconds: 3.0
  max_seconds: 8.0
```

Each successful `/v1/images/generations` request waits for an independently sampled floating-point duration after all its images have been generated and its response assembled. This covers ordinary and dynamic-batch generation, once per HTTP request rather than per image. The wait is asynchronous; other requests and batch processing continue normally. Response status, headers, extensions, and body are returned unchanged.

The section is optional and disabled by default. Set `enabled: false` to bypass the wait; change the nonnegative `min_seconds` / `max_seconds` values to adjust the range. Omitted bounds default to 3 and 8 seconds. Invalid numeric ranges or recoverable delay failures skip the wait and log a warning; invalid YAML or incorrect field types still follow the existing configuration-load error handling. Errors, health checks, image-download authorization, and other endpoints are not delayed.

Configuration is read at startup: restart the API after changing it, and rebuild first when deploying source changes. Allow for generation time **plus** the maximum delay in client/reverse-proxy timeouts. Published images are protected from periodic cleanup while awaiting their generation response, including the optional delay and body transmission. The configured `image_cleanup.max_age_seconds` retention starts when the complete response body has been handed off to the HTTP transport and its body is released—not when the image was generated. This is a server-side handoff, not proof that the client has received or read the response. Each HTTP request has an independent lifetime, including siblings in one dynamic batch. Cancellation before body handoff removes the undelivered files. Successful one-time image downloads still consume their files immediately. Orphan files after a process restart retain the existing mtime-based safety-net cleanup. Protection is process-local; do not run multiple API processes with cleanup against the same output directory. These lifecycle rules also apply with the delay disabled or its middleware removed; `public_image_access` continues to control download access.

## 📂 Workflow Setup (Important!)

To use a specific model/workflow via the API, you must export it from ComfyUI in **API Format**.

1. Open **ComfyUI** in your browser.
2. Click the **Gear Icon** (Settings) and check **"Enable Dev mode Options"**.
3. Load or create your desired workflow.
4. Click the **"Save (API Format)"** button (do **not** use the regular "Save" button).
5. Rename the saved JSON file to the model name you want to use (e.g., `flux-dev.json`).
6. Place this file in the `workflows/` directory (or the folder defined in your config).

**Example Mapping:**
- If you request `model: "flux-dev"`, the proxy loads `./workflows/flux-dev.json`.

### Supported Nodes for Parameter Injection
The proxy looks for specific node types/titles to inject API parameters:
- **Prompt**: `CLIPTextEncode`, `CR Text`, `easy promptLine` (Title: "Positive Prompt" or default).
- **Negative Prompt**: `CLIPTextEncode` (Title: "Negative Prompt").
- **Seed**: `KSampler`, `easy seed`.
- **Size (WxH)**: `EmptyLatentImage`, `EmptySD3LatentImage`, `EmptyFlux2LatentImage`.

### 🧩 Customizing Node Mapping

If your workflow uses special custom nodes that are not automatically recognized by the default rules (e.g., complex node inputs or unique node names), you can easily modify the mapping logic yourself.

The mapping logic is located in `apps/api/src/comfyui.rs` inside the `create_json_payload` function.

**💡 Pro Tip**: The easiest way to adapt it is to upload your exported workflow (`.json`) and the project code (`src/comfyui.rs`) to an AI Agent (like Claude Code) and ask:

```
"Please help me modify the mapping logic in comfyui.rs to support [your_workflow].json."
```

## 🏃‍♂️ Running the Server

```bash
# Navigate to the api directory first
cd apps/api

# Run directly (development)
cargo run

# Or build release and run
cargo build --release

# Linux/macOS:
./target/release/comfyui-openai-api

# Windows (PowerShell):
.\target\release\comfyui-openai-api.exe
```

The server will start at `http://0.0.0.0:8080`.

## 🔌 Usage Examples

### 1. CURL

```bash
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "flux-dev",
    "prompt": "A cyberpunk city at night, neon lights, rain",
    "n": 1,
    "size": "1024x1024"
  }'
```

*Note: `model` must match the filename in your `workflows` folder (without .json).*

### 2. Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-needed"
)

response = client.images.generate(
    model="flux-dev",  # Matches ./workflows/flux-dev.json
    prompt="A cute cat sitting on a windowsill, watercolor style",
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].b64_json
# Process the base64 image...
```
## Reference Project
[ComfyUI OpenAI API](https://github.com/pnyxai/comfyui-openai-api)

## 📝 License

[MIT](LICENSE)

## Current server deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the deployed configuration, service files, and ComfyUI dependencies.
