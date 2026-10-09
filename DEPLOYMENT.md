# Current GPU server deployment

This repository contains the deployed API source, dependency lockfile, API-format workflows, startup script, and sanitized deployment configuration. The original upstream history and MIT license are retained.

## Repositories

- Origin: https://github.com/EldenPdx/Comfyui2OpenAI.git
- Upstream: https://github.com/qup1010/Comfyui2Openai.git

## Files and private state

- `apps/api/config/config.deploy.sample.yaml`: sanitized snapshot of the deployed configuration. SMTP is disabled and mailbox addresses are placeholders.
- `apps/api/config/config.sample.yaml`: generic configuration example.
- `apps/api/workflows/IMG2.0-S.json`: the active workflow, including parallel image saving.
- `workflows/`: additional workflow definitions; these are not loaded by the current startup script.
- `start_api.sh`: starts the release binary in the `ComfyApi` tmux session.
- `deploy/comfyui-openai-api.service`: the API systemd unit used on the GPU server.
- `deploy/Caddyfile`: the Caddy reverse-proxy configuration used on the GPU server.

The real `apps/api/config/config.yaml`, environment files, local backup snapshots, logs, and build artifacts stay outside Git. Keep private configuration separately. Supply the `SMTP_PASSWORD` environment variable to the API process if SMTP alerts are enabled; never commit its value.

## Restore on another server

1. Install Rust, tmux, and Caddy. Deploy the compatible ComfyUI backend and its dependencies first.
2. Clone this repository and build the API:

   ```sh
   cd Comfyui2OpenAI/apps/api
   cargo build --release --locked
   cp config/config.deploy.sample.yaml config/config.yaml
   ```

3. Adjust `image_cleanup.output_directory` and the service's user, group, home directory, working directory, and executable path for the new server. Keep `comfyui_backend.workflows_folder: ./workflows` relative to `apps/api`, where the startup script launches the binary.
4. Use `start_api.sh` for a manual start, or adapt and install the provided API systemd unit. Its `Requires=comfyui.service` expects a working ComfyUI systemd unit on the new server. Install the Caddy configuration and ensure the log directory is writable by Caddy.
5. Verify `GET /health`, `GET /v1/images/gpu-count`, image generation, and retrieval of returned image URLs before switching client traffic.

The API reads, renames, and deletes files in ComfyUI's local output directory. Keep both services on the same server unless shared storage or the image-access implementation is explicitly adapted. Drain outstanding requests and retrieve pending images before cutting over.

## Ports

- Public HTTP: Caddy on `8080`.
- API: `127.0.0.1:18081`.
- Backend registration/control: `127.0.0.1:18082`.
- ComfyUI instances: `127.0.0.1:8188` and `127.0.0.1:8189`.

## Separate ComfyUI dependencies

ComfyUI is a separate project and is not bundled in this API repository. The current server uses:

- ComfyUI commit `4e024cb1e8423be735b919c585669dc189735c7b`.
- Custom node files `custom_nodes/comfyui_dynamic_batch.py` and `custom_nodes/comfyui_parallel_save.py`, providing `OpenAIBatchCLIPTextEncode`, `OpenAIBatchNoise`, and `ParallelSaveImage`.
- `models/checkpoints/sd_xl_turbo_1.0_fp16.safetensors`.
- `gpu_instance_coordinator.py` and `start_comfy.sh` for per-GPU instances and registration with the API control port.
- Python 3.12.3 and PyTorch 2.14.0+cu130, running on two NVIDIA RTX 5090 GPUs with driver 595.84. Re-evaluate driver compatibility and batch capacity when changing hardware.

Back up those ComfyUI files and model weights separately. Preserve API workflow filenames and the `Positive Prompt` / `Negative Prompt` node titles. The coordinator should discover the new machine's GPU UUIDs rather than reuse the old machine's runtime state.
