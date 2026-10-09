# ComfyUI OpenAI API Proxy (Rust)

[🇺🇸 English README](README.md)

这是一个使用 Rust 编写的高性能异步代理服务器，它为你的本地或远程 ComfyUI 实例提供了一个兼容 OpenAI 标准的 API 接口 (`/v1/images/generations`)。

这使得你可以使用任何支持 OpenAI DALL-E API 的客户端（例如 ChatGPT Web UI、Open WebUI、LangChain 等）来调用强大的 ComfyUI 图像生成工作流。

## 🚀 主要功能

- **OpenAI 兼容**: 完美替代 `v1/images/generations` 端点。
- **WebSocket 支持**: 通过 ComfyUI 的 WebSocket 连接进行实时的任务状态追踪。
- **动态工作流映射**: API 请求中的 `model` 名称直接映射到本地的 JSON 工作流文件。
- **智能参数注入**: 自动将 Prompt（正向/负向）、随机种子 (Seed) 和图片尺寸注入到 ComfyUI 工作流的对应节点中。
- **高性能**: 基于 Rust、Axum 和 Tokio 构建，低延迟、高并发。

## 🛠️ 前置要求

- **Rust**: 确保你已安装 Rust 和 Cargo ([安装 Rust](https://www.rust-lang.org/tools/install))。
- **ComfyUI**: 一个正在运行的 ComfyUI 实例（本地或远程均可）。

## ⚙️ 配置说明

1. **配置文件**:
   确保 `apps/api/config/config.yaml` 文件存在。你可以在此自定义服务器端口和 ComfyUI 后端地址。

   ```yaml
   log_level: info

   server:
     host: "0.0.0.0"
     port: 8080        # 代理服务器监听的端口

   comfyui_backend:
     host: "localhost" # 你的 ComfyUI IP 地址
     port: 8188        # 你的 ComfyUI 端口
     client_id: "openai-proxy-v1"
     workflows_folder: "./workflows" # 存放 JSON 工作流文件的目录
     use_ws: true      # 推荐: true (开启实时状态更新)

   routing:
     timeout_seconds: 300
     max_payload_size_mb: 10
   ```

2. **环境变量 (可选)**:
   你可以通过设置 `CONFIG_PATH` 环境变量来指定配置文件的路径。

### 可选响应延迟

在 `apps/api/config/config.yaml`（或 `CONFIG_PATH` 指定的配置文件）顶层添加：

```yaml
delay_layer:
  enabled: true
  min_seconds: 3.0
  max_seconds: 8.0
```

每个成功的 `/v1/images/generations` 请求，在所有图片生成完成、响应组装完毕后，独立随机等待上述范围内的浮点秒数。普通生成和动态批处理均适用，每个 HTTP 请求只延迟一次，而非每张图片一次。异步等待不会阻塞其他请求或批处理，响应状态码、响应头、扩展信息及响应体保持原样。

配置项可省略，默认关闭。设置 `enabled: false` 可直接跳过延迟；修改非负的 `min_seconds` / `max_seconds` 可调整范围，省略上下界时分别使用 3 和 8 秒。无效数值范围或可恢复的延迟异常会记录警告并跳过等待；YAML 格式或字段类型错误仍按既有配置加载错误处理。错误响应、健康检查、图片下载鉴权和其他接口不延迟。

配置在启动时读取，修改后须重启 API；部署源码变更时需先重新构建。客户端和反向代理超时应覆盖“生成耗时 + 最大延迟”。生成响应等待交付期间（包括可选延迟和响应体传输），已发布图片受保护，不会被定时清理。完整响应体交给 HTTP 传输层并释放响应体后，才开始计算 `image_cleanup.max_age_seconds` 保留期，而不是从图片生成时计算。这是服务端交付边界，不代表客户端已经收到或读取。每个 HTTP 请求独立计时，同一动态批次中的不同请求也互不影响；交付前取消请求会回收未交付图片。一次性图片下载成功后仍立即删除对应文件。进程重启后的孤儿文件继续按文件修改时间兜底清理。保护登记为进程内状态，不应让多个 API 进程同时清理同一个输出目录。关闭延迟或移除延迟中间件时，这些生命周期规则仍然生效；图片下载权限仍由 `public_image_access` 控制。

## 📂 工作流设置 (重要!)

要通过 API 使用特定的模型/工作流，你必须从 ComfyUI 中以 **API 格式** 导出它。

1. 打开浏览器中的 **ComfyUI** 界面。
2. 点击右侧菜单的 **齿轮图标** (设置)，勾选 **"Enable Dev mode Options"** (开启开发者模式选项)。
3. 加载或搭建你想要使用的工作流。
4. 点击面板上的 **"Save (API Format)"** 按钮 (注意：**不要**使用普通的 "Save" 按钮)。
5. 将保存的 JSON 文件重命名为你想要的模型名称 (例如 `flux-dev.json`)。
6. 将该文件放入 `workflows/` 目录 (或你在配置中定义的目录)。

**映射示例:**
- 如果 API 请求中指定 `model: "flux-dev"`, 代理服务器将自动加载 `./workflows/flux-dev.json`。

### 支持参数注入的节点
代理服务器会查找特定的节点类型/标题来自动注入 API 参数：
- **Prompt (正向提示词)**: `CLIPTextEncode`, `CR Text`, `easy promptLine` (标题需包含: "Positive Prompt", "CLIP文本编码", "提示词行" 或保持默认)。
- **Negative Prompt (负向提示词)**: `CLIPTextEncode` (标题需包含: "Negative Prompt" 或 "条件零化")。
- **Seed (随机种子)**: `KSampler`, `easy seed`。
- **Size (尺寸 WxH)**: `EmptyLatentImage`, `EmptySD3LatentImage`, `EmptyFlux2LatentImage`。

### 🧩 自定义节点映射

如果你使用的工作流非常特殊（例如使用了自定义节点、复杂的输入结构或特殊的节点名称），导致默认规则无法识别，你可以自行修改配对逻辑。

映射逻辑位于 `apps/api/src/comfyui.rs` 文件中的 `create_json_payload` 函数内。

**💡 小技巧**: 最简单的修改方法是将你导出的工作流文件 (`.json`) 和项目代码 (`src/comfyui.rs`) 发给 AI Agent (如 Copilot，Claude Code等等) 并提问：

```
"请帮我修改 comfyui.rs 里的映射逻辑来适配[你的工作流].json"

```

修改起来非常简单！

## 🏃‍♂️ 运行服务器

```bash
# 首先进入 api 目录
cd apps/api

# 直接运行 (开发模式)
cargo run

# 编译 release 版本并运行
cargo build --release

# Linux/macOS:
./target/release/comfyui-openai-api

# Windows (PowerShell):
.\target\release\comfyui-openai-api.exe
```

服务器启动后将监听于 `http://0.0.0.0:8080`。

## 🔌 调用示例

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

*注意: `model` 字段必须与 `workflows` 文件夹中的文件名一致 (不含 .json 后缀)。*

### 2. Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-needed"
)

response = client.images.generate(
    model="flux-dev",  # 对应 ./workflows/flux-dev.json
    prompt="A cute cat sitting on a windowsill, watercolor style",
    size="1024x1024",
    quality="standard",
    n=1,
)

# 获取 Base64 编码的图片
image_url = response.data[0].b64_json
# 后续处理...
```


## 参考项目
[ComfyUI OpenAI API](https://github.com/pnyxai/comfyui-openai-api)

## 📝 许可证

[MIT](LICENSE)

## 当前服务器部署

当前配置、服务文件和 ComfyUI 依赖见 [DEPLOYMENT.md](DEPLOYMENT.md)。
