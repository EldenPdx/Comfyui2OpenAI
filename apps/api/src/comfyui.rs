//! ComfyUI Backend Communication & Request Processing Module
//!
//! This module handles the translation between OpenAI API format and ComfyUI format,
//! manages image generation requests, and retrieves generated images from the backend.

use axum::{
    body::{Body, BodyDataStream, Bytes, HttpBody},
    extract::{Query, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{Response as AxumResponse},
};
use log::{debug, error, warn, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore}, time::{timeout, timeout_at, Instant}};
use std::{cmp::Reverse, collections::{HashMap, VecDeque}, str::FromStr, sync::{Arc, LazyLock, Mutex, Weak}, time::Duration};
use crate::ws::WebSocketManager;
use std::time::{SystemTime, UNIX_EPOCH};
use std::fs;
use std::path::{Component, Path, PathBuf};
use rand::Rng;
use futures::Stream;
use std::{pin::Pin, task::{Context, Poll}};

pub async fn health(State(state): State<Arc<ProxyState>>) -> StatusCode {
    if state.backend_pool.has_ready() { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }
}

/// Metadata for ComfyUI output images
///
/// Used to construct the query string for retrieving images.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ImageFile {
    /// Filename in the ComfyUI output directory
    filename: String,
    /// Subfolder within the output directory
    subfolder: String,
    /// Image type (e.g., "output" for final result, "temp" for preview)
    #[serde(rename = "type")]
    type_field: String,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageQuality {
    Low,
    Medium,
    High,
}

fn request_image_quality(
    request: &serde_json::Map<String, Value>,
) -> Result<ImageQuality, ProxyError> {
    let random_quality = || match rand::rng().random_range(0..3) {
        0 => ImageQuality::Low,
        1 => ImageQuality::Medium,
        _ => ImageQuality::High,
    };
    match request.get("quality") {
        Some(Value::String(value)) if value == "low" => Ok(ImageQuality::Low),
        Some(Value::String(value)) if value == "medium" => Ok(ImageQuality::Medium),
        Some(Value::String(value)) if value == "high" => Ok(ImageQuality::High),
        None | Some(Value::Null) => Ok(random_quality()),
        Some(Value::String(value)) if value == "auto" => Ok(random_quality()),
        _ => Err(ProxyError::Validation(
            "quality must be 'auto', 'low', 'medium', or 'high'".to_string(),
        )),
    }
}

#[derive(Debug, Clone)]
struct UsageEstimate {
    input_tokens: u64,
    output_tokens_per_image: u64,
}

fn gpt_image2_output_tokens(width: u64, height: u64, quality: ImageQuality) -> u64 {
    let q = match quality {
        ImageQuality::Low => 16_u128,
        ImageQuality::Medium => 48,
        ImageQuality::High => 96,
    };
    let width = width as u128;
    let height = height as u128;
    let long = width.max(height);
    let short = width.min(height);
    let aspect_factor = (2 * q * short + long) / (2 * long);
    let numerator = q * aspect_factor * (2_000_000 + width * height);
    u64::try_from(numerator.div_ceil(4_000_000)).unwrap_or(u64::MAX)
}

fn estimate_usage(payload: &[u8], quality: ImageQuality) -> Result<UsageEstimate, ProxyError> {
    let json: Value = serde_json::from_slice(payload)
        .map_err(|e| ProxyError::Json(format!("Failed to estimate usage: {}", e)))?;
    let prompt = json.get("prompt")
        .and_then(|value| value.as_object())
        .ok_or_else(|| ProxyError::Json("Failed to estimate usage from workflow".to_string()))?;

    let mut input_tokens = 0;
    let mut width = 512;
    let mut height = 512;

    for node in prompt.values() {
        match node.get("class_type").and_then(|value| value.as_str()) {
            Some("CLIPTextEncode") => {
                if let Some(text) = node.pointer("/inputs/text").and_then(|value| value.as_str()) {
                    input_tokens += estimate_clip_tokens(text);
                }
            }
            Some("EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage") => {
                width = node.pointer("/inputs/width")
                    .and_then(|value| value.as_u64())
                    .filter(|value| *value > 0)
                    .unwrap_or(width);
                height = node.pointer("/inputs/height")
                    .and_then(|value| value.as_u64())
                    .filter(|value| *value > 0)
                    .unwrap_or(height);
            }
            _ => {}
        }
    }

    Ok(UsageEstimate {
        input_tokens,
        output_tokens_per_image: gpt_image2_output_tokens(width, height, quality),
    })
}

fn estimate_clip_tokens(text: &str) -> u64 {
    let mut content_tokens = 0_u64;
    let mut ascii_run = 0_u64;

    let flush_ascii_run = |content_tokens: &mut u64, ascii_run: &mut u64| {
        if *ascii_run > 0 {
            *content_tokens += ascii_run.div_ceil(6);
            *ascii_run = 0;
        }
    };

    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            ascii_run += 1;
        } else {
            flush_ascii_run(&mut content_tokens, &mut ascii_run);
            if !character.is_whitespace() {
                content_tokens += if character.is_ascii() { 1 } else { 3 };
            }
        }
    }
    flush_ascii_run(&mut content_tokens, &mut ascii_run);

    // SDXL encodes each prompt with CLIP-L and CLIP-G in 75-token chunks,
    // with start/end tokens added to every chunk.
    let chunks = content_tokens.div_ceil(75).max(1);
    (content_tokens + chunks * 2) * 2
}

fn build_image_response(data: Vec<Value>, created: i64, usage: &UsageEstimate) -> Value {
    let image_count = data.len() as u64;
    let output_tokens = usage.output_tokens_per_image
        .saturating_mul(image_count)
        .saturating_add(
            rand::rng()
                .random_range(7_500..=15_000_u64)
                .saturating_mul(image_count),
        );
    let total_tokens = usage.input_tokens.saturating_add(output_tokens);

    serde_json::json!({
        "created": created,
        "data": data,
        "usage": {
            "input_tokens": usage.input_tokens,
            "input_tokens_details": {
                "image_tokens": 0,
                "text_tokens": usage.input_tokens
            },
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "output_tokens_details": {
                "image_tokens": output_tokens,
                "text_tokens": 0
            }
        }
    })
}


const MAX_BATCH_SIZE: usize = 16;
const BATCH_WAIT_MS: u64 = 15;
pub(crate) const BATCH_QUEUE_CAPACITY: usize = 128;
const MAX_IMAGES_PER_REQUEST: usize = 10;
const PUBLIC_IMAGE_ORIGIN: &str = "http://106.75.237.89:8080";

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchKey {
    model: String,
    width: u64,
    height: u64,
    authorization: Option<String>,
    query_string: String,
}

pub(crate) struct BatchSlot {
    key: BatchKey,
    request_id: String,
    prompt: String,
    negative_prompt: String,
    seed: u64,
    seed_offset: usize,
    deadline: Instant,
    _capacity: Arc<OwnedSemaphorePermit>,
    reply: oneshot::Sender<Result<PendingImage, ProxyError>>,
}

impl BatchSlot {
    #[cfg(test)]
    fn test_slot(prompt: &str, negative_prompt: &str, seed: u64, seed_offset: usize) -> Self {
        let (reply, receiver) = oneshot::channel();
        std::mem::forget(receiver);
        let capacity = Arc::new(Semaphore::new(1));
        let capacity = Arc::new(capacity.try_acquire_owned().unwrap());
        Self {
            key: BatchKey {
                model: "IMG2.0-S".to_string(),
                width: 512,
                height: 512,
                authorization: None,
                query_string: String::new(),
            },
            request_id: "test".to_string(),
            prompt: prompt.to_string(),
            negative_prompt: negative_prompt.to_string(),
            seed,
            seed_offset,
            deadline: Instant::now() + Duration::from_secs(60),
            _capacity: capacity,
            reply,
        }
    }
}

pub(crate) type BatchSender = mpsc::Sender<BatchSlot>;


fn reserve_batch_capacity(
    capacity: &Arc<Semaphore>,
    slots: usize,
) -> Result<Arc<OwnedSemaphorePermit>, ProxyError> {
    capacity.clone()
        .try_acquire_many_owned(slots as u32)
        .map(Arc::new)
        .map_err(|_| ProxyError::Busy("Dynamic batch queue is full".to_string()))
}

fn batch_queue_order(queues: &[BatchQueue], start: usize) -> Vec<usize> {
    let mut indexes = (0..queues.len()).collect::<Vec<_>>();
    indexes.sort_by_key(|index| (
        Reverse(queues[*index].capacity.available_permits()),
        (*index + queues.len() - start) % queues.len(),
    ));
    indexes
}


async fn collect_batch(
    first: BatchSlot,
    receiver: &mut mpsc::Receiver<BatchSlot>,
    pending: &mut VecDeque<BatchSlot>,
    max_batch_size: usize,
    wait: Duration,
) -> Vec<BatchSlot> {
    let key = first.key.clone();
    let deadline = Instant::now() + wait;
    let mut batch = Vec::with_capacity(max_batch_size);
    if !first.reply.is_closed() {
        batch.push(first);
    }

    while batch.len() < max_batch_size {
        while let Some(index) = pending.iter().position(|slot| slot.reply.is_closed() || slot.key == key) {
            let slot = pending.remove(index).expect("pending index must exist");
            if !slot.reply.is_closed() {
                batch.push(slot);
                if batch.len() == max_batch_size {
                    return batch;
                }
            }
        }

        match timeout_at(deadline, receiver.recv()).await {
            Ok(Some(slot)) if slot.reply.is_closed() => {}
            Ok(Some(slot)) if slot.key == key => batch.push(slot),
            Ok(Some(slot)) => pending.push_back(slot),
            Ok(None) | Err(_) => break,
        }
    }

    batch
}

fn create_batch_json_payload(
    slots: &[BatchSlot],
    workflows: &HashMap<String, Value>,
    client_id: &str,
    batch_id: &str,
) -> Result<Bytes, ProxyError> {
    let first = slots.first()
        .ok_or_else(|| ProxyError::Internal("Cannot build an empty batch".to_string()))?;
    if slots.len() > MAX_BATCH_SIZE || slots.iter().any(|slot| slot.key != first.key) {
        return Err(ProxyError::Internal("Batch contains incompatible slots".to_string()));
    }

    let workflow = workflows.get(&first.key.model)
        .ok_or_else(|| ProxyError::Json(format!("Workflow '{}' not found", first.key.model)))?;
    let mut payload = serde_json::json!({
        "prompt": workflow.clone(),
        "client_id": client_id,
    });
    let prompt = payload["prompt"].as_object_mut()
        .ok_or_else(|| ProxyError::Json("Workflow prompt must be an object".to_string()))?;
    let positive = serde_json::to_string(&slots.iter().map(|slot| &slot.prompt).collect::<Vec<_>>())
        .map_err(|error| ProxyError::Json(format!("Failed to serialize batch prompts: {}", error)))?;
    let negative = serde_json::to_string(&slots.iter().map(|slot| &slot.negative_prompt).collect::<Vec<_>>())
        .map_err(|error| ProxyError::Json(format!("Failed to serialize batch negative prompts: {}", error)))?;
    let seeds = serde_json::to_string(&slots.iter().map(|slot| {
        serde_json::json!({"seed": slot.seed, "offset": slot.seed_offset})
    }).collect::<Vec<_>>())
        .map_err(|error| ProxyError::Json(format!("Failed to serialize batch seeds: {}", error)))?;

    let mut found_positive = false;
    let mut found_negative = false;
    let mut found_noise = false;
    let mut found_latent = false;
    let mut found_output = false;

    for node in prompt.values_mut() {
        let class_type = node.get("class_type").and_then(Value::as_str).unwrap_or("").to_string();
        let title = node.pointer("/_meta/title").and_then(Value::as_str).unwrap_or("").to_string();
        let Some(inputs) = node.get_mut("inputs").and_then(Value::as_object_mut) else {
            continue;
        };

        match class_type.as_str() {
            "EmptyLatentImage" => {
                inputs.insert("width".to_string(), serde_json::json!(first.key.width));
                inputs.insert("height".to_string(), serde_json::json!(first.key.height));
                inputs.insert("batch_size".to_string(), serde_json::json!(slots.len()));
                found_latent = true;
            }
            "OpenAIBatchCLIPTextEncode" if title == "Positive Prompt" => {
                inputs.insert("prompts_json".to_string(), Value::String(positive.clone()));
                found_positive = true;
            }
            "OpenAIBatchCLIPTextEncode" if title == "Negative Prompt" => {
                inputs.insert("prompts_json".to_string(), Value::String(negative.clone()));
                found_negative = true;
            }
            "OpenAIBatchNoise" => {
                inputs.insert("seeds_json".to_string(), Value::String(seeds.clone()));
                found_noise = true;
            }
            "SaveImage" | "ParallelSaveImage" => {
                let prefix = inputs.get("filename_prefix").and_then(Value::as_str).unwrap_or("ComfyUI");
                inputs.insert("filename_prefix".to_string(), Value::String(format!("{}_{}", prefix, batch_id)));
                found_output = true;
            }
            _ => {}
        }
    }

    if !(found_positive && found_negative && found_noise && found_latent && found_output) {
        return Err(ProxyError::Json(
            "Workflow is missing required dynamic batch nodes".to_string(),
        ));
    }

    serde_json::to_vec(&payload)
        .map(Bytes::from)
        .map_err(|error| ProxyError::Json(format!("Failed to serialize batch workflow: {}", error)))
}

// =================================================================================
//  Section: Workflow Loading
// =================================================================================

/// Workflow Loader
///
/// Responsible for loading ComfyUI workflow JSON files from a specified directory.
/// These files define the specific image generation process and are referenced by the `model` name in API requests.
pub struct WorkflowsLoader;

impl WorkflowsLoader {
    /// Scans and loads all JSON files from the folder
    ///
    /// Uses the filename (without extension) as the key. This allows clients to specify the workflow
    /// using the `model` field in the request.
    ///
    /// # Arguments
    /// * `folder_path` - Path to the directory containing workflow JSON files
    ///
    /// # Returns
    /// - `HashMap<String, Value>`: Map of Workflow Name -> JSON Content
    /// - `String`: Error message if directory doesn't exist or JSON parsing fails
    ///
    /// # Example
    /// ```no_run
    /// let workflows = WorkflowsLoader::load_from_folder("./workflows")?;
    /// // Access workflow with: workflows.get("animagine-xl-4")
    /// ```
    pub fn load_from_folder(folder_path: &str) -> Result<HashMap<String, Value>, String> {
        let path = Path::new(folder_path);

        // Verify folder existence
        if !path.exists() {
            return Err(format!("Workflows folder does not exist: {}", folder_path));
        }

        // Verify it is a directory
        if !path.is_dir() {
            return Err(format!("Workflows path is not a directory: {}", folder_path));
        }

        let mut workflows = HashMap::new();

        // Read all entries in the directory
        let entries = fs::read_dir(path)
            .map_err(|e| format!("Failed to read workflows directory: {}", e))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let file_path = entry.path();

            // Process only JSON files
            if file_path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
            {
                // Extract filename as workflow name
                let filename = file_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or_else(|| {
                        format!("Failed to get filename for: {:?}", file_path)
                    })?
                    .to_string();

                // Read JSON content
                let file_content = fs::read_to_string(&file_path)
                    .map_err(|e| {
                        format!("Failed to read JSON file {}: {}", file_path.display(), e)
                    })?;

                // Parse JSON
                let json_value: Value = serde_json::from_str(&file_content)
                    .map_err(|e| {
                        format!("Failed to parse JSON from {}: {}", file_path.display(), e)
                    })?;

                info!("✅ Loaded workflow: {}", filename);
                workflows.insert(filename, json_value);
            }
        }

        info!("📦 Successfully loaded {} workflow(s)", workflows.len());
        Ok(workflows)
    }
}

use crate::proxy::{BatchQueue, ProxyState, ProxyError, handle_request_error, handle_timeout_error};

// =================================================================================
//  Section: Request Handling (Main Logic)
// =================================================================================

#[derive(Debug)]
struct GenerationInput {
    model: String,
    prompt: String,
    negative_prompt: String,
    n: usize,
    width: u64,
    height: u64,
    seed: u64,
    usage: UsageEstimate,
}

fn request_model(body: &[u8]) -> Result<String, ProxyError> {
    let json: Value = serde_json::from_slice(body)
        .map_err(|error| ProxyError::Json(format!("Failed to parse JSON: {}", error)))?;
    let request = json.as_object()
        .ok_or_else(|| ProxyError::Json("Request body must be a JSON object".to_string()))?;
    Ok(request.get("model").and_then(Value::as_str).unwrap_or("IMG2.0-S").to_string())
}

fn workflow_dimensions(workflow: &Value) -> (u64, u64) {
    workflow.as_object().and_then(|nodes| {
        nodes.values().find_map(|node| {
            match node.get("class_type").and_then(Value::as_str) {
                Some("EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage") => Some((
                    node.pointer("/inputs/width").and_then(Value::as_u64).unwrap_or(512),
                    node.pointer("/inputs/height").and_then(Value::as_u64).unwrap_or(512),
                )),
                _ => None,
            }
        })
    }).unwrap_or((512, 512))
}

fn workflow_negative_prompt(workflow: &Value) -> String {
    workflow.as_object().and_then(|nodes| {
        nodes.values().find_map(|node| {
            if node.pointer("/_meta/title").and_then(Value::as_str) != Some("Negative Prompt") {
                return None;
            }
            if let Some(text) = node.pointer("/inputs/text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            node.pointer("/inputs/prompts_json").and_then(Value::as_str)
                .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
                .and_then(|values| values.into_iter().next())
        })
    }).unwrap_or_default()
}

fn parse_generation_input(
    body: &[u8],
    workflows: &HashMap<String, Value>,
) -> Result<GenerationInput, ProxyError> {
    let json: Value = serde_json::from_slice(body)
        .map_err(|error| ProxyError::Json(format!("Failed to parse JSON: {}", error)))?;
    let request = json.as_object()
        .ok_or_else(|| ProxyError::Json("Request body must be a JSON object".to_string()))?;
    let model = request.get("model").and_then(Value::as_str).unwrap_or("IMG2.0-S").to_string();
    let workflow = workflows.get(&model)
        .ok_or_else(|| ProxyError::Json(format!("Workflow '{}' not found", model)))?;
    let prompt = request.get("prompt").and_then(Value::as_str)
        .ok_or_else(|| ProxyError::Validation("prompt must be a string".to_string()))?
        .to_string();
    let negative_prompt = request.get("negative_prompt").and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| workflow_negative_prompt(workflow));
    let n = request.get("n").map(|value| value.as_u64()).unwrap_or(Some(1))
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| (1..=MAX_IMAGES_PER_REQUEST).contains(value))
        .ok_or_else(|| ProxyError::Validation(format!(
            "n must be between 1 and {}", MAX_IMAGES_PER_REQUEST
        )))?;

    let (mut width, mut height) = workflow_dimensions(workflow);
    if let Some(size) = request.get("size") {
        let size = size.as_str()
            .ok_or_else(|| ProxyError::Validation("size must be a string".to_string()))?;
        if size != "auto" {
            let (requested_width, requested_height) = size.split_once('x')
                .and_then(|(width, height)| Some((width.parse::<u64>().ok()?, height.parse::<u64>().ok()?)))
                .filter(|(width, height)| *width > 0 && *height > 0)
                .ok_or_else(|| ProxyError::Validation("size must be 'auto' or WIDTHxHEIGHT".to_string()))?;
            width = requested_width;
            height = requested_height;
        }
    }

    let seed = match request.get("seed") {
        Some(value) => value.as_u64()
            .ok_or_else(|| ProxyError::Validation("seed must be an unsigned integer".to_string()))?,
        None => rand::rng().random_range(0..1_000_000_000),
    };
    let quality = request_image_quality(request)?;
    let input_tokens = estimate_clip_tokens(&prompt)
        .saturating_add(estimate_clip_tokens(&negative_prompt));

    Ok(GenerationInput {
        model,
        prompt,
        negative_prompt,
        n,
        width,
        height,
        seed,
        usage: UsageEstimate {
            input_tokens,
            output_tokens_per_image: gpt_image2_output_tokens(width, height, quality),
        },
    })
}

fn build_query_string(params: &HashMap<String, String>) -> String {
    if params.is_empty() {
        return String::new();
    }
    let mut pairs = params.iter().collect::<Vec<_>>();
    pairs.sort_unstable_by(|left, right| left.0.cmp(right.0).then(left.1.cmp(right.1)));
    serde_urlencoded::to_string(pairs)
        .map(|query| format!("?{}", query))
        .unwrap_or_default()
}

fn json_response(value: &Value) -> Result<AxumResponse, ProxyError> {
    let body = serde_json::to_vec(value)
        .map_err(|error| ProxyError::Json(format!("Failed to serialize JSON response: {}", error)))?;
    AxumResponse::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("content-length", body.len())
        .body(Body::from(body))
        .map_err(|error| ProxyError::Internal(format!("Failed to build response: {}", error)))
}

/// Handles OpenAI image generation requests. SD-Turbo requests use the dynamic
/// batch worker; other workflows keep the existing one-request execution path.
pub async fn generations_response(
    State(state): State<Arc<ProxyState>>,
    Query(params): Query<HashMap<String, String>>,
    headers: HeaderMap,
    body: Body,
    req_id: &str,
) -> Result<AxumResponse, ProxyError> {
    let body = axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024)
        .await
        .map_err(|error| ProxyError::Internal(format!("Failed to read request body: {}", error)))?;
    let mut response = if request_model(&body)? == "IMG2.0-S" {
        batched_generations_response(state, params, headers, body, req_id).await
    } else {
        direct_generations_response(
            State(state), Query(params), headers, Body::from(body), req_id,
        ).await
    }?;
    if let Some(images) = response.extensions_mut().remove::<ResponseImages>() {
        let (parts, body) = response.into_parts();
        let remaining = body.size_hint().exact();
        return Ok(AxumResponse::from_parts(parts, Body::from_stream(ImageResponseStream {
            inner: body.into_data_stream(),
            images: images.take(),
            remaining,
            completed: false,
        })));
    }
    Ok(response)
}

async fn batched_generations_response(
    state: Arc<ProxyState>,
    params: HashMap<String, String>,
    headers: HeaderMap,
    body: Bytes,
    req_id: &str,
) -> Result<AxumResponse, ProxyError> {
    let input = parse_generation_input(&body, &state.workflows)?;
    let deadline = Instant::now() + Duration::from_secs(state.timeout);
    let key = BatchKey {
        model: input.model.clone(),
        width: input.width,
        height: input.height,
        authorization: headers.get("authorization")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string),
        query_string: build_query_string(&params),
    };
    let batch_queues = state.backend_pool.ready_queues();
    if batch_queues.is_empty() {
        return Err(ProxyError::Unavailable("No ready IMG2.0-S backend".to_string()));
    }
    let start = state.backend_pool.next_index(batch_queues.len());
    let queue_indexes = batch_queue_order(batch_queues.as_slice(), start);
    let (queue, capacity) = queue_indexes.into_iter().find_map(|index| {
        let queue = &batch_queues[index];
        if queue.sender.is_closed() {
            return None;
        }
        reserve_batch_capacity(&queue.capacity, input.n)
            .ok()
            .map(|capacity| (queue, capacity))
    }).ok_or_else(|| ProxyError::Busy("Dynamic batch queues are full".to_string()))?;
    let permits = queue.sender.try_reserve_many(input.n).map_err(|error| match error {
        mpsc::error::TrySendError::Full(_) => {
            ProxyError::Busy("Dynamic batch queue is full".to_string())
        }
        mpsc::error::TrySendError::Closed(_) => {
            ProxyError::Unavailable("Dynamic batch worker is not running".to_string())
        }
    })?;
    let mut receivers = Vec::with_capacity(input.n);

    for (offset, permit) in permits.enumerate() {
        let (reply, receiver) = oneshot::channel();
        receivers.push(receiver);
        permit.send(BatchSlot {
            key: key.clone(),
            request_id: req_id.to_string(),
            prompt: input.prompt.clone(),
            negative_prompt: input.negative_prompt.clone(),
            seed: input.seed,
            seed_offset: offset,
            deadline,
            _capacity: capacity.clone(),
            reply,
        });
    }

    let mut images = Vec::with_capacity(input.n);
    for receiver in receivers {
        let image = match timeout_at(deadline, receiver).await {
            Ok(Ok(result)) => result?,
            Ok(Err(_)) => return Err(ProxyError::Unavailable(
                "Dynamic batch worker stopped before returning a result".to_string(),
            )),
            Err(_) => return Err(ProxyError::Upstream(format!(
                "Image generation timed out after {} seconds", state.timeout
            ))),
        };
        images.push(image);
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ProxyError::Internal("Failed to get current time".to_string()))?
        .as_secs() as i64;
    let data = images.iter().map(|image| image.data.clone()).collect();
    let mut response = json_response(&build_image_response(data, created, &input.usage))?;
    response.extensions_mut().insert(ResponseImages::new(
        images.into_iter().map(|image| image.cleanup).collect()
    ));
    Ok(response)
}

async fn direct_generations_response(
    State(state): State<Arc<ProxyState>>,
    Query(params): Query<HashMap<String, String>>,
    headers: HeaderMap,
    body: Body,
    req_id: &str,
) -> Result<AxumResponse, ProxyError> {
    let body_bytes = axum::body::to_bytes(body, state.max_payload_size_mb * 1024 * 1024)
        .await
        .map_err(|error| ProxyError::Internal(format!("Failed to read request body: {}", error)))?;
    let request: Value = serde_json::from_slice(&body_bytes)
        .map_err(|error| ProxyError::Json(format!("Failed to parse JSON: {}", error)))?;
    let request = request.as_object()
        .ok_or_else(|| ProxyError::Json("Request body must be a JSON object".to_string()))?;
    let quality = request_image_quality(request)?;
    let processed_body = create_json_payload(
        body_bytes,
        state.workflows.clone(),
        state.backend_client_id.clone(),
        req_id,
    ).await?;
    let usage = estimate_usage(&processed_body, quality)?;
    let target_base = format!("{}:{}", state.backend_url, state.backend_port);
    let full_url = format!("http://{}/prompt{}", target_base, build_query_string(&params));
    submit_processed_body(&state, full_url, target_base, headers, processed_body, usage, req_id).await
}

async fn submit_processed_body(
    state: &ProxyState,
    full_url: String,
    target_base: String,
    headers: HeaderMap,
    processed_body: Bytes,
    usage: UsageEstimate,
    req_id: &str,
) -> Result<AxumResponse, ProxyError> {
    let mut upstream_headers = reqwest::header::HeaderMap::new();
    upstream_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_value) = reqwest::header::HeaderValue::from_bytes(auth.as_bytes()) {
            upstream_headers.insert(reqwest::header::AUTHORIZATION, auth_value);
        }
    }

    debug!("┃ [{}] 🚀 Requesting ComfyUI /prompt ({} bytes)", req_id, processed_body.len());
    let request = state.client
        .request(Method::POST, &full_url)
        .headers(upstream_headers)
        .body(processed_body)
        .send();
    let timeout_duration = Duration::from_secs(state.timeout);
    let upstream_response = match timeout(timeout_duration, request).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => return Err(handle_request_error(error, &full_url)),
        Err(_) => return Err(handle_timeout_error(&full_url, timeout_duration)),
    };
    if !upstream_response.status().is_success() {
        let status = upstream_response.status();
        let body = upstream_response.text().await.unwrap_or_default();
        return Err(ProxyError::Upstream(format!(
            "ComfyUI /prompt returned {}: {}", status, body
        )));
    }

    handle_regular_response(
        upstream_response,
        target_base,
        headers,
        Path::new(&state.output_directory),
        state.use_ws,
        &state.client,
        &state.ws_manager,
        usage,
        req_id,
    ).await
}

pub(crate) fn dynamic_batch_channel() -> (
    BatchSender,
    mpsc::Receiver<BatchSlot>,
    Arc<Semaphore>,
) {
    let (sender, receiver) = mpsc::channel(BATCH_QUEUE_CAPACITY);
    (sender, receiver, Arc::new(Semaphore::new(BATCH_QUEUE_CAPACITY)))
}

pub(crate) async fn warm_dynamic_batch_backend(
    state: &Arc<ProxyState>,
    queue: &BatchQueue,
) -> Result<(), ProxyError> {
    let workflow = state.workflows.get("IMG2.0-S")
        .ok_or_else(|| ProxyError::Validation("Workflow not found for model: IMG2.0-S".to_string()))?;
    let (width, height) = workflow_dimensions(workflow);
    let deadline = Instant::now() + Duration::from_secs(state.timeout);
    let capacity = reserve_batch_capacity(&queue.capacity, 1)?;
    let permit = queue.sender.try_reserve().map_err(|error| match error {
        mpsc::error::TrySendError::Full(_) => ProxyError::Busy("Dynamic batch queue is full".to_string()),
        mpsc::error::TrySendError::Closed(_) => ProxyError::Unavailable("Dynamic batch worker is not running".to_string()),
    })?;
    let (reply, receiver) = oneshot::channel();
    permit.send(BatchSlot {
        key: BatchKey {
            model: "IMG2.0-S".to_string(),
            width,
            height,
            authorization: None,
            query_string: String::new(),
        },
        request_id: format!("warmup-{}", uuid::Uuid::new_v4()),
        prompt: "warmup".to_string(),
        negative_prompt: workflow_negative_prompt(workflow),
        seed: 0,
        seed_offset: 0,
        deadline,
        _capacity: capacity,
        reply,
    });
    let image = match timeout_at(deadline, receiver).await {
        Ok(Ok(result)) => result?,
        Ok(Err(_)) => return Err(ProxyError::Unavailable(
            "Dynamic batch worker stopped during warmup".to_string(),
        )),
        Err(_) => return Err(ProxyError::Upstream(
            "IMG2.0-S backend warmup timed out".to_string(),
        )),
    };
    drop(image);
    Ok(())
}

pub(crate) async fn run_dynamic_batch_worker(
    state: Weak<ProxyState>,
    mut receiver: mpsc::Receiver<BatchSlot>,
) {
    let mut pending = VecDeque::new();

    loop {
        let first = loop {
            let slot = match pending.pop_front() {
                Some(slot) => Some(slot),
                None => receiver.recv().await,
            };
            match slot {
                Some(slot) if slot.reply.is_closed() => continue,
                Some(slot) => break slot,
                None => return,
            }
        };
        let batch = collect_batch(
            first,
            &mut receiver,
            &mut pending,
            MAX_BATCH_SIZE,
            Duration::from_millis(BATCH_WAIT_MS),
        ).await;
        let now = Instant::now();
        let (expired, batch): (Vec<_>, Vec<_>) = batch.into_iter()
            .filter(|slot| !slot.reply.is_closed())
            .partition(|slot| slot.deadline <= now);
        for slot in expired {
            let _ = slot.reply.send(Err(ProxyError::Upstream(
                "Image generation timed out while waiting in the batch queue".to_string(),
            )));
        }
        if batch.is_empty() {
            continue;
        }

        let Some(state) = state.upgrade() else {
            let error = ProxyError::Unavailable("API server is shutting down".to_string());
            for slot in batch {
                let _ = slot.reply.send(Err(error.clone()));
            }
            return;
        };
        let batch_id = uuid::Uuid::new_v4().to_string();
        info!(
            "🧺 Dynamic batch {}: {} slot(s), requests [{}]",
            batch_id,
            batch.len(),
            batch.iter().map(|slot| slot.request_id.as_str()).collect::<Vec<_>>().join(",")
        );

        let deadline = batch.iter().map(|slot| slot.deadline).min().unwrap();
        let result = match timeout_at(deadline, execute_dynamic_batch(&state, &batch, &batch_id)).await {
            Ok(result) => result,
            Err(_) => Err(ProxyError::Upstream(format!(
                "Dynamic batch {} timed out before completion", batch_id
            ))),
        };
        match result {
            Ok(images) if images.len() == batch.len() => {
                for (slot, image) in batch.into_iter().zip(images) {
                    let _ = slot.reply.send(Ok(image));
                }
            }
            Ok(images) => {
                let error = ProxyError::Upstream(format!(
                    "ComfyUI returned {} image(s) for {} batch slot(s)",
                    images.len(), batch.len()
                ));
                for slot in batch {
                    let _ = slot.reply.send(Err(error.clone()));
                }
            }
            Err(error) => {
                for slot in batch {
                    let _ = slot.reply.send(Err(error.clone()));
                }
            }
        }
    }
}

async fn execute_dynamic_batch(
    state: &ProxyState,
    batch: &[BatchSlot],
    batch_id: &str,
) -> Result<Vec<PendingImage>, ProxyError> {
    let payload = create_batch_json_payload(
        batch,
        &state.workflows,
        &state.backend_client_id,
        batch_id,
    )?;
    let first = &batch[0];
    let mut headers = HeaderMap::new();
    if let Some(authorization) = &first.key.authorization {
        headers.insert(
            "authorization",
            HeaderValue::from_str(authorization)
                .map_err(|error| ProxyError::Validation(format!("Invalid authorization header: {}", error)))?,
        );
    }
    let target_base = format!("{}:{}", state.backend_url, state.backend_port);
    let full_url = format!("http://{}/prompt{}", target_base, first.key.query_string);
    let usage = UsageEstimate {
        input_tokens: batch.iter().map(|slot| {
            estimate_clip_tokens(&slot.prompt).saturating_add(estimate_clip_tokens(&slot.negative_prompt))
        }).sum(),
        // This response is internal; the batch splitter returns each request's own usage.
        output_tokens_per_image: gpt_image2_output_tokens(
            first.key.width,
            first.key.height,
            ImageQuality::Low,
        ),
    };
    let mut response = submit_processed_body(
        state,
        full_url,
        target_base,
        headers,
        payload,
        usage,
        batch_id,
    ).await?;
    let protection = response.extensions_mut().remove::<ResponseImages>().unwrap_or_default();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), MAX_BATCH_SIZE * 16 * 1024 * 1024)
        .await
        .map_err(|error| ProxyError::Internal(format!("Failed to read batch response: {}", error)))?;
    if !status.is_success() {
        return Err(ProxyError::Upstream(format!("Batch response returned {}", status)));
    }
    let json: Value = serde_json::from_slice(&body)
        .map_err(|error| ProxyError::Json(format!("Failed to parse batch response: {}", error)))?;
    let data = json.get("data").and_then(Value::as_array)
        .ok_or_else(|| ProxyError::Upstream("Batch response did not contain image data".to_string()))?;
    let images = data.iter().map(|image| {
        PendingImage::new(image.clone(), Path::new(&state.output_directory))
    }).collect::<Result<Vec<_>, _>>()?;
    for guard in protection.take() {
        guard.disarm();
    }
    Ok(images)
}


// =================================================================================
//  Section: Payload Transformation
// =================================================================================

/// Transforms OpenAI API request into ComfyUI prompt request
///
/// This function executes the key translation layer between OpenAI image generation specs
/// and ComfyUI workflows.
///
/// # Mapping Logic
/// - `model` (OpenAI) -> Workflow Name (ComfyUI) - Used to look up specific workflow JSON
/// - `prompt` (OpenAI) -> Positive Prompt (ComfyUI node input)
/// - `negative_prompt` (OpenAI Extension) -> Negative Prompt
/// - `seed` (OpenAI) -> Random Seed (KSampler node)
/// - `size` (OpenAI) -> Image Size (EmptyLatentImage node)
/// - `n` (OpenAI) -> Batch Size / Number of images
///
/// # Arguments
/// * `body` - Original request body from OpenAI API call (Bytes)
/// * `workflows` - Map of available ComfyUI workflow definitions
/// * `client_id` - Client ID used for WebSocket pairing
///
/// # Returns
/// - Serialized ComfyUI prompt JSON (Bytes), ready to be submitted to backend
/// - `ProxyError`: If validation fails or transformation errors occur
async fn create_json_payload(
    body: Bytes,
    workflows: Arc<HashMap<String, Value>>,
    client_id: String,
    req_id: &str,
) -> Result<Bytes, ProxyError> {
    // Return empty if body is empty
    if body.is_empty() {
        return Ok(body);
    }
    let mut rng = rand::rng();

    // Parse input OpenAI format JSON
    let json: Value = serde_json::from_slice(&body)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    // Initialize ComfyUI request structure
    let mut workflow_use = serde_json::json!({
            "prompt": "",
            "client_id": ""
        });

    // Process OpenAI request fields
    if let Some(openai_request) = json.as_object() {
        // OpenAI allows model to be omitted or null; this service defaults to IMG2.0-S.
        let model_name = openai_request.get("model")
            .and_then(|value| value.as_str())
            .unwrap_or("IMG2.0-S");
        if let Some(workflow) = workflows.get(model_name) {
            // debug!("┃ [{}] 📦 Retrieved workflow '{}'", req_id, model_name); // Reduce noise
            info!("┃ [STEP 2/5] 🧩 Loading Workflow: '{}'", model_name);
            // Use selected workflow as the base prompt graph
            workflow_use["prompt"] = workflow.clone();
        } else {
            return Err(ProxyError::Json(format!("Workflow '{}' not found", model_name)));
        }

        // Modify parameters in the workflow
        if let Some(obj) = workflow_use.as_object_mut() {
            // Set Client ID for WebSocket pairing
            obj.insert(
                "client_id".to_string(),
                Value::String(client_id.clone()),
            );

            // Iterate and modify workflow nodes, injecting OpenAI request parameters
            if let Some(workflow_prompt) = workflow_use.get_mut("prompt").and_then(|v| v.as_object_mut()){
                for (_node_id, node_data) in workflow_prompt {
                    if let Some(class_type_ref) = node_data["class_type"].as_str() {
                        let class_type = class_type_ref.to_string();
                        match class_type.as_str() {
                            // Handle Random Seed
                            "KSampler" | "SamplerCustom" | "easy seed" => {
                                if let Some(inputs_data_sampler) = node_data["inputs"].as_object_mut() {
                                    let seed_key = if class_type == "SamplerCustom" { "noise_seed" } else { "seed" };
                                    // Inject Seed
                                    if let Some(seed_data) = openai_request.get("seed").and_then(|v| v.as_i64()) {
                                        debug!("┃ [{}] ✏️ Requested seed: {}", req_id, seed_data);
                                        inputs_data_sampler.insert(
                                            seed_key.to_string(),
                                            serde_json::json!(seed_data),
                                        );
                                    } else {
                                        // Generate random seed if not requested
                                        let random_number: u64 = rng.random_range(0..1_000_000_000);
                                        debug!("┃ [{}] No seed in JSON, using random seed: {}", req_id, random_number);
                                        inputs_data_sampler.insert(
                                            seed_key.to_string(),
                                            serde_json::json!(random_number),
                                        );
                                    }
                                }
                            }
                            // Handle Image Size and Batch Size
                            "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage" => {
                                if let Some(inputs_data_size) = node_data["inputs"].as_object_mut() {
                                    // Parse WIDTHxHEIGHT; "auto" keeps the workflow defaults.
                                    if let Some(size_data) = openai_request.get("size").and_then(|v| v.as_str()) {
                                        debug!("┃ [{}] ✏️ Requested image size: {}", req_id, size_data);
                                        if let Some((width, height)) = size_data.split_once('x')
                                            .and_then(|(width, height)| Some((width.parse::<i64>().ok()?, height.parse::<i64>().ok()?)))
                                        {
                                            inputs_data_size.insert(
                                                "width".to_string(),
                                                serde_json::json!(width),
                                            );
                                            inputs_data_size.insert(
                                                "height".to_string(),
                                                serde_json::json!(height),
                                            );
                                        }
                                    }

                                    // Set batch_size (number of images)
                                    if let Some(copies_num_data) = openai_request.get("n").and_then(|v| v.as_i64()) {
                                        debug!("┃ [{}] ✏️ Requested n (batch_size): {}", req_id, copies_num_data);
                                        inputs_data_size.insert(
                                            "batch_size".to_string(),
                                            serde_json::json!(copies_num_data),
                                        );
                                    }
                                }
                            }
                            // Handle Text Prompts
                            "CLIPTextEncode" | "CR Text" | "easy promptLine" => {
                                // Check if it's Positive or Negative prompt node
                                if let Some(meta_data) = node_data["_meta"].as_object() {
                                    if let Some(title) = meta_data["title"].as_str() {
                                        // Inject Positive Prompt
                                        // Matches common node titles for positive prompts (English & Chinese)
                                        if title == "Positive Prompt" || title == "CLIP文本编码" || title == "🔤 CR Text" || title == "提示词行" {
                                            if let Some(inputs_data) = node_data["inputs"].as_object_mut() {
                                                if let Some(prompt_input) = openai_request.get("prompt").and_then(|v| v.as_str()) {
                                                    debug!("┃ [{}] ✏️ Requested prompt: {}", req_id, prompt_input);

                                                    // Determine field name based on class_type
                                                    let key = match class_type.as_str() {
                                                        "easy promptLine" => "prompt",
                                                        _ => "text",
                                                    };

                                                    inputs_data.insert(
                                                        key.to_string(),
                                                        Value::String(prompt_input.to_string()),
                                                    );
                                                } else {
                                                    return Err(ProxyError::Json(format!("Failed to get prompt from JSON")));
                                                }
                                            }
                                        }
                                        // Inject Negative Prompt (Optional)
                                        // Matches common node titles for negative prompts
                                        else if title == "Negative Prompt" || title == "条件零化" {
                                            if let Some(inputs_data) = node_data["inputs"].as_object_mut() {
                                                if let Some(neg_prompt_input) = openai_request.get("negative_prompt").and_then(|v| v.as_str()) {
                                                    debug!("┃ [{}] ✏️ Requested negative prompt: {}", req_id, neg_prompt_input);
                                                    inputs_data.insert(
                                                        "text".to_string(),
                                                        Value::String(neg_prompt_input.to_string()),
                                                    );
                                                } else {
                                                    debug!("┃ [{}] No negative_prompt in JSON, using workflow default", req_id);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // Make SaveImage execute for every request, even when upstream nodes are cached.
                            "SaveImage" => {
                                if let Some(inputs) = node_data["inputs"].as_object_mut() {
                                    let prefix = inputs.get("filename_prefix")
                                        .and_then(|value| value.as_str())
                                        .unwrap_or("ComfyUI");
                                    inputs.insert(
                                        "filename_prefix".to_string(),
                                        Value::String(format!("{}_{}", prefix, req_id)),
                                    );
                                }
                            }
                            // Skip irrelevant nodes
                            _ => {
                                continue
                            }
                        }
                    }
                }
            }
        }

        // info!("┃ [{}] 🔧 JSON payload generated", req_id);
    }

    // Serialize modified ComfyUI prompt to JSON bytes
    let modified_json = serde_json::to_vec(&workflow_use)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;

    Ok(Bytes::from(modified_json))
}

// =================================================================================
//  Section: Response Handling & Image Retrieval
// =================================================================================

/// Handles the response from ComfyUI backend
///
/// This function:
/// 1. Extracts the `prompt_id` from the backend response.
/// 2. Waits for job completion via WebSocket.
/// 3. Retrieves generated image metadata from the backend.
/// 4. Returns one-time image URLs in OpenAI API format.
async fn handle_regular_response(
    upstream_response: reqwest::Response,
    target_base: String,
    request_headers: HeaderMap,
    output_directory: &Path,
    use_ws: bool,
    client: &Client,
    ws_manager: &Option<Arc<WebSocketManager>>,
    usage: UsageEstimate,
    req_id: &str,
) -> Result<AxumResponse, ProxyError> {
    let status = upstream_response.status();
    let response_headers = upstream_response.headers().clone();

    debug!("┃ [{}] 📄 Handling regular response with status: {}", req_id, status);

    // Read response body from backend
    let body_bytes = upstream_response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read response body: {}", e)))?;

    debug!("┃ [{}] 📥 Response body: {} bytes", req_id, body_bytes.len());

    // Parse backend response as JSON
    let json: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ProxyError::Json(format!("Failed to parse JSON: {}", e)))?;

    debug!(
            "┃ [{}] 📝 Received response: {}",
            req_id,
            json.to_string()
        );

    // Extract the prompt_id which identifies this job in ComfyUI
    let prompt_id = json.get("prompt_id").and_then(|v| v.as_str());
    if let Some(pid) = prompt_id {
        debug!(
            "┃ [{}] 📝 Found prompt_id in response: {}. Waiting for job completion...",
            req_id,
            pid
        );
        info!("┃ [STEP 3/5] ⏳ Waiting for Job Completion (ID: {})", pid);

        // Block until the job completes via WebSocket
        if use_ws {
            if let Some(manager) = ws_manager {
                let authorization = request_headers.get("authorization")
                    .and_then(|value| value.to_str().ok());
                match timeout(
                    Duration::from_secs(600),
                    manager.wait_for_job_completion(pid, authorization),
                ).await {
                    Ok(Ok(())) => {},
                    Ok(Err(error)) => return Err(ProxyError::Upstream(format!(
                        "Failed to wait for ComfyUI job {}: {}", pid, error
                    ))),
                    Err(_) => return Err(ProxyError::Upstream(format!(
                        "ComfyUI job {} timed out after 600 seconds", pid
                    ))),
                }
            } else {
                warn!("┃ [{}] ⚠️ WebSocket manager is not initialized but use_ws is true", req_id);
            }
        } else {
            loop {
                // Poll the queue until the job is finished (not in queue)
                let is_done = check_queue(
                    target_base.clone(),
                    Some(pid),
                    request_headers.clone(),
                    client,
                    req_id,
                )
                .await?;

                if is_done {
                    debug!("┃ [{}] ⚡ Job {} completed (not found in queue)", req_id, pid);
                    break
                }

                tokio::time::sleep(Duration::from_millis(2000)).await;
            }
        }
    }

    // Fetch generated images from ComfyUI backend and prepare response
    let (image_response_json, cleanup) = retrieve_image_from_history(
        target_base,
        prompt_id,
        request_headers.clone(),
        client,
        output_directory,
        &usage,
        req_id,
    )
    .await?;

    // Serialize the image response JSON to bytes
    let output_json = serde_json::to_vec(&image_response_json)
        .map_err(|e| ProxyError::Json(format!("Failed to serialize JSON: {}", e)))?;
    let output_body_bytes = Bytes::from(output_json);
    debug!(
        "┃ [{}] ✏️ JSON regular response: {} bytes",
        req_id,
        output_body_bytes.len()
    );

    // Copy and update response headers
    let mut headers: HeaderMap<HeaderValue> = HeaderMap::with_capacity(response_headers.len());
    for (name, value) in response_headers.iter() {
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_str(name.as_str()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            if name.as_str() == "content-length" {
                // Update content-length to match the final response size
                if let Ok(value) =
                    HeaderValue::from_str(format!("{}", output_body_bytes.len()).as_str())
                {
                    headers.insert(name, value);
                }
            } else {
                headers.insert(name, value);
            }
        }
    }

    // Build the final HTTP response
    let mut response = AxumResponse::builder().status(status.as_u16());

    for (name, value) in headers.iter() {
        response = response.header(name, value);
    }

    debug!("┃ [{}] ✅ Regular response built successfully", req_id);

    let mut response = response
        .body(Body::from(output_body_bytes))
        .map_err(|e| ProxyError::Internal(format!("Failed to build regular response: {}", e)))?;
    response.extensions_mut().insert(ResponseImages::new(vec![cleanup]));
    Ok(response)
}

/// Retrieves generated images from ComfyUI backend
///
/// This function:
/// 1. Queries the ComfyUI history endpoint for the given prompt_id.
/// 2. Extracts image metadata from the job outputs.
/// 3. Builds one-time public URLs for the generated images.
/// 4. Formats the response in OpenAI API standards.
///
/// # Arguments
/// * `target_base` - ComfyUI backend address (host:port)
/// * `prompt_id` - The job ID to retrieve results for
/// * `headers` - Original request headers (may contain auth)
/// * `client` - HTTP client for backend communication
///
/// # Returns
/// - JSON response in OpenAI image generation format with image URLs.
/// - `ProxyError` if history lookup or image retrieval fails.
async fn retrieve_image_from_history(
    target_base: String,
    prompt_id: Option<&str>,
    headers: HeaderMap,
    client: &Client,
    output_directory: &Path,
    usage: &UsageEstimate,
    req_id: &str,
) -> Result<(Value, ImageCleanupGuard), ProxyError> {

    // Validate that we have a prompt_id
    let prompt_id = match prompt_id {
        Some(id) => id,
        None => {
            error!("⚠️ No prompt_id received!");
            return Err(ProxyError::Upstream(format!(
                    "No prompt_id received.",
                )));
        }
    };

    // Construct URL to ComfyUI history endpoint
    let history_url: String = format!("http://{}/history/{}", target_base, prompt_id);

    info!("┃ [STEP 4/5] 🔍 Checking History/Validation (ID: {})", prompt_id);
    debug!("┃ [{}] 🔍 Checking history at {} for {}", req_id, target_base, prompt_id);

    // Prepare headers for backend requests
    let mut upstream_headers = reqwest::header::HeaderMap::new();

    // Forward authorization headers if present
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_value) = reqwest::header::HeaderValue::from_bytes(auth.as_bytes()) {
            upstream_headers.insert(reqwest::header::AUTHORIZATION, auth_value);
        }
    }

    // Build request to history endpoint
    let request_builder = client
        .request(Method::GET, &history_url)
        .headers(upstream_headers.clone());

    // Query history with timeout protection
    let request_future = request_builder.send();
    let timeout_duration = Duration::from_secs(5);

    let upstream_response = match tokio::time::timeout(timeout_duration, request_future).await {
        Ok(Ok(response)) => {
            debug!(
                "┃ [{}] ✅ Got response from history backend: {} - Headers: {:?}",
                req_id,
                response.status(),
                response.headers()
            );
            response
        }
        Ok(Err(e)) => {
            return Err(handle_request_error(e, &history_url));
        }
        Err(_) => {
            return Err(handle_timeout_error(&history_url, timeout_duration));
        }
    };

    if !upstream_response.status().is_success() {
        return Err(ProxyError::Upstream(format!(
            "ComfyUI history returned status {} for {}",
            upstream_response.status(), prompt_id
        )));
    }

    // Parse history response
    let response_body = upstream_response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read history response body: {}", e)))?;
    let history_json: Value = serde_json::from_slice(&response_body)
        .map_err(|e| ProxyError::Json(format!("Failed to parse history JSON: {}", e)))?;

    // Extract image metadata from job outputs
    let mut image_files: Vec<ImageFile> = Vec::new();
    if let Some(prompt_hist) = history_json.get(prompt_id).and_then(|v| v.as_object()) {
        let status = prompt_hist.get("status").and_then(Value::as_object);
        let completed = status.and_then(|value| value.get("completed")).and_then(Value::as_bool);
        let status_str = status.and_then(|value| value.get("status_str")).and_then(Value::as_str);
        if completed != Some(true) || status_str != Some("success") {
            return Err(ProxyError::Upstream(format!(
                "ComfyUI job {} failed or did not complete: {}",
                prompt_id,
                prompt_hist.get("status").unwrap_or(&Value::Null)
            )));
        }
        if let Some(out_nodes) = prompt_hist.get("outputs").and_then(|v| v.as_object())
        {
            // Iterate through output nodes looking for generated images
            for (_node_id, out_node_data) in out_nodes {
                if let Some(all_images_data) = out_node_data.get("images").and_then(|v| v.as_array())
                {
                    // Process each image in the node output
                    for image_data in all_images_data {
                        if let Some(type_field) = image_data["type"].as_str() {
                            // Only include output images (not temporary/preview)
                            if type_field == "output" {
                                if let Some(filename) = image_data["filename"].as_str() {
                                    let subfolder = image_data["subfolder"].as_str().unwrap_or("");
                                    debug!("┃ [{}] Found image: {} (subfolder: {})", req_id, filename, subfolder);
                                    image_files.push(ImageFile {
                                        filename: filename.to_string(),
                                        subfolder: subfolder.to_string(),
                                        type_field: type_field.to_string()
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        error!("┃ [{}] ⚠️ No prompt_id history found", req_id);
        return Err(ProxyError::Upstream(format!(
                "No prompt_id history found.",
            )));
    }

    debug!("┃ [{}] 📦 Collected {} image files", req_id, image_files.len());
    if image_files.is_empty() {
        return Err(ProxyError::Upstream(format!(
            "ComfyUI job {} completed without output images", prompt_id
        )));
    }
    info!("┃ [STEP 5/5] 🔗 Preparing One-Time Image URLs");
    let (response_data, cleanup_guard) = prepare_image_urls(output_directory, &image_files)?;

    // Create OpenAI API format response with timestamp
    let created = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_err(|_| ProxyError::Internal("Failed to get current time".to_string()))?
    .as_secs() as i64;

    let response = build_image_response(response_data, created, usage);
    Ok((response, cleanup_guard))

}

// A queued result owns its file until the HTTP response has been assembled.
// Dropped receivers, partial n-image failures, and rejected batches clean up automatically.
struct PendingImage {
    data: Value,
    cleanup: ImageCleanupGuard,
}

impl PendingImage {
    fn new(data: Value, output_directory: &Path) -> Result<Self, ProxyError> {
        let query = data.get("url").and_then(Value::as_str)
            .and_then(|url| url.split_once('?').map(|(_, query)| query))
            .ok_or_else(|| ProxyError::Internal("Batch image did not contain a URL".to_string()))?;
        let image: ImageFile = serde_urlencoded::from_str(query)
            .map_err(|error| ProxyError::Internal(format!("Invalid batch image URL: {}", error)))?;
        Ok(Self { data, cleanup: ImageCleanupGuard::new(output_directory, &[image]) })
    }
}

// Only active response files are tracked. On process restart, orphan files fall
// back to the persisted mtime-based sweep; no permanent "pending" files remain.
// Protection stays live during mtime reset; registration and sweep/delete share a lock.
// ponytail: process-local registry; use shared leases if multiple API processes share output.
static PROTECTED_IMAGES: LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct ImageCleanupGuard {
    output_directory: PathBuf,
    images: Vec<ImageFile>,
    protected_paths: Vec<PathBuf>,
}

impl ImageCleanupGuard {
    fn new(output_directory: &Path, images: &[ImageFile]) -> Self {
        let mut guard = Self {
            output_directory: output_directory.to_path_buf(),
            images: Vec::new(),
            protected_paths: Vec::new(),
        };
        for image in images {
            guard.protect(image.clone());
        }
        guard
    }

    fn protect(&mut self, image: ImageFile) {
        if let Some(path) = generated_image_path(&self.output_directory, &image) {
            let mut protected = PROTECTED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
            *protected.entry(path.clone()).or_default() += 1;
            self.protected_paths.push(path);
        }
        self.images.push(image);
    }

    fn disarm(mut self) {
        self.images.clear();
    }

    fn delivered(self) {
        // Keep protection until every timestamp has been reset, so a concurrent
        // sweep cannot delete a just-delivered image using its generation time.
        let now = SystemTime::now();
        for path in &self.protected_paths {
            if let Err(error) = fs::File::open(path).and_then(|file| file.set_modified(now))
                && error.kind() != std::io::ErrorKind::NotFound
            {
                warn!("Failed to reset delivered image retention {}: {}", path.display(), error);
            }
        }
        self.disarm();
    }
}

impl Drop for ImageCleanupGuard {
    fn drop(&mut self) {
        let mut protected = PROTECTED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
        remove_generated_images(&self.output_directory, &self.images);
        for path in &self.protected_paths {
            if let Some(count) = protected.get_mut(path) {
                *count -= 1;
                if *count == 0 {
                    protected.remove(path);
                }
            }
        }
    }
}

/// Transfer cleanup ownership through internal batch responses without starting TTL.
#[derive(Clone, Default)]
struct ResponseImages(Arc<Mutex<Vec<ImageCleanupGuard>>>);

impl ResponseImages {
    fn new(images: Vec<ImageCleanupGuard>) -> Self {
        Self(Arc::new(Mutex::new(images)))
    }

    fn take(&self) -> Vec<ImageCleanupGuard> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// The JSON response owns its files throughout delay and body transmission.
/// This tracks handoff to the HTTP transport, not acknowledgement by the client.
struct ImageResponseStream {
    inner: BodyDataStream,
    images: Vec<ImageCleanupGuard>,
    remaining: Option<u64>,
    completed: bool,
}

impl Stream for ImageResponseStream {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = Pin::new(&mut self.inner).poll_next(cx);
        match &result {
            Poll::Ready(Some(Ok(bytes))) => {
                if let Some(remaining) = &mut self.remaining {
                    *remaining = remaining.saturating_sub(bytes.len() as u64);
                    // Hyper may not poll EOF after receiving Content-Length bytes.
                    self.completed = *remaining == 0;
                }
            }
            Poll::Ready(None) => self.completed = self.remaining.is_none_or(|n| n == 0),
            Poll::Ready(Some(Err(_))) => self.completed = false,
            Poll::Pending => {}
        }
        result
    }
}

impl Drop for ImageResponseStream {
    fn drop(&mut self) {
        if self.completed {
            for guard in self.images.drain(..) {
                guard.delivered();
            }
        }
        // On cancellation/error, the remaining guards immediately remove files.
    }
}

fn generated_image_path(output_directory: &Path, image: &ImageFile) -> Option<PathBuf> {
    if image.type_field != "output" {
        return None;
    }

    let filename = Path::new(&image.filename);
    if !matches!(filename.components().collect::<Vec<_>>().as_slice(), [Component::Normal(_)]) {
        return None;
    }

    let subfolder = Path::new(&image.subfolder);
    if subfolder.components().any(|component| {
        !matches!(component, Component::Normal(_) | Component::CurDir)
    }) {
        return None;
    }

    let mut directory = output_directory.to_path_buf();
    for component in subfolder.components() {
        directory.push(component.as_os_str());
        if fs::symlink_metadata(&directory).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return None;
        }
    }
    Some(directory.join(filename))
}

fn image_content_type(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

fn prepare_image_urls(
    output_directory: &Path,
    images: &[ImageFile],
) -> Result<(Vec<Value>, ImageCleanupGuard), ProxyError> {
    let mut cleanup = ImageCleanupGuard::new(output_directory, images);
    let mut data = Vec::with_capacity(images.len());
    for image in images {
        let path = generated_image_path(output_directory, image)
            .ok_or_else(|| ProxyError::Validation("Invalid generated image path".to_string()))?;
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| ProxyError::Upstream(format!("Generated image is unavailable: {}", error)))?;
        if !metadata.file_type().is_file() || image_content_type(&path).is_none() {
            return Err(ProxyError::Validation("Expected a regular image file".to_string()));
        }
        // Each image gets an independent capability, not a predictable batch sibling filename.
        let published = ImageFile {
            filename: format!("image-{}.{}", uuid::Uuid::new_v4(), path.extension().unwrap().to_str().unwrap()),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };
        cleanup.protect(published.clone());
        fs::rename(&path, output_directory.join(&published.filename))
            .map_err(|error| ProxyError::Internal(format!("Failed to publish generated image: {}", error)))?;
        let query = serde_urlencoded::to_string(&published)
            .map_err(|error| ProxyError::Json(format!("Failed to serialize image URL: {}", error)))?;
        data.push(serde_json::json!({
            "url": format!("{}/v1/images/files?{}", PUBLIC_IMAGE_ORIGIN, query)
        }));
    }
    Ok((data, cleanup))
}

async fn take_generated_image(
    output_directory: &Path,
    image: &ImageFile,
) -> Result<(Bytes, &'static str), ProxyError> {
    let path = generated_image_path(output_directory, image)
        .ok_or_else(|| ProxyError::Validation("Invalid generated image path".to_string()))?;
    let content_type = image_content_type(&path)
        .ok_or_else(|| ProxyError::Validation("Unsupported image file type".to_string()))?;
    let token = path.file_stem().and_then(|value| value.to_str())
        .and_then(|value| value.strip_prefix("image-"))
        .and_then(|value| uuid::Uuid::parse_str(value).ok());
    if token.is_none() || !image.subfolder.is_empty() {
        return Err(ProxyError::NotFound("Image URL has expired or was already used".to_string()));
    }
    let output_directory = output_directory.to_path_buf();
    let serving_image = ImageFile {
        filename: format!(".serving-{}.{}", uuid::Uuid::new_v4(), path.extension().unwrap().to_str().unwrap()),
        subfolder: String::new(),
        type_field: "output".to_string(),
    };

    // Keep claim/read/delete in one task so HTTP cancellation cannot skip cleanup.
    tokio::task::spawn_blocking(move || {
        let file_error = |error: std::io::Error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ProxyError::NotFound("Image URL has expired or was already used".to_string())
            } else {
                ProxyError::Internal(format!("Failed to access generated image: {}", error))
            }
        };
        if !fs::symlink_metadata(&path).map_err(file_error)?.file_type().is_file() {
            return Err(ProxyError::Validation("Expected a regular image file".to_string()));
        }
        let serving_path = output_directory.join(&serving_image.filename);
        // Atomic rename allows exactly one concurrent GET to claim the image.
        fs::rename(&path, &serving_path).map_err(file_error)?;
        let _cleanup = ImageCleanupGuard::new(&output_directory, &[serving_image]);
        if !fs::symlink_metadata(&serving_path).map_err(file_error)?.file_type().is_file() {
            return Err(ProxyError::Validation("Expected a regular image file".to_string()));
        }
        let bytes = fs::read(&serving_path).map_err(file_error)?;
        match fs::remove_file(&serving_path) {
            Ok(()) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Err(error) => return Err(file_error(error)),
        }
        Ok((Bytes::from(bytes), content_type))
    }).await.map_err(|error| ProxyError::Internal(format!("Image download task failed: {}", error)))?
}

pub async fn image_file_response(
    output_directory: &str,
    mut params: HashMap<String, String>,
) -> Result<AxumResponse, ProxyError> {
    let image = ImageFile {
        filename: params.remove("filename").unwrap_or_default(),
        subfolder: params.remove("subfolder").unwrap_or_default(),
        type_field: params.remove("type").unwrap_or_default(),
    };
    let (bytes, content_type) = take_generated_image(Path::new(output_directory), &image).await?;
    let mut response = AxumResponse::new(Body::from(bytes));
    response.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    Ok(response)
}

fn remove_generated_images(output_directory: &Path, images: &[ImageFile]) -> usize {
    let mut removed = 0;

    for image in images {
        let Some(path) = generated_image_path(output_directory, image) else {
            warn!("Skipping unsafe or non-output image path: {:?}", image);
            continue;
        };

        match fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => warn!("Failed to remove generated image {}: {}", path.display(), error),
        }
    }

    removed
}

pub(crate) fn cleanup_stale_images_before(output_directory: &Path, cutoff: SystemTime) -> usize {
    let entries = match fs::read_dir(output_directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return 0,
        Err(error) => {
            warn!("Failed to scan image output directory {}: {}", output_directory.display(), error);
            return 0;
        }
    };

    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                warn!("Failed to inspect cleanup path {}: {}", path.display(), error);
                continue;
            }
        };

        if file_type.is_dir() {
            removed += cleanup_stale_images_before(&path, cutoff);
            continue;
        }

        let is_image = path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| ["png", "jpg", "jpeg", "webp", "gif"].iter()
                .any(|candidate| extension.eq_ignore_ascii_case(candidate)))
            .unwrap_or(false);
        if !file_type.is_file() || !is_image {
            continue;
        }

        let protected = PROTECTED_IMAGES.lock().unwrap_or_else(|e| e.into_inner());
        if protected.contains_key(&path) {
            continue;
        }
        let is_stale = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .map(|modified| modified < cutoff)
            .unwrap_or(false);
        if !is_stale {
            continue;
        }

        match fs::remove_file(&path) {
            Ok(()) => removed += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => warn!("Failed to remove stale image {}: {}", path.display(), error),
        }
    }

    removed
}

fn cleanup_stale_images(output_directory: &Path, max_age: Duration) -> usize {
    let cutoff = SystemTime::now()
        .checked_sub(max_age)
        .unwrap_or(UNIX_EPOCH);
    cleanup_stale_images_before(output_directory, cutoff)
}

pub async fn run_image_cleanup_loop(
    output_directory: String,
    interval_seconds: u64,
    max_age_seconds: u64,
) {
    let output_directory = PathBuf::from(output_directory);
    let interval = Duration::from_secs(interval_seconds.max(1));
    let max_age = Duration::from_secs(max_age_seconds.max(1));

    loop {
        let removed = cleanup_stale_images(&output_directory, max_age);
        if removed > 0 {
            info!(
                "🧹 Periodic image cleanup removed {} stale file(s) from {}",
                removed,
                output_directory.display()
            );
        }
        tokio::time::sleep(interval).await;
    }
}

// =================================================================================
//  Section: Queue Management
// =================================================================================

/// Checks a job state from ComfyUI backend
///
/// This function:
/// 1. Queries the ComfyUI queue endpoint for the given prompt_id.
/// 2. Looks for the given job ID in both queues (pending and running).
/// 3. Returns true if not found (implying the job is finished).
///
/// # Arguments
/// * `target_base` - ComfyUI backend address (host:port)
/// * `prompt_id` - The job ID to check
/// * `headers` - Original request headers (may contain auth)
/// * `client` - HTTP client for backend communication
///
/// # Returns
/// - `true` if the given job id is NOT in the queues (hence finished).
/// - `ProxyError` if queue lookup fails.
async fn check_queue(
    target_base: String,
    prompt_id: Option<&str>,
    headers: HeaderMap,
    client: &Client,
    req_id: &str,
) -> Result<bool, ProxyError> {

    // Validate that we have a prompt_id
    let prompt_id = match prompt_id {
        Some(id) => id,
        None => {
            error!("⚠️ No prompt_id received!");
            return Err(ProxyError::Upstream(format!(
                    "No prompt_id received.",
                )));
        }
    };

    // Construct URL to ComfyUI history endpoint
    let history_url: String = format!("http://{}/queue", target_base);

    debug!("┃ [{}] 🔍 Checking queue at {}", req_id, target_base);

    // Prepare headers for backend requests
    let mut upstream_headers = reqwest::header::HeaderMap::new();

    // Forward authorization headers if present
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_value) = reqwest::header::HeaderValue::from_bytes(auth.as_bytes()) {
            upstream_headers.insert(reqwest::header::AUTHORIZATION, auth_value);
        }
    }

    // Log headers for debugging
    debug!("┃ [{}] 📋 Headers to send (if any):", req_id);
    for (name, value) in upstream_headers.iter() {
        debug!("┃ [{}]    {}: {}", req_id, name, value.to_str().unwrap_or("[unprintable]"));
    }

    // Build request to history endpoint
    let request_builder = client
        .request(Method::GET, &history_url)
        .headers(upstream_headers.clone());

    debug!("┃ [{}] ⏳ Sending queue request to backend...", req_id);

    // Query history with timeout protection
    let request_future = request_builder.send();
    let timeout_duration = Duration::from_secs(5);

    let upstream_response = match tokio::time::timeout(timeout_duration, request_future).await {
        Ok(Ok(response)) => {
            debug!(
                "┃ [{}] ✅ Got response from queue backend: {} - Headers: {:?}",
                req_id,
                response.status(),
                response.headers()
            );
            response
        }
        Ok(Err(e)) => {
            return Err(handle_request_error(e, &history_url));
        }
        Err(_) => {
            return Err(handle_timeout_error(&history_url, timeout_duration));
        }
    };

    if !upstream_response.status().is_success() {
        return Err(ProxyError::Upstream(format!(
            "ComfyUI history returned status {} for {}",
            upstream_response.status(), prompt_id
        )));
    }

    // Parse history response
    let response_body = upstream_response
        .bytes()
        .await
        .map_err(|e| ProxyError::Upstream(format!("Failed to read queue response body: {}", e)))?;
    let queu_json: Value = serde_json::from_slice(&response_body)
        .map_err(|e| ProxyError::Json(format!("Failed to parse queue JSON: {}", e)))?;

    // Check running queue
    if let Some(queue_running) = queu_json.get("queue_running").and_then(|v| v.as_array()){

        for queue_elem in queue_running.iter() {
            if queue_elem[1] == prompt_id {
                return Ok(false)
            }
        }
    }
    // Check pending queue
    if let Some(queue_pending) = queu_json.get("queue_pending").and_then(|v| v.as_array()){
        for queue_elem in queue_pending.iter() {
            if queue_elem[1] == prompt_id {
                return Ok(false)
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn same_seed_uses_distinct_save_paths_for_each_request() {
        let workflow = serde_json::json!({
            "13": {
                "inputs": { "noise_seed": 0 },
                "class_type": "SamplerCustom"
            },
            "27": {
                "inputs": { "filename_prefix": "ComfyUI" },
                "class_type": "SaveImage"
            }
        });
        let workflows = Arc::new(HashMap::from([("IMG2.0-S".to_string(), workflow)]));
        let body = Bytes::from_static(br#"{"model":"IMG2.0-S","seed":42}"#);

        let first = create_json_payload(body.clone(), workflows.clone(), "client".to_string(), "request-one")
            .await
            .unwrap();
        let second = create_json_payload(body, workflows, "client".to_string(), "request-two")
            .await
            .unwrap();
        let first: Value = serde_json::from_slice(&first).unwrap();
        let second: Value = serde_json::from_slice(&second).unwrap();

        assert_eq!(first["prompt"]["13"]["inputs"]["noise_seed"], 42);
        assert_eq!(second["prompt"]["13"]["inputs"]["noise_seed"], 42);
        assert_eq!(first["prompt"]["27"]["inputs"]["filename_prefix"], "ComfyUI_request-one");
        assert_eq!(second["prompt"]["27"]["inputs"]["filename_prefix"], "ComfyUI_request-two");
    }

    #[test]
    fn cleanup_guard_removes_files_when_scope_exits() {
        let root = std::env::temp_dir().join(format!("comfyui-cleanup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("ComfyUI_00001_.png");
        std::fs::write(&file, b"test image").unwrap();

        {
            let image = ImageFile {
                filename: file.file_name().unwrap().to_string_lossy().into_owned(),
                subfolder: String::new(),
                type_field: "output".to_string(),
            };
            let _guard = ImageCleanupGuard::new(&root, &[image]);
        }

        assert!(!file.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cleanup_guard_can_be_disarmed_for_url_delivery() {
        let root = std::env::temp_dir().join(format!("comfyui-url-cleanup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("ComfyUI_00001_.png");
        std::fs::write(&file, b"test image").unwrap();
        let image = ImageFile {
            filename: file.file_name().unwrap().to_string_lossy().into_owned(),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };

        ImageCleanupGuard::new(&root, &[image]).disarm();

        assert!(file.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn generated_image_url_is_one_time_and_deletes_the_file() {
        let root = std::env::temp_dir().join(format!("comfyui-url-response-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join(format!("image-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&file, b"png bytes").unwrap();
        let image = ImageFile {
            filename: file.file_name().unwrap().to_string_lossy().into_owned(),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };

        let (bytes, content_type) = take_generated_image(&root, &image).await.unwrap();

        assert_eq!(bytes, Bytes::from_static(b"png bytes"));
        assert_eq!(content_type, "image/png");
        assert!(!file.exists());
        assert!(matches!(
            take_generated_image(&root, &image).await,
            Err(ProxyError::NotFound(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn undelivered_batch_reply_removes_its_published_image() {
        let root = std::env::temp_dir().join(format!("comfyui-url-undelivered-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let image = ImageFile {
            filename: "batch_00001_.png".to_string(),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };
        fs::write(root.join(&image.filename), b"undelivered").unwrap();
        let (mut urls, cleanup) = prepare_image_urls(&root, &[image]).unwrap();
        let pending = PendingImage::new(urls.remove(0), &root).unwrap();
        cleanup.disarm();
        let (sender, receiver) = oneshot::channel();
        assert!(sender.send(pending).is_ok());
        drop(receiver);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn image_urls_hide_batch_filenames_and_rollback_partial_publication() {
        let root = std::env::temp_dir().join(format!("comfyui-url-publish-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let images: Vec<_> = (0..2).map(|index| ImageFile {
            filename: format!("shared-batch_{index}.png"),
            subfolder: String::new(),
            type_field: "output".to_string(),
        }).collect();
        for image in &images {
            fs::write(root.join(&image.filename), b"image").unwrap();
        }
        let (urls, cleanup) = prepare_image_urls(&root, &images).unwrap();
        assert_eq!(urls.len(), 2);
        assert_ne!(urls[0], urls[1]);
        for url in &urls {
            let url = url["url"].as_str().unwrap();
            assert!(!url.contains("shared-batch"));
            let query: HashMap<String, String> = serde_urlencoded::from_str(url.split_once('?').unwrap().1).unwrap();
            assert!(query["filename"].starts_with("image-"));
            assert!(root.join(&query["filename"]).is_file());
        }
        // Failing to deliver a response must remove the newly published files too.
        drop(cleanup);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::write(root.join(&images[0].filename), b"partial").unwrap();
        assert!(prepare_image_urls(&root, &images).is_err());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn concurrent_downloads_consume_an_image_only_once() {
        let root = std::env::temp_dir().join(format!("comfyui-url-race-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let image = ImageFile {
            filename: format!("image-{}.png", uuid::Uuid::new_v4()),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };
        fs::write(root.join(&image.filename), b"one download").unwrap();
        let (first, second) = tokio::join!(
            take_generated_image(&root, &image),
            take_generated_image(&root, &image),
        );
        assert_eq!(usize::from(first.is_ok()) + usize::from(second.is_ok()), 1);
        assert!(matches!(first, Err(ProxyError::NotFound(_)))
            || matches!(second, Err(ProxyError::NotFound(_))));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn image_download_rejects_symlinks_and_non_image_paths() {
        let root = std::env::temp_dir().join(format!("comfyui-url-safety-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let outside = root.with_extension("png");
        fs::write(&outside, b"must never serve").unwrap();
        let image = ImageFile {
            filename: format!("image-{}.png", uuid::Uuid::new_v4()),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };
        std::os::unix::fs::symlink(&outside, root.join(&image.filename)).unwrap();
        assert!(take_generated_image(&root, &image).await.is_err());
        assert_eq!(fs::read(&outside).unwrap(), b"must never serve");
        let traversal = ImageFile { subfolder: "..".to_string(), ..image.clone() };
        assert!(take_generated_image(&root, &traversal).await.is_err());
        let non_image = ImageFile { filename: "config.yaml".to_string(), ..image.clone() };
        fs::write(root.join("config.yaml"), b"private config").unwrap();
        assert!(take_generated_image(&root, &non_image).await.is_err());
        assert!(root.join("config.yaml").exists());
        fs::remove_file(outside).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generated_image_cleanup_rejects_paths_outside_output_directory() {
        let root = std::env::temp_dir().join(format!("comfyui-cleanup-safety-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let outside = root.parent().unwrap().join(format!("outside-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&outside, b"must keep").unwrap();

        let image = ImageFile {
            filename: format!("../{}", outside.file_name().unwrap().to_string_lossy()),
            subfolder: String::new(),
            type_field: "output".to_string(),
        };

        assert_eq!(remove_generated_images(&root, &[image]), 0);
        assert!(outside.exists());
        std::fs::remove_file(outside).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn periodic_cleanup_removes_stale_images_but_keeps_other_files() {
        let root = std::env::temp_dir().join(format!("comfyui-periodic-cleanup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("nested")).unwrap();
        std::fs::write(root.join("stale.png"), b"stale image").unwrap();
        std::fs::write(root.join("nested/stale.webp"), b"stale image").unwrap();
        std::fs::write(root.join("keep.txt"), b"keep").unwrap();

        let removed = cleanup_stale_images_before(&root, SystemTime::now() + Duration::from_secs(1));

        assert_eq!(removed, 2);
        assert!(!root.join("stale.png").exists());
        assert!(!root.join("nested/stale.webp").exists());
        assert!(root.join("keep.txt").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gpt_image2_output_tokens_match_official_calculator_examples() {
        assert_eq!(gpt_image2_output_tokens(1024, 1024, ImageQuality::Low), 196);
        assert_eq!(gpt_image2_output_tokens(1024, 1024, ImageQuality::Medium), 1756);
        assert_eq!(gpt_image2_output_tokens(1024, 1024, ImageQuality::High), 7024);
        assert_eq!(gpt_image2_output_tokens(1024, 1536, ImageQuality::Low), 158);
    }

    #[test]
    fn image_url_response_multiplies_random_output_tokens_bonus_by_image_count() {
        let payload = serde_json::json!({
            "prompt": {
                "5": {
                    "class_type": "EmptyLatentImage",
                    "inputs": { "width": 512, "height": 512, "batch_size": 2 }
                },
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": { "text": "a cat" }
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": { "text": "text, watermark" }
                }
            }
        });
        let usage = estimate_usage(
            &serde_json::to_vec(&payload).unwrap(),
            ImageQuality::Low,
        ).unwrap();
        let response = build_image_response(
            vec![serde_json::json!({"url": "one"}), serde_json::json!({"url": "two"})],
            123,
            &usage,
        );

        assert_eq!(response["created"], 123);
        assert_eq!(response["data"].as_array().unwrap().len(), 2);
        assert_eq!(response["data"][0], serde_json::json!({"url": "one"}));
        assert_eq!(response["usage"]["input_tokens"], 20);
        assert_eq!(response["usage"]["input_tokens_details"], serde_json::json!({
            "image_tokens": 0, "text_tokens": 20
        }));
        let output_tokens = response["usage"]["output_tokens"].as_u64().unwrap();
        assert_eq!(response["usage"]["output_tokens_details"]["image_tokens"], output_tokens);
        assert_eq!(response["usage"]["output_tokens_details"]["text_tokens"], 0);
        assert_eq!(response["usage"]["total_tokens"], 20 + output_tokens);

        let mut outputs = vec![output_tokens];
        for _ in 1..32 {
            let response = build_image_response(
                vec![serde_json::json!({"url": "one"}), serde_json::json!({"url": "two"})],
                123,
                &usage,
            );
            outputs.push(response["usage"]["output_tokens"].as_u64().unwrap());
        }
        assert!(outputs.iter().all(|output| {
            let bonus = *output - 290;
            (15_000..=30_000).contains(&bonus) && bonus % 2 == 0
        }));
        assert!(outputs.windows(2).any(|pair| pair[0] != pair[1]));
    }


    #[test]
    fn generation_image_count_defaults_to_one_and_enforces_one_to_ten() {
        let workflow = serde_json::json!({
            "5": {
                "inputs": { "width": 512, "height": 512, "batch_size": 1 },
                "class_type": "EmptyLatentImage"
            }
        });
        let workflows = HashMap::from([("IMG2.0-S".to_string(), workflow)]);

        let default = parse_generation_input(br#"{"prompt":"a cat"}"#, &workflows).unwrap();
        assert_eq!(default.n, 1);

        for n in [1, 10] {
            let body = format!(r#"{{"prompt":"a cat","n":{n}}}"#);
            assert_eq!(parse_generation_input(body.as_bytes(), &workflows).unwrap().n, n);
        }
        for n in [0, 11] {
            let body = format!(r#"{{"prompt":"a cat","n":{n}}}"#);
            assert!(matches!(
                parse_generation_input(body.as_bytes(), &workflows),
                Err(ProxyError::Validation(message)) if message == "n must be between 1 and 10"
            ));
        }
    }

    #[test]
    fn generation_usage_uses_requested_quality() {
        let workflow = serde_json::json!({
            "5": {
                "inputs": { "width": 512, "height": 512, "batch_size": 1 },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": { "text": "" },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "Positive Prompt" }
            }
        });
        let workflows = HashMap::from([("IMG2.0-S".to_string(), workflow)]);

        for (quality, expected) in [("low", 145), ("medium", 1303), ("high", 5212)] {
            let body = format!(r#"{{"prompt":"a cat","quality":"{}"}}"#, quality);
            let input = parse_generation_input(body.as_bytes(), &workflows).unwrap();
            let response = build_image_response(
                vec![serde_json::json!({"b64_json": "image"})],
                123,
                &input.usage,
            );
            let output_tokens = response["usage"]["output_tokens"].as_u64().unwrap();
            assert!((expected + 7_500..=expected + 15_000).contains(&output_tokens));
            assert_eq!(
                response["usage"]["total_tokens"],
                input.usage.input_tokens + output_tokens
            );
        }

        for body in [r#"{"prompt":"a cat"}"#, r#"{"prompt":"a cat","quality":"auto"}"#] {
            let input = parse_generation_input(body.as_bytes(), &workflows).unwrap();
            let expected = input.usage.output_tokens_per_image;
            let response = build_image_response(
                vec![serde_json::json!({"b64_json": "image"})],
                123,
                &input.usage,
            );
            let output_tokens = response["usage"]["output_tokens"].as_u64().unwrap();
            assert!([145, 1303, 5212].contains(&expected));
            assert!((expected + 7_500..=expected + 15_000).contains(&output_tokens));
        }
    }

    #[tokio::test]
    async fn official_image_parameters_are_accepted_even_when_unused() {
        let workflow = serde_json::json!({
            "5": {
                "inputs": { "width": 512, "height": 512, "batch_size": 1 },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": { "text": "" },
                "class_type": "CLIPTextEncode",
                "_meta": { "title": "Positive Prompt" }
            }
        });
        let workflows = Arc::new(HashMap::from([("IMG2.0-S".to_string(), workflow)]));
        let body = Bytes::from_static(br#"{
            "prompt":"a cat",
            "background":"transparent",
            "model":null,
            "moderation":"low",
            "n":1,
            "output_compression":80,
            "output_format":"webp",
            "partial_images":2,
            "quality":"high",
            "response_format":"b64_json",
            "size":"auto",
            "stream":true,
            "style":"natural",
            "user":"customer-123"
        }"#);

        let payload = create_json_payload(body, workflows, "client".to_string(), "request")
            .await
            .unwrap();
        let payload: Value = serde_json::from_slice(&payload).unwrap();

        assert_eq!(payload["prompt"]["6"]["inputs"]["text"], "a cat");
        assert_eq!(payload["prompt"]["5"]["inputs"]["width"], 512);
        assert_eq!(payload["prompt"]["5"]["inputs"]["height"], 512);
        assert_eq!(payload["prompt"]["5"]["inputs"]["batch_size"], 1);
    }


    #[test]
    fn batch_payload_preserves_prompt_seed_and_slot_order() {
        let workflow = serde_json::json!({
            "5": {"inputs":{"width":512,"height":512,"batch_size":1},"class_type":"EmptyLatentImage"},
            "6": {"inputs":{"prompts_json":"[\"\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Positive Prompt"}},
            "7": {"inputs":{"prompts_json":"[\"text, watermark\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Negative Prompt"}},
            "23": {"inputs":{"seeds_json":"[{\"seed\":0,\"offset\":0}]"},"class_type":"OpenAIBatchNoise"},
            "27": {"inputs":{"filename_prefix":"ComfyUI"},"class_type":"SaveImage"}
        });
        let workflows = HashMap::from([("IMG2.0-S".to_string(), workflow)]);
        let slots = vec![
            BatchSlot::test_slot("cat", "bad anatomy", 42, 0),
            BatchSlot::test_slot("dog", "watermark", 7, 0),
            BatchSlot::test_slot("dog", "watermark", 7, 1),
        ];

        let payload = create_batch_json_payload(&slots, &workflows, "client", "batch-1").unwrap();
        let payload: Value = serde_json::from_slice(&payload).unwrap();

        assert_eq!(payload["prompt"]["5"]["inputs"]["batch_size"], 3);
        assert_eq!(payload["prompt"]["6"]["inputs"]["prompts_json"], "[\"cat\",\"dog\",\"dog\"]");
        assert_eq!(payload["prompt"]["7"]["inputs"]["prompts_json"], "[\"bad anatomy\",\"watermark\",\"watermark\"]");
        let seeds: Value = serde_json::from_str(
            payload["prompt"]["23"]["inputs"]["seeds_json"].as_str().unwrap()
        ).unwrap();
        assert_eq!(seeds, serde_json::json!([
            {"seed":42,"offset":0},
            {"seed":7,"offset":0},
            {"seed":7,"offset":1}
        ]));
        assert_eq!(payload["prompt"]["27"]["inputs"]["filename_prefix"], "ComfyUI_batch-1");

        let ten_slots = (0..10)
            .map(|offset| BatchSlot::test_slot("cat", "", 42, offset))
            .collect::<Vec<_>>();
        let payload = create_batch_json_payload(&ten_slots, &workflows, "client", "batch-10")
            .unwrap();
        let payload: Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(payload["prompt"]["5"]["inputs"]["batch_size"], 10);
    }


    #[test]
    fn batch_payload_accepts_parallel_png_output() {
        let workflow = serde_json::json!({
            "5": {"inputs":{"width":512,"height":512,"batch_size":1},"class_type":"EmptyLatentImage"},
            "6": {"inputs":{"prompts_json":"[\"\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Positive Prompt"}},
            "7": {"inputs":{"prompts_json":"[\"text, watermark\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Negative Prompt"}},
            "23": {"inputs":{"seeds_json":"[{\"seed\":0,\"offset\":0}]"},"class_type":"OpenAIBatchNoise"},
            "27": {"inputs":{"filename_prefix":"ComfyUI"},"class_type":"ParallelSaveImage"}
        });
        let workflows = HashMap::from([("IMG2.0-S".to_string(), workflow)]);
        let slots = vec![BatchSlot::test_slot("cat", "", 42, 0)];

        let payload = create_batch_json_payload(&slots, &workflows, "client", "batch-1")
            .unwrap();
        let payload: Value = serde_json::from_slice(&payload).unwrap();

        assert_eq!(payload["prompt"]["27"]["inputs"]["filename_prefix"], "ComfyUI_batch-1");
    }


    #[test]
    fn batch_payload_supports_sixteen_internal_slots() {
        let workflow = serde_json::json!({
            "5": {"inputs":{"width":512,"height":512,"batch_size":1},"class_type":"EmptyLatentImage"},
            "6": {"inputs":{"prompts_json":"[\"\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Positive Prompt"}},
            "7": {"inputs":{"prompts_json":"[\"text, watermark\"]"},"class_type":"OpenAIBatchCLIPTextEncode","_meta":{"title":"Negative Prompt"}},
            "23": {"inputs":{"seeds_json":"[{\"seed\":0,\"offset\":0}]"},"class_type":"OpenAIBatchNoise"},
            "27": {"inputs":{"filename_prefix":"ComfyUI"},"class_type":"SaveImage"}
        });
        let workflows = HashMap::from([("IMG2.0-S".to_string(), workflow)]);
        let slots = (0..16)
            .map(|offset| BatchSlot::test_slot("cat", "", 42, offset))
            .collect::<Vec<_>>();

        let payload = create_batch_json_payload(&slots, &workflows, "client", "batch-16")
            .unwrap();
        let payload: Value = serde_json::from_slice(&payload).unwrap();

        assert_eq!(payload["prompt"]["5"]["inputs"]["batch_size"], 16);
    }


    #[tokio::test]
    async fn batch_collector_groups_compatible_slots_and_defers_other_sizes() {
        let (sender, mut receiver) = mpsc::channel(8);
        let first = BatchSlot::test_slot("cat", "", 1, 0);
        let mut incompatible = BatchSlot::test_slot("large", "", 2, 0);
        incompatible.key.width = 1024;
        incompatible.key.height = 1024;
        let matching = BatchSlot::test_slot("dog", "", 3, 0);
        sender.send(incompatible).await.unwrap();
        sender.send(matching).await.unwrap();
        let mut pending = std::collections::VecDeque::new();

        let batch = collect_batch(
            first,
            &mut receiver,
            &mut pending,
            8,
            Duration::from_millis(2),
        ).await;

        assert_eq!(batch.iter().map(|slot| slot.prompt.as_str()).collect::<Vec<_>>(), vec!["cat", "dog"]);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].prompt, "large");
    }


    #[test]
    fn batch_capacity_counts_slots_moved_out_of_the_channel() {
        let capacity = Arc::new(tokio::sync::Semaphore::new(2));
        let reserved = reserve_batch_capacity(&capacity, 2).unwrap();

        assert!(matches!(
            reserve_batch_capacity(&capacity, 1),
            Err(ProxyError::Busy(_))
        ));

        drop(reserved);
        assert!(reserve_batch_capacity(&capacity, 2).is_ok());
    }

    #[test]
    fn batch_queue_order_prefers_capacity_and_rotates_ties() {
        let (first_sender, _first_receiver, first_capacity) = dynamic_batch_channel();
        let (second_sender, _second_receiver, second_capacity) = dynamic_batch_channel();
        let queues = vec![
            BatchQueue { sender: first_sender, capacity: first_capacity.clone() },
            BatchQueue { sender: second_sender, capacity: second_capacity },
        ];

        assert_eq!(batch_queue_order(&queues, 1), vec![1, 0]);
        let _busy = reserve_batch_capacity(&first_capacity, 10).unwrap();
        assert_eq!(batch_queue_order(&queues, 0), vec![1, 0]);
    }

}
