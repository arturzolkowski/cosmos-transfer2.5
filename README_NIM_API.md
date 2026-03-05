# Cosmos Transfer2.5 — Persistent Benchmark API

`nim_api_persistent.py` is a FastAPI-based inference server designed for performance
benchmarking of Cosmos Transfer2.5. It is compatible with the `benchmarking-speed`
client used in JET CI and exposes a `/v1/infer` endpoint that mirrors the NIM API schema.

## Why a persistent server?

The standard CLI (`cosmos_transfer2/inference.py`) spawns a fresh process for every
request, incurring ~60s of model-load overhead each time. This server loads the model
once at startup and handles subsequent requests in-memory, making it suitable for
latency and throughput benchmarks.

## Architecture

- **Rank 0** runs the FastAPI/uvicorn HTTP server and handles request parsing.
- **Ranks 1…N** run a worker loop, receiving inference jobs via NCCL broadcast.
- All ranks participate in the forward pass (context parallelism).
- The model is loaded in multibranch mode (`edge + depth + seg + vis`) at startup,
  matching the NIM checkpoint layout.

## Building the container

`nim_api_persistent.py`, `fastapi`, and `uvicorn` are all included in the standard
nightly image. No separate build step is required beyond the usual nightly build.

```bash
# Step 1: build the nightly base image
docker build -f docker/nightly.Dockerfile -t cosmos-transfer2.5:nightly .

# Step 2 (optional): tag it as the API image for clarity
docker build -f Dockerfile.api -t cosmos-transfer2.5-api:latest .
```

To use a different PyTorch base image pass `--build-arg BASE_IMAGE=...` to step 1.

## Running

Model weights must be available in the HF cache inside the container. Either bake them
in during the build or mount them at runtime (see below).

### Single GPU
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v /path/to/hf_cache:/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -p 8000:8000 \
  cosmos-transfer2.5:nightly \
  python3 /workspace/nim_api_persistent.py
```

### Multi-GPU (e.g. 2 GPUs)
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  -v /path/to/hf_cache:/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -e MASTER_ADDR=localhost \
  -e MASTER_PORT=29500 \
  -p 8000:8000 \
  cosmos-transfer2.5:nightly \
  torchrun --nproc_per_node=2 /workspace/nim_api_persistent.py
```

## Health endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/health/live` | Returns 200 immediately |
| `GET /v1/health/ready` | Returns 200 once model is loaded, 503 otherwise |
| `GET /health` | Returns uptime, request count, GPU count |

## Request schema (`POST /v1/infer`)

```json
{
  "prompt": "A robot arm assembling parts",
  "video": "<base64-encoded mp4 or https:// URL>",
  "guidance": 3,
  "resolution": "480",
  "num_steps": 35,
  "seed": 42,
  "edge": {
    "control": "<base64-encoded mp4 or https:// URL>",
    "control_weight": 1.0
  },
  "depth": {
    "control": "<base64-encoded mp4 or https:// URL>",
    "control_weight": 1.0
  },
  "seg": {
    "control": "<base64-encoded mp4 or https:// URL>",
    "control_weight": 1.0
  },
  "vis": {
    "control": "<base64-encoded mp4 or https:// URL>",
    "control_weight": 1.0
  }
}
```

All control fields are optional. If none are provided, `edge` is used by default.
`video` and control inputs accept either a base64-encoded MP4 or a public HTTPS URL.

## Response

```json
{
  "b64_video": "<base64-encoded output mp4>",
  "seed": 42
}
```

## Benchmark payloads

Test payloads are in [`assets/benchmark_nim/`](assets/benchmark_nim/):

| Payload | Resolution | Controls | Use case |
|---|---|---|---|
| `low_payload_spec.json` | 480p | edge only | Latency / single-control |
| `high_payload_spec.json` | 720p | edge + depth + seg | Throughput / multi-control |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | HTTP port to listen on |
| `HF_HUB_OFFLINE` | `1` | Use only locally cached model weights |
| `ENABLE_PARALLEL_TOKENIZER` | `1` | Enable parallel tokenizer (v1.5.0+) |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `86400` | NCCL heartbeat timeout (24h keeps workers alive between requests) |
