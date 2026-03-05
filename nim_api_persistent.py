#!/usr/bin/env python3
import base64
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import requests
import torch
import torch.distributed as dist

# Will be imported after distributed init
Control2WorldInference = None
InferenceArguments = None
SetupArguments = None


def log(msg: str, rank: int = None, level: str = "INFO"):
    """Structured logging with timestamp, rank, and level."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    r = rank if rank is not None else (get_rank() if dist.is_initialized() else 0)
    print(f"[{ts}|{level}|Rank {r}] {msg}", flush=True)


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# Global inference engine (loaded once)
INFERENCE_ENGINE = None
ENGINE_LOCK = threading.Lock()
STARTUP_TIME = time.time()
REQUEST_COUNT = 0


def apply_nim_env_settings():
    """Apply ONLY the environment variables that NIM sets."""
    settings = {
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
        'CUDA_MODULE_LOADING': 'LAZY',
        'TORCHINDUCTOR_CACHE_DIR': '/tmp/.cache/',
        'TORCH_NCCL_USE_COMM_NONBLOCKING': '0',
        'NCCL_CUMEM_HOST_ENABLE': '0',
    }
    for k, v in settings.items():
        os.environ.setdefault(k, v)
    
    if is_rank0():
        log("Environment settings applied:")
        for k, v in settings.items():
            log(f"  {k}={os.environ.get(k, 'NOT SET')}")


def log_gpu_info():
    """Log GPU information for debugging."""
    try:
        gpu_count = torch.cuda.device_count()
        log(f"CUDA available: {torch.cuda.is_available()}, GPU count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log(f"  GPU {i}: {name}, {mem_total:.1f} GB")
    except Exception as e:
        log(f"Could not get GPU info: {e}", level="WARN")


def log_memory_usage(label: str = ""):
    """Log current GPU memory usage."""
    try:
        rank = get_rank()
        allocated = torch.cuda.memory_allocated(rank) / (1024**3)
        reserved = torch.cuda.memory_reserved(rank) / (1024**3)
        log(f"GPU memory {label}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    except Exception:
        pass


def load_inference_engine(hint_keys: list[str] = None):
    """Load the inference engine once at startup."""
    global INFERENCE_ENGINE, Control2WorldInference, InferenceArguments, SetupArguments
    
    rank = get_rank()
    world_size = get_world_size()
    load_start = time.time()
    
    log(f"Importing cosmos modules...")
    
    try:
        from cosmos_transfer2.inference import Control2WorldInference as C2WI
        from cosmos_transfer2.config import InferenceArguments as IA, SetupArguments as SA
        
        Control2WorldInference = C2WI
        InferenceArguments = IA
        SetupArguments = SA
        log(f"Modules imported in {time.time() - load_start:.1f}s")
    except Exception as e:
        log(f"FATAL: Failed to import cosmos modules: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        raise
    
    if hint_keys is None:
        hint_keys = ['edge']
    
    model_name = hint_keys[0] if hint_keys else 'edge'
    multibranch = len(hint_keys) > 1
    
    log(f"Creating inference engine:")
    log(f"  hint_keys={hint_keys} ({'multibranch' if multibranch else 'single-control'})")
    log(f"  model={model_name}, world_size={world_size}")
    # Default: enabled in v1.5.0; set ENABLE_PARALLEL_TOKENIZER=0 to disable
    enable_parallel_tokenizer = os.environ.get('ENABLE_PARALLEL_TOKENIZER', '1').strip().lower() not in ('0', 'false', 'no')
    log(f"  enable_parallel_tokenizer={enable_parallel_tokenizer} (ENABLE_PARALLEL_TOKENIZER={os.environ.get('ENABLE_PARALLEL_TOKENIZER', '1 (default)')})")
    log(f"  disable_guardrails=True")

    try:
        # Build kwargs, only including fields that exist in this version's SetupArguments
        setup_kwargs = dict(
            model=model_name,
            context_parallel_size=world_size,
            disable_guardrails=True,
            output_dir=Path('/tmp/cosmos_output'),
        )
        # These fields were added in later versions; skip them if not supported
        sa_fields = SetupArguments.model_fields
        if 'enable_parallel_tokenizer' in sa_fields:
            setup_kwargs['enable_parallel_tokenizer'] = enable_parallel_tokenizer
        if 'benchmark' in sa_fields:
            setup_kwargs['benchmark'] = False
        
        setup_args = SetupArguments(**setup_kwargs)
        log(f"SetupArguments created: checkpoint_path={setup_args.checkpoint_path}")
        
        engine_start = time.time()
        engine = Control2WorldInference(
            args=setup_args,
            batch_hint_keys=hint_keys,
        )
        
        INFERENCE_ENGINE = engine
        load_elapsed = time.time() - load_start
        engine_elapsed = time.time() - engine_start
        
        log(f"Inference engine ready!")
        log(f"  Engine init: {engine_elapsed:.1f}s")
        log(f"  Total load time: {load_elapsed:.1f}s")
        log_memory_usage("after model load")
        
    except Exception as e:
        log(f"FATAL: Failed to create inference engine: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        raise
    
    return engine


# ============================================================
# Helper: Save base64/URL data to a temp file
# ============================================================

def is_url(data):
    return isinstance(data, str) and (data.startswith('http://') or data.startswith('https://'))


def get_bytes(data, timeout=120):
    """Get bytes from URL or base64 string."""
    if is_url(data):
        log(f"Downloading from URL: {data[:100]}...")
        dl_start = time.time()
        resp = requests.get(data, timeout=timeout)
        resp.raise_for_status()
        log(f"Downloaded {len(resp.content)} bytes in {time.time() - dl_start:.1f}s")
        return resp.content
    return base64.b64decode(data.split('base64,')[-1])


def save_media_to_file(data, temp_dir, filename):
    """Save base64 or URL media to a temp file. Returns path or None."""
    if data is None:
        return None
    media_bytes = get_bytes(data)
    path = os.path.join(temp_dir, filename)
    with open(path, 'wb') as f:
        f.write(media_bytes)
    return path


# ============================================================
# Build control configs from request, matching NIM's schema
# ============================================================

def build_control_configs(request_data: dict):
    """
    Build control configs from request data.
    
    By this point, any base64/URL control/mask data has already been decoded
    to files by the API endpoint. Only file paths remain in the dicts.
    
    NIM's control schema per control type (edge/depth/vis/seg):
      - control_path: path to pre-computed control video (optional)
      - control_weight: float, default 1.0
      - mask_path: path to mask video (optional)
      - mask_prompt: str (optional)
      - preset_edge_threshold: str (edge only)
      - preset_blur_strength: str (vis only)
      - control_prompt: str (seg only)
    """
    from cosmos_transfer2.config import EdgeConfig, DepthConfig, SegConfig, BlurConfig
    
    edge_cfg = None
    depth_cfg = None
    seg_cfg = None
    vis_cfg = None
    
    if request_data.get('edge'):
        d = request_data['edge']
        edge_cfg = EdgeConfig(
            control_weight=d.get('control_weight', 1.0),
            control_path=d.get('control_path'),
            mask_path=d.get('mask_path'),
            mask_prompt=d.get('mask_prompt'),
            preset_edge_threshold=d.get('preset_edge_threshold', 'medium'),
        )
    
    if request_data.get('depth'):
        d = request_data['depth']
        depth_cfg = DepthConfig(
            control_weight=d.get('control_weight', 1.0),
            control_path=d.get('control_path'),
            mask_path=d.get('mask_path'),
            mask_prompt=d.get('mask_prompt'),
        )
    
    if request_data.get('seg'):
        d = request_data['seg']
        seg_cfg = SegConfig(
            control_weight=d.get('control_weight', 1.0),
            control_path=d.get('control_path'),
            mask_path=d.get('mask_path'),
            mask_prompt=d.get('mask_prompt'),
            control_prompt=d.get('control_prompt'),
        )
    
    if request_data.get('vis'):
        d = request_data['vis']
        vis_cfg = BlurConfig(
            control_weight=d.get('control_weight', 1.0),
            control_path=d.get('control_path'),
            mask_path=d.get('mask_path'),
            mask_prompt=d.get('mask_prompt'),
            preset_blur_strength=d.get('preset_blur_strength', 'medium'),
        )
    
    # Default to edge if no control specified
    if not any([edge_cfg, depth_cfg, seg_cfg, vis_cfg]):
        edge_cfg = EdgeConfig(control_weight=1.0)
    
    return edge_cfg, depth_cfg, seg_cfg, vis_cfg


def run_inference_request(request_data: dict) -> Optional[str]:
    """Run a single inference request using the pre-loaded engine."""
    global INFERENCE_ENGINE
    
    if INFERENCE_ENGINE is None:
        raise RuntimeError("Inference engine not loaded!")
    
    # Build control configs from pre-processed data (base64 already decoded to files)
    edge_cfg, depth_cfg, seg_cfg, vis_cfg = build_control_configs(request_data)
    
    # Build kwargs matching InferenceArguments
    kwargs = {
        'name': request_data.get('name', str(uuid.uuid4())[:8]),
        'video_path': request_data['video_path'],
        'prompt': request_data.get('prompt', 'A video'),
        'guidance': request_data.get('guidance', 3),           # NIM default
        'num_steps': request_data.get('num_steps', 35),        # NIM default
        'seed': request_data.get('seed', 42),
        'resolution': str(request_data.get('resolution', '480')),  # NIM default, ensure string
        'num_conditional_frames': request_data.get('num_conditional_frames', 1),
        'num_video_frames_per_chunk': request_data.get('num_video_frames_per_chunk', 93),
        'edge': edge_cfg,
        'depth': depth_cfg,
        'seg': seg_cfg,
        'vis': vis_cfg,
    }
    
    # Only add negative_prompt if explicitly provided and non-empty
    neg_prompt = request_data.get('negative_prompt')
    if neg_prompt:
        kwargs['negative_prompt'] = neg_prompt
    # Otherwise InferenceArguments uses DEFAULT_NEGATIVE_PROMPT
    
    # sigma_max (optional)
    if request_data.get('sigma_max') is not None:
        kwargs['sigma_max'] = str(request_data['sigma_max'])
    
    # image_context (optional - path to already-saved file)
    if request_data.get('image_context_path'):
        kwargs['image_context_path'] = request_data['image_context_path']
    
    sample = InferenceArguments(**kwargs)
    output_dir = Path(request_data['output_dir'])
    
    output_paths = INFERENCE_ENGINE.generate([sample], output_dir)
    
    if output_paths:
        return output_paths[0]
    return None


def worker_loop():
    """Worker loop for non-rank-0 processes."""
    rank = get_rank()
    log(f"Entering worker loop...")
    req_num = 0
    
    while True:
        signal = torch.zeros(1, dtype=torch.int32, device='cuda')
        dist.broadcast(signal, src=0)
        
        sig_val = signal.item()
        if sig_val == 0:
            log(f"Shutdown signal received")
            break
        elif sig_val == 1:
            req_num += 1
            data_len = torch.zeros(1, dtype=torch.int64, device='cuda')
            dist.broadcast(data_len, src=0)
            
            data_buffer = torch.zeros(int(data_len.item()), dtype=torch.uint8, device='cuda')
            dist.broadcast(data_buffer, src=0)
            
            request_json = bytes(data_buffer.cpu().numpy()).decode('utf-8')
            request_data = json.loads(request_json)
            
            log(f"Worker request #{req_num}, data_len={data_len.item()} bytes")
            
            try:
                run_inference_request(request_data)
                log(f"Worker request #{req_num} completed")
            except Exception as e:
                log(f"Worker request #{req_num} FAILED: {e}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")


def broadcast_inference(request_data: dict):
    """Rank 0: Broadcast inference request to all workers."""
    signal = torch.ones(1, dtype=torch.int32, device='cuda')
    dist.broadcast(signal, src=0)
    
    data_bytes = json.dumps(request_data).encode('utf-8')
    data_len = torch.tensor([len(data_bytes)], dtype=torch.int64, device='cuda')
    dist.broadcast(data_len, src=0)
    
    data_buffer = torch.tensor(list(data_bytes), dtype=torch.uint8, device='cuda')
    dist.broadcast(data_buffer, src=0)
    
    log(f"Broadcast {len(data_bytes)} bytes to {get_world_size()} workers")


def broadcast_shutdown():
    """Rank 0: Signal workers to shutdown."""
    signal = torch.zeros(1, dtype=torch.int32, device='cuda')
    dist.broadcast(signal, src=0)


def create_fastapi_app():
    """Create the FastAPI application (rank 0 only)."""
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Cosmos Transfer2.5 Persistent API", version="1.5.0")
    
    @app.post("/v1/infer")
    async def infer(request: Any = Body(...)):
        global REQUEST_COUNT
        REQUEST_COUNT += 1
        req_id = REQUEST_COUNT

        # Benchmarking clients send warmup as raw bytes or a double-serialized JSON string.
        if isinstance(request, (str, bytes)):
            request = json.loads(request)

        log(f"[Req #{req_id}] Received request, keys: {list(request.keys())}")
        
        video_data = request.get('video') or request.get('b64_video') or request.get('input_video')
        if not video_data:
            log(f"[Req #{req_id}] Missing 'video' field", level="ERROR")
            raise HTTPException(400, "Missing 'video' field")
        
        # Use /workspace/outputs for temp files - accessible by all ranks
        temp_dir = tempfile.mkdtemp(dir="/workspace/outputs")
        
        try:
            # Save input video
            video_bytes = get_bytes(video_data)
            video_path = os.path.join(temp_dir, 'input.mp4')
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            log(f"[Req #{req_id}] Input video: {len(video_bytes)} bytes -> {video_path}")
            
            # Handle seed: None = random, 0 is a valid seed
            seed = request.get('seed')
            if seed is None:
                seed = random.randint(1, 2**16)
            
            output_dir = os.path.join(temp_dir, 'output')
            
            # Detect which controls are requested
            active_controls = [c for c in ['edge', 'depth', 'seg', 'vis'] if request.get(c)]
            
            # Build request_data matching NIM's Transfer2Request schema
            request_data = {
                'name': str(uuid.uuid4())[:8],
                'video_path': video_path,
                'output_dir': output_dir,
                'prompt': request.get('prompt', 'A video'),
                'guidance': request.get('guidance', 3),           # NIM default
                'num_steps': request.get('num_steps', 35),        # NIM default
                'seed': seed,
                'resolution': str(request.get('resolution', '480')),  # NIM default
                'num_conditional_frames': request.get('num_conditional_frames', 1),
                'num_video_frames_per_chunk': request.get('num_video_frames_per_chunk', 93),
            }
            
            # Optional fields - only include if provided
            if request.get('negative_prompt'):
                request_data['negative_prompt'] = request['negative_prompt']
            
            if request.get('sigma_max') is not None:
                request_data['sigma_max'] = request['sigma_max']
            
            # Handle image_context (base64/URL -> save to file)
            if request.get('image_context'):
                ic_path = save_media_to_file(request['image_context'], temp_dir, 'image_context.png')
                request_data['image_context_path'] = ic_path
            
            # Process control configs: decode base64/URL to files BEFORE broadcast
            # so we don't send huge blobs over NCCL
            for ctrl in ['edge', 'depth', 'seg', 'vis']:
                if request.get(ctrl):
                    ctrl_data = dict(request[ctrl])  # copy to avoid mutating request
                    # Save control video if provided as base64/URL
                    if ctrl_data.get('control'):
                        ctrl_path = save_media_to_file(ctrl_data['control'], temp_dir, f'{ctrl}_control.mp4')
                        ctrl_data['control_path'] = ctrl_path
                        del ctrl_data['control']  # Remove base64, keep only path
                        log(f"[Req #{req_id}] {ctrl} control video saved: {ctrl_path}")
                    # Save mask if provided as base64/URL
                    if ctrl_data.get('mask'):
                        mask_path_file = save_media_to_file(ctrl_data['mask'], temp_dir, f'{ctrl}_mask.mp4')
                        ctrl_data['mask_path'] = mask_path_file
                        del ctrl_data['mask']  # Remove base64, keep only path
                    request_data[ctrl] = ctrl_data
            
            # Default to edge if no control specified
            if not active_controls:
                request_data['edge'] = {'control_weight': 1.0}
                active_controls = ['edge (default)']
            
            # Log full request details
            log(f"[Req #{req_id}] Inference params:")
            log(f"  controls={active_controls}")
            log(f"  guidance={request_data['guidance']}, num_steps={request_data['num_steps']}")
            log(f"  resolution={request_data['resolution']}, seed={seed}")
            log(f"  negative_prompt={'yes' if request_data.get('negative_prompt') else 'default'}")
            log(f"  sigma_max={request_data.get('sigma_max', 'not set')}")
            log(f"  GPUs={get_world_size()}")
            
            start = time.time()
            
            with ENGINE_LOCK:
                if get_world_size() > 1:
                    broadcast_inference(request_data)
                output_path = run_inference_request(request_data)
            
            elapsed = time.time() - start
            
            if not output_path or not os.path.exists(output_path):
                log(f"[Req #{req_id}] FAILED: No output video generated after {elapsed:.2f}s", level="ERROR")
                return JSONResponse({'b64_video': None, 'seed': seed, 'error': 'No video generated'})
            
            output_size = os.path.getsize(output_path)
            b64_video = base64.b64encode(open(output_path, 'rb').read()).decode()
            
            log(f"[Req #{req_id}] SUCCESS: {elapsed:.2f}s, output={output_size} bytes, b64_len={len(b64_video)}")
            log_memory_usage(f"[Req #{req_id}] after inference")
            
            return JSONResponse({'b64_video': b64_video, 'seed': seed})
            
        except HTTPException:
            raise
        except Exception as e:
            log(f"[Req #{req_id}] ERROR: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            raise HTTPException(500, str(e))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @app.get("/health")
    async def health():
        uptime = time.time() - STARTUP_TIME
        return {
            "status": "healthy",
            "model_loaded": INFERENCE_ENGINE is not None,
            "uptime_seconds": round(uptime),
            "requests_served": REQUEST_COUNT,
            "world_size": get_world_size(),
        }
    
    @app.get("/v1/health/live")
    async def health_live():
        return {"status": "live"}
    
    @app.get("/v1/health/ready")
    async def health_ready():
        if INFERENCE_ENGINE is not None:
            return {"status": "ready"}
        raise HTTPException(503, "Model not loaded")
    
    @app.get("/")
    async def root():
        uptime = time.time() - STARTUP_TIME
        return {
            "message": "Cosmos Transfer2.5 PERSISTENT API",
            "version": "1.5.0",
            "model_loaded": INFERENCE_ENGINE is not None,
            "world_size": get_world_size(),
            "uptime_seconds": round(uptime),
            "requests_served": REQUEST_COUNT,
        }
    
    return app


def main():
    """Main entry point - run with torchrun."""
    global STARTUP_TIME
    STARTUP_TIME = time.time()
    
    log("=" * 60, rank=0)
    log("Cosmos Transfer2.5 PERSISTENT API - Starting", rank=0)
    log("=" * 60, rank=0)
    log(f"Python: {sys.version}", rank=0)
    log(f"PyTorch: {torch.__version__}", rank=0)
    log(f"CUDA available: {torch.cuda.is_available()}", rank=0)
    
    apply_nim_env_settings()
    
    # Set large NCCL timeout so workers don't die while waiting for HTTP requests
    # Default is 600s (10min), which causes crashes during idle periods between API calls
    os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '86400')  # 24 hours
    
    if 'RANK' in os.environ:
        log(f"Distributed mode: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}", rank=0)
        from datetime import timedelta
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=24))
        torch.cuda.set_device(get_rank())
    else:
        log("Single-process mode (no RANK env var)", rank=0)
    
    rank = get_rank()
    world_size = get_world_size()
    
    log(f"Process initialized: rank={rank}, world_size={world_size}, "
        f"cuda_device={torch.cuda.current_device()}")
    
    if rank == 0:
        log_gpu_info()
    
    # Ensure outputs dir exists
    os.makedirs('/workspace/outputs', exist_ok=True)
    
    # Load model on all ranks
    # Must load ALL control types that the speed benchmark tests:
    #   low_payload  = edge only (480p)
    #   high_payload = edge + depth + seg (720p)
    # When len(hint_keys) > 1, the engine loads the multibranch checkpoint
    # (same as NIM, which supports all controls)
    log("Loading inference engine with multibranch model (edge+depth+seg+vis)...")
    load_inference_engine(hint_keys=['edge', 'depth', 'seg', 'vis'])
    
    if dist.is_initialized():
        log("Waiting at barrier for all ranks...")
        dist.barrier()
        log("All ranks synchronized")
    
    startup_elapsed = time.time() - STARTUP_TIME
    
    if rank == 0:
        import uvicorn
        app = create_fastapi_app()
        port = int(os.environ.get('API_PORT', '8000'))
        
        log("=" * 60)
        log("SERVER READY")
        log("=" * 60)
        log(f"  Model: multibranch (edge+depth+seg+vis)")
        log(f"  GPUs: {world_size}")
        log(f"  Port: {port}")
        log(f"  Defaults: guidance=3, resolution=480, num_steps=35")
        log(f"  Startup time: {startup_elapsed:.1f}s")
        log("=" * 60)
        
        try:
            uvicorn.run(app, host="0.0.0.0", port=port)
        except Exception as e:
            log(f"FATAL: Uvicorn crashed: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
        finally:
            log("Shutting down...")
            if dist.is_initialized() and world_size > 1:
                broadcast_shutdown()
    else:
        worker_loop()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    log("Process exited cleanly")


if __name__ == "__main__":
    main()
