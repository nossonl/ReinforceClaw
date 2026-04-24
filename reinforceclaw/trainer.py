"""Adaptive LoRA trainer with MLX and CUDA backends."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import fcntl
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from . import db
from .backend_cuda import CUDABackend
from .backend_mlx import MLXBackend, mlx_drain

GB = 1024 ** 3
MIN_TRAINING_BUDGET = 2 * GB
MAC_OS_RESERVE = 8 * GB
LINUX_OS_RESERVE = 4 * GB
TRAIN_LOG_PATH = Path.home() / ".reinforceclaw" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.lock"
BACKGROUND_IDLE_LOAD = 0.85
BACKGROUND_WINDOW_MINUTES = 180
_LOG_SECURED = False
_UNSET = object()
_HF_TOKEN_CACHE = _UNSET

# lazy MLX imports
mx = nn = optim = mlx_load = linear_to_lora_layers = tree_map = tree_unflatten = None
_HF_SOURCE_CACHE: dict[str, str] = {}
_HF_ALLOW_PATTERNS = ("*.json", "*.jinja", "*.model", "*.txt", "*.safetensors", "*.index.json", "*.tiktoken")
_HF_IGNORE_PATTERNS = ("*.bin", "*.h5", "*.msgpack", "*.ot", "*.npz", "*.gguf", "*.onnx", "*.tflite", "*.zip")


def _ensure_mlx():
    global mx, nn, optim, mlx_load, linear_to_lora_layers, tree_map, tree_unflatten
    if mx is not None:
        return
    import mlx.core
    import mlx.nn
    import mlx.optimizers
    import mlx.utils
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers as _linear_to_lora_layers

    mx, nn, optim = mlx.core, mlx.nn, mlx.optimizers
    mlx_load, linear_to_lora_layers = load, _linear_to_lora_layers
    tree_map, tree_unflatten = mlx.utils.tree_map, mlx.utils.tree_unflatten


_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>authorization:\s*bearer\s+|[?&]token=|\btoken\s*=\s*|"
    r"\b(?:HF_TOKEN|HUGGINGFACE_HUB_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|OPENAI_API_KEY|"
    r"ANTHROPIC_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|XAI_API_KEY|AWS_ACCESS_KEY_ID|"
    r"AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|api[-_]?key)\s*[:=]\s*['\"]?)"
    r"(?P<secret>[^\s'\"&]+)|(?P<standalone>hf_[A-Za-z0-9_-]{12,}|"
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{16,}|A[KS]IA[A-Z0-9]{16}|"
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|[sp]k_live_[A-Za-z0-9]{16,}|"
    r"https?://[^\s/@:]+:[^\s/@]+@)"
)


def _scrub_secret(value):
    if isinstance(value, str):
        return _SECRET_RE.sub(lambda m: (m.group("prefix") or "") + "[redacted]", value)
    if isinstance(value, (bytes, bytearray)):
        return _scrub_secret(bytes(value).decode("utf-8", "replace"))
    if isinstance(value, dict):
        return {k: _scrub_secret(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_secret(v) for v in value]
    return value


def _child_env():
    env = dict(os.environ)
    for key in tuple(env):
        if key.endswith("_API_KEY") or key in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}:
            env.pop(key, None)
    return env


def _log_event(event: str, **fields) -> None:
    global _LOG_SECURED
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    record = _scrub_secret({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields})
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(TRAIN_LOG_PATH, flags, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        if not _LOG_SECURED:
            os.fchmod(fh.fileno(), 0o600)
            _LOG_SECURED = True
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _trust_remote_code(config) -> bool:
    return bool(config.get("trust_remote_code", False))


def _warn_remote_code(config) -> None:
    if _trust_remote_code(config):
        msg = f"WARNING: trust_remote_code=True executes model repo Python: {config.get('model')}"
        print(msg, file=sys.stderr)
        _log_event("remote_code_warning", model=config.get("model"))


def _hf_tokenizer_kwargs(config, *, local_files_only: bool) -> dict:
    return {"trust_remote_code": _trust_remote_code(config), "local_files_only": local_files_only}


def _prepare_tokenizer_source(model_source: str, model_name: str) -> str:
    if "gemma-4" not in str(model_name).lower():
        return model_source
    root = Path(model_source)
    if not root.exists():
        return model_source
    config_path = root / "tokenizer_config.json"
    if not config_path.exists():
        return model_source
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return model_source
    extras = data.get("extra_special_tokens")
    if not isinstance(extras, list):
        return model_source
    token_keys = {
        key: value for key, value in data.items()
        if key.endswith("_token") and isinstance(value, str)
    }
    mapped = {}
    used = set()
    for idx, token in enumerate(extras):
        key = next((name for name, value in token_keys.items() if value == token and name not in used), None)
        key = key or f"extra_special_token_{idx}"
        mapped[key] = token
        used.add(key)
    data["extra_special_tokens"] = mapped
    target = db.secure_private_dir(
        Path.home() / ".reinforceclaw" / "tokenizers" / hashlib.sha256(str(root).encode()).hexdigest()[:16]
    )
    for child in root.iterdir():
        if child.is_file() and child.name != "tokenizer_config.json" and not child.name.endswith((".safetensors", ".bin", ".gguf")):
            dest = target / child.name
            if dest.exists():
                continue
            try:
                os.symlink(child, dest)
            except OSError:
                shutil.copy2(child, dest)
    config = target / "tokenizer_config.json"
    config.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    db.secure_private_file(config)
    return str(target)


def _acquire_lock() -> int | None:
    db.secure_private_dir(TRAIN_LOCK_PATH.parent)
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    return fd


def _release_lock(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _set_low_priority() -> None:
    try:
        os.nice(19)
    except (OSError, PermissionError):
        pass
    if platform.system() == "Darwin":
        try:
            os.setpriority(os.PRIO_DARWIN_NONUI, 0, 0)
        except (AttributeError, OSError):
            pass


def _keep_awake() -> subprocess.Popen | None:
    """Prevent macOS sleep for the lifetime of the current process. Idempotent."""
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.Popen(
            ["caffeinate", "-isw", str(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_child_env(),
            start_new_session=True,
        )
    except FileNotFoundError:
        return None


def _os_reserve_bytes(unified_memory: bool) -> int:
    return MAC_OS_RESERVE if unified_memory else LINUX_OS_RESERVE


def _load_ratio() -> float:
    if not hasattr(os, "getloadavg"):
        return 0.0
    try:
        return os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
    except OSError:
        return 0.0


def _free_mlx() -> None:
    mlx_drain(collect_garbage=True)


def _hf_token() -> str | None:
    global _HF_TOKEN_CACHE
    if _HF_TOKEN_CACHE is not _UNSET:
        return _HF_TOKEN_CACHE
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            _HF_TOKEN_CACHE = value
            return value
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        value = token_path.read_text(encoding="utf-8").strip()
        if value:
            _HF_TOKEN_CACHE = value
            return value
    _HF_TOKEN_CACHE = None
    return None


def resolve_hf_model_source(model_name: str) -> str:
    path = Path(model_name)
    if path.exists():
        return str(path)
    cached = _HF_SOURCE_CACHE.get(model_name)
    if cached and Path(cached).exists():
        return cached
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name
    kwargs = {
        "repo_id": model_name,
        "allow_patterns": list(_HF_ALLOW_PATTERNS),
        "ignore_patterns": list(_HF_IGNORE_PATTERNS),
    }
    token = _hf_token()
    if token:
        kwargs["token"] = token
    try:
        try:
            local_path = snapshot_download(resume_download=True, **kwargs)
        except TypeError:
            local_path = snapshot_download(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"huggingface_download_failed:{_scrub_secret(str(exc))}") from exc
    _HF_SOURCE_CACHE[model_name] = local_path
    return local_path


def _feedback_source(config) -> str | None:
    value = config.get("feedback_source")
    return str(value) if value else None


def _select_backend(config):
    forced = config.get("compute_backend")
    if forced == "mlx":
        return MLXBackend()
    if forced == "cuda":
        return CUDABackend()
    return MLXBackend() if platform.system() == "Darwin" else CUDABackend()


def _has_minimum_headroom(hardware) -> bool:
    reserve = _os_reserve_bytes(hardware.unified_memory)
    available = hardware.available_memory_bytes
    if hardware.unified_memory:
        return available is None or available - reserve >= MIN_TRAINING_BUDGET
    host_available = getattr(hardware, "system_available_memory_bytes", None)
    gpu_ok = available is None or available >= MIN_TRAINING_BUDGET
    host_ok = host_available is None or host_available - reserve >= MIN_TRAINING_BUDGET
    return gpu_ok and host_ok


def _settle_backend_hardware(backend, hardware=None, rounds: int = 2, delay: float = 0.5):
    hardware = hardware or backend.hardware()
    if backend.name != "mlx":
        return hardware
    best = hardware
    best_available = hardware.available_memory_bytes or 0
    stable = 0
    for _ in range(max(1, rounds)):
        backend.clear_all()
        backend.synchronize()
        time.sleep(delay)
        current = backend.hardware()
        current_available = current.available_memory_bytes or 0
        if current_available > best_available:
            best = current
            best_available = current_available
            stable = 0
            continue
        stable += 1
        if stable >= 2:
            break
    return best


def _scheduled_window_open(config) -> bool:
    schedule = config.get("train_schedule", "auto")
    if schedule in ("manual", "auto") or not config.get("_background"):
        return True
    try:
        hour, minute = (int(part) for part in schedule.split(":", 1))
    except ValueError:
        return False
    now = datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < target:
        return False
    return now < target + timedelta(minutes=float(config.get("schedule_window_minutes", BACKGROUND_WINDOW_MINUTES)))


def _lora_init_kwargs(config) -> dict:
    """Opt-in LoRA init variants. Defaults to PEFT's gaussian. EVA needs activations,
    so we fall back to gaussian if the user enabled it but didn't supply a data loader."""
    choice = (config.get("lora_init") or "default").lower()
    if choice in ("default", "", "gaussian", "kaiming"):
        return {}
    if choice in ("pissa", "olora", "loftq"):
        return {"init_lora_weights": choice}
    if choice == "eva":
        return {"init_lora_weights": "eva"}  # caller must pass data via peft.init_lora_weights later
    return {}


def _build_optimizer_torch(torch, model, cfg, config):
    """AdamW with optional LoRA+ (higher LR on B matrices) and fused=True when CUDA
    supports it (~10-15% free speedup on modern GPUs). Ratio 0 / unset = vanilla."""
    ratio = float(config.get("lora_plus_ratio", 0.0) or 0.0)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    b_params = [p for n, p in trainable if ".lora_B" in n]

    def build(kwargs):
        if ratio <= 0 or not b_params:
            return torch.optim.AdamW([p for _, p in trainable], lr=cfg["lr"], **kwargs)
        other = [p for n, p in trainable if ".lora_B" not in n]
        return torch.optim.AdamW(
            [{"params": other, "lr": cfg["lr"]},
             {"params": b_params, "lr": cfg["lr"] * ratio}],
            **kwargs,
        )

    try:
        return build({"fused": True} if _fused_adamw_supported(torch) else {})
    except (TypeError, RuntimeError):
        return build({})


def _fused_adamw_supported(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _maybe_apply_liger(model, config) -> None:
    """Liger-Kernel fused ops (~20% throughput, ~60% CE-memory cut). Opt-in."""
    if not config.get("use_liger"):
        return
    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model)
        _log_event("liger_applied", model_type=getattr(getattr(model, "config", None), "model_type", "unknown"))
    except Exception as exc:
        _log_event("liger_skipped", detail=str(exc)[:200])


def _maybe_torch_compile(model, config):
    """torch.compile for kernel fusion. Opt-in — conflicts with some grad-checkpoint setups."""
    mode = config.get("compile_backend") or "none"
    if mode == "none":
        return model
    try:
        import torch
        compiled = torch.compile(model, mode=mode if mode in ("reduce-overhead", "max-autotune", "default") else "reduce-overhead")
        _log_event("torch_compile_applied", mode=mode)
        return compiled
    except Exception as exc:
        _log_event("torch_compile_skipped", detail=str(exc)[:200])
        return model


def _cuda_activity(backend) -> dict | None:
    device_index = getattr(getattr(backend, "device", None), "index", 0) or 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util": int(util.gpu),
            "mem_used": int(mem.used),
            "mem_total": int(mem.total),
            "source": "nvml",
        }
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        if not out:
            return None
        util, used, total = (int(part.strip()) for part in out.split(",", 2))
        mib = 1024 * 1024
        return {"gpu_util": util, "mem_used": used * mib, "mem_total": total * mib, "source": "nvidia-smi"}
    except Exception:
        pass
    try:
        hardware = backend.hardware()
        if hardware.available_memory_bytes is None:
            return None
        used = max(0, hardware.total_memory_bytes - hardware.available_memory_bytes)
        return {"gpu_util": 0, "mem_used": used, "mem_total": hardware.total_memory_bytes, "source": "fallback"}
    except Exception:
        return None


def _background_block_reason(config, backend, hardware) -> str | None:
    """Busy-system guard. Applies to scheduled nightly runs too — a 3am kick-off
    should still stand down if another process is hammering the GPU, not trample it."""
    if not config.get("_background"):
        return None
    if not _scheduled_window_open(config):
        return "outside_schedule_window"
    if _load_ratio() > float(config.get("idle_load_threshold", BACKGROUND_IDLE_LOAD)):
        return "high_cpu_load"
    reserve = _os_reserve_bytes(hardware.unified_memory)
    if hardware.unified_memory:
        available = hardware.available_memory_bytes
        if available is not None and available < reserve + int(1.5 * MIN_TRAINING_BUDGET):
            return "memory_busy"
        return None
    activity = _cuda_activity(backend)
    if activity is None:
        return "missing_cuda_idle_telemetry"
    if activity.get("source") != "fallback":
        if activity["gpu_util"] > int(config.get("cuda_idle_gpu_util", 80)):
            return "gpu_busy"
        mem_threshold = float(config.get("cuda_idle_mem_fraction", 0.80))
        if activity["mem_total"] and activity["mem_used"] / activity["mem_total"] > mem_threshold:
            return "gpu_memory_busy"
    host_available = getattr(hardware, "system_available_memory_bytes", None)
    gpu_available = hardware.available_memory_bytes
    host_busy = host_available is not None and host_available < reserve + MIN_TRAINING_BUDGET
    gpu_busy = gpu_available is not None and gpu_available < max(MIN_TRAINING_BUDGET, int(hardware.total_memory_bytes * 0.20))
    if host_busy:
        return "host_memory_busy"
    if gpu_busy:
        return "low_free_vram"
    return None


def smoke_status(config, conn):
    cfg = {**config, "_background": True}
    trainable = db.count_trainable_untrained(conn, source=_feedback_source(cfg))
    batch_min = cfg.get("batch_min", 24)
    if trainable < batch_min:
        return {"would_train": False, "reason": "below_threshold", "trainable": trainable, "batch_min": batch_min}
    compat = model_compatibility(cfg)
    if compat["ok"] is False:
        return {
            "would_train": False,
            "reason": compat["reason"],
            "detail": compat.get("detail"),
            "trainable": trainable,
            "batch_min": batch_min,
            "backend": compat.get("backend"),
        }
    try:
        backend = _select_backend(cfg)
    except Exception as exc:
        return {"would_train": False, "reason": "backend_unavailable", "detail": str(exc), "trainable": trainable, "batch_min": batch_min}
    if backend.name == "cuda":
        try:
            _torch_stack()
        except RuntimeError as exc:
            return {"would_train": False, "reason": str(exc), "backend": backend.name, "trainable": trainable, "batch_min": batch_min}
    hardware = backend.hardware()
    if not _has_minimum_headroom(hardware):
        return {"would_train": False, "reason": "insufficient_headroom", "backend": backend.name, "trainable": trainable, "batch_min": batch_min}
    block = _background_block_reason(cfg, backend, hardware)
    avail_gb, host_gb = _hardware_gbs(hardware)
    return {
        "would_train": block is None,
        "reason": block or "ready",
        "backend": backend.name,
        "trainable": trainable,
        "batch_min": batch_min,
        "schedule": cfg.get("train_schedule", "auto"),
        "available_gb": avail_gb,
        "host_available_gb": host_gb,
    }


def model_compatibility(config):
    from . import profile as _profile

    model = str(config.get("model", ""))
    mp = config.get("model_profile") if config.get("model_profile_model") == model else None
    try:
        prof = _profile.ModelProfile(**mp) if isinstance(mp, dict) else _profile.detect(model)
    except TypeError:
        prof = _profile.detect(model)
    if prof.provider == "ollama":
        return {"ok": False, "reason": "ollama_model_not_trainable",
                "detail": (f"{model} is an Ollama inference tag. You can rate its responses, "
                           "but training needs the matching local/HuggingFace weights first; "
                           "the trained adapter can then be attached back to the matching Ollama model.")}
    if prof.provider == "gguf" or "gguf" in model.lower():
        return {"ok": False, "reason": "gguf_models_not_trainable",
                "detail": (f"{model} is a GGUF/llama.cpp inference file. You can rate its responses, "
                           "but training needs the matching local/HuggingFace weights first; "
                           "the trained adapter can then be used for inference with matching GGUF/Ollama setups.")}
    if not prof.trainable:
        return {"ok": False, "reason": "cloud_api_not_trainable",
                "detail": (f"{model} is a closed API ({prof.family}) — weights are not public so RL can't update it. "
                           f"You can keep rating its responses; pick a local model as the train target (reinforceclaw init) "
                           f"and those ratings will feed into training a local adapter you control.")}
    try:
        backend = _select_backend(config)
    except Exception as exc:
        return {"ok": False, "reason": "backend_unavailable", "detail": str(exc)}

    if backend.name == "cuda":
        if prof.provider == "mlx":
            return {"ok": False, "reason": "mlx_model_on_cuda_backend", "backend": backend.name, "detail": model}
        lowered = model.lower()
        if any(tag in lowered for tag in ("awq", "gptq", "exl2")):
            return {"ok": False, "reason": "backend_specific_quantized_repo", "backend": backend.name, "detail": model}
        try:
            _torch_stack()
        except RuntimeError as exc:
            return {"ok": False, "reason": "missing_cuda_training_dependency", "backend": backend.name, "detail": str(exc)}
    return {"ok": True, "backend": backend.name}


@dataclass(frozen=True)
class TrainingPlan:
    effective_batch_size: int
    grad_accum: int
    steps: int
    memory_limit_bytes: int
    training_budget_bytes: int
    resident_model_bytes: int
    aggressive_checkpointing: bool
    busy: bool


def _slice_steps(config, total_steps: int) -> int:
    if not config.get("_background"):
        return total_steps
    limit = max(1, int(config.get("background_slice_steps", 2)))
    return min(total_steps, limit)


class AdaptiveMemoryGuard:
    def __init__(self, backend, limit_bytes: int):
        self.backend = backend
        self.limit_bytes = max(int(limit_bytes), MIN_TRAINING_BUDGET)
        self.backend.apply_limits(self.limit_bytes)

    def check(self, label: str) -> None:
        used = self.backend.current_memory_bytes()
        if used <= self.limit_bytes:
            return
        self.backend.clear_all()
        self.backend.synchronize()
        used = self.backend.current_memory_bytes()
        if used <= self.limit_bytes:
            return
        raise MemoryError(
            f"memory pressure at {label}: {used / 1e9:.2f}GB > {self.limit_bytes / 1e9:.2f}GB"
        )

    def log_step(self, label: str, step: int | None = None) -> None:
        snap = self.backend.memory_snapshot()
        _log_event(
            "memory",
            label=label,
            step=step,
            limit_gb=round(self.limit_bytes / 1e9, 3),
            **{k: round(v, 3) for k, v in snap.items()},
        )


_TRANSIENT_RESOURCE_BLOCKS = {
    "high_cpu_load",
    "gpu_busy",
    "gpu_memory_busy",
    "host_memory_busy",
    "low_free_vram",
    "memory_busy",
}


def _pressure_retry_limit(config) -> int:
    return max(0, int(config.get("pressure_retry_limit", 2)))


def _pressure_cooldown_seconds(config) -> float:
    return max(0.0, float(config.get("pressure_cooldown_s", 3.0)))


def _pressure_cooldown(backend, config, step: int, retry: int, *, reason: str | None = None, detail: str | None = None) -> None:
    _log_event(
        "pressure_retry",
        backend=backend.name,
        step=step,
        retry=retry,
        reason=reason,
        detail=detail,
    )
    backend.clear_cache()
    backend.synchronize()
    time.sleep(_pressure_cooldown_seconds(config))


def _skip(reason: str, **fields) -> dict:
    return {"status": "skipped", "reason": reason, **fields}


def _hardware_gbs(hw):
    return (
        round((hw.available_memory_bytes or 0) / 1e9, 3),
        round((getattr(hw, "system_available_memory_bytes", 0) or 0) / 1e9, 3),
    )


def _conn_db_path(conn) -> str | None:
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:
        return None
    for row in rows:
        name = row[1] if not isinstance(row, dict) else row.get("name")
        file = row[2] if not isinstance(row, dict) else row.get("file")
        if name == "main" and file:
            return str(file)
    return None


def _fresh_process_train_retry(config, conn) -> dict | None:
    if config.get("_fresh_process_retry_done"):
        return None
    db_path = _conn_db_path(conn)
    if not db_path:
        return None
    _free_mlx()
    time.sleep(3)
    cfg = {**config, "_fresh_process_retry_done": True, "_skip_lock_once": True}
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    cfg_path = TRAIN_LOG_PATH.parent / f"retry-{os.getpid()}-{time.time_ns()}.json"
    fd = os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from pathlib import Path; "
            "from reinforceclaw import db, trainer; "
            "cfg = json.loads(Path(sys.argv[1]).read_text()); "
            "conn = db.connect(Path(sys.argv[2])); "
            "print('__REINFORCECLAW_RESULT__' + json.dumps(trainer.train_result(cfg, conn)))"
        ),
        str(cfg_path),
        db_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=_child_env(),
        )
    finally:
        cfg_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        _log_event("fresh_process_retry_failed", returncode=proc.returncode, stderr=_scrub_secret(proc.stderr[-2000:]))
        return None
    lines = [line[len("__REINFORCECLAW_RESULT__"):] for line in proc.stdout.splitlines() if line.startswith("__REINFORCECLAW_RESULT__")]
    if not lines:
        return None
    result = json.loads(lines[-1])
    result.setdefault("retry_mode", "fresh_process")
    return result


def _cleanup_resume_checkpoint(config, checkpoint_path=None):
    root = _resume_dir(config).resolve()
    if checkpoint_path:
        try:
            path = Path(checkpoint_path).resolve()
        except OSError:
            return
        if root in path.parents:
            shutil.rmtree(path.parent, ignore_errors=True)
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _resume_state(conn, config):
    state = db.get_training_state(conn)
    if not state:
        _cleanup_resume_checkpoint(config)
        return None
    expected = {
        "model": config.get("model"),
        "backend": config.get("compute_backend") or ("mlx" if platform.system() == "Darwin" else "cuda"),
        "loss_fn": str(config.get("loss_fn", "mis-po")).lower(),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_target": _lora_target(config),
        "grad_accum": int(config.get("grad_accum", 4)),
        "requested_steps": int(config.get("steps", 8)),
        "lr": float(config.get("lr", 8e-6)),
        "kl_coeff": float(config.get("kl_coeff", 0.08)),
        "batch_size": int(config.get("batch_size", config.get("batch_min", 32))),
        "replay_ratio": float(config.get("replay_ratio", 0.0)),
        "traj_clip": [float(x) for x in config.get("traj_clip", [0.996, 1.001])],
        "token_clip": [float(x) for x in config.get("token_clip", [0.5, 2.0])],
    }
    for key, value in expected.items():
        if state.get(key) != value:
            db.clear_training_state(conn)
            _cleanup_resume_checkpoint(config, state.get("checkpoint_path"))
            return None
    path = state.get("checkpoint_path")
    if path and not Path(path).exists():
        db.clear_training_state(conn)
        _cleanup_resume_checkpoint(config)
        return None
    return state


def _load_batch(conn, config, plan, resume):
    if resume:
        batch_ids = list(resume["batch_ids"])
        batch = db.get_feedback_by_ids(conn, batch_ids)
        fresh_ids = list(resume["fresh_ids"])
        if len(batch) != len(batch_ids):
            raise RuntimeError("resume batch is missing feedback rows")
        by_id = {item["id"]: item for item in batch}
        missing_fresh = [i for i in fresh_ids if i not in by_id]
        if missing_fresh:
            raise RuntimeError("resume fresh batch is missing feedback rows")
        fresh = [by_id[i] for i in fresh_ids if i in by_id]
        return batch, fresh_ids, fresh
    return _build_batch(
        conn,
        plan.effective_batch_size,
        config.get("replay_ratio", 0.0),
        source=_feedback_source(config),
    )


def _paused_result(backend_name, cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_version):
    return {
        "status": "paused",
        "reason": "resume_pending",
        "remaining_steps": remaining_steps,
        "batch_size": len(batch),
        "steps": total_steps,
        "checkpoint_path": checkpoint_path,
        "parent_version": parent_version,
        "version": parent_version,
        "backend": backend_name,
        "loss_fn": cfg["loss_fn"],
        "lora_target": _lora_target(config),
    }


def _update_ema_from_fresh(ema_mean, ema_count, fresh, decay):
    """EMA tracks newly rated feedback only; replay rows already affected it when first seen."""
    for item in fresh:
        ema_count += 1
        ema_mean = decay * ema_mean + (1 - decay) * item["rating"]
    return ema_mean, ema_count


def _finalize_training(conn, config, backend, cfg, batch, fresh_ids, adapter_path,
                       ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
                       resume_checkpoint=None):
    metrics = {
        "status": "trained",
        "avg_loss": total_loss / max(total_steps, 1),
        "batch_size": len(batch),
        "steps": cfg["steps"],
        "ema_mean": ema_mean,
        "ema_count": ema_count,
        "peak_memory_gb": round(backend.peak_memory_bytes() / 1e9, 3),
        "backend": backend.name,
        "loss_fn": cfg["loss_fn"],
        "lora_target": _lora_target(config),
        "version": new_v,
        "path": adapter_path,
        "feedback_ids": fresh_ids,
    }
    db.record_training_round(conn, ema_mean, ema_count, new_v, adapter_path, parent_v, metrics, fresh_ids, clear_state=True)
    _cleanup_resume_checkpoint(config, resume_checkpoint)
    if config.get("adapter_keep"):
        active_dir = Path(adapter_path).parent.resolve()
        for path in db.cleanup_adapters(conn, keep=config["adapter_keep"]):
            old_dir = Path(path).parent.resolve()
            if old_dir != active_dir:
                shutil.rmtree(old_dir, ignore_errors=True)
    return metrics


def _training_state_payload(config, cfg, backend_name, batch, fresh_ids, resume, latest,
                            run_id, checkpoint_path, remaining_steps, total_target_steps):
    return {
        "run_id": run_id,
        "model": config["model"],
        "backend": backend_name,
        "loss_fn": str(cfg["loss_fn"]).lower(),
        "lora_rank": int(config.get("lora_rank", 8)),
        "lora_target": _lora_target(config),
        "grad_accum": int(config.get("grad_accum", 4)),
        "requested_steps": int(config.get("steps", 8)),
        "lr": float(config.get("lr", 8e-6)),
        "kl_coeff": float(config.get("kl_coeff", 0.08)),
        "batch_size": int(config.get("batch_size", config.get("batch_min", 32))),
        "replay_ratio": float(config.get("replay_ratio", 0.0)),
        "traj_clip": [float(x) for x in cfg["traj_clip"]],
        "token_clip": [float(x) for x in cfg["token_clip"]],
        "checkpoint_path": checkpoint_path,
        "batch_ids": [item["id"] for item in batch],
        "fresh_ids": fresh_ids,
        "remaining_steps": remaining_steps,
        "total_steps": total_target_steps,
        "parent_version": resume.get("parent_version") if resume else (latest["version"] if latest else None),
    }


def _build_train_cfg(config, plan: TrainingPlan):
    defaults = {
        "loss_fn": "mis-po",
        "steps": plan.steps,
        "lr": 8e-6,
        "token_clip": [0.5, 2.0],
        "traj_clip": [0.996, 1.001],
        "kl_coeff": 0.08,
        "grad_accum": plan.grad_accum,
        "grad_clip": 1.0,
        "ema_decay": 0.99,
        "pos_weight": 1.0,
        "adv_clip": 2.0,
        "max_passes": 1.0,
        "adapter_keep": 20,
    }
    return {key: config.get(key, value) for key, value in defaults.items()}


def _tighten_small_batch_cfg(cfg: dict, batch_size: int) -> dict:
    if batch_size <= 2:
        cfg = {**cfg, "grad_accum": 1}
    elif batch_size <= 4:
        cfg = {**cfg, "grad_accum": min(cfg["grad_accum"], 2)}
    return cfg


def _trajectory_scale_mlx(delta_mean, cfg):
    low, high = cfg["traj_clip"]
    eps = mx.array(1e-6)
    return mx.minimum(mx.array(1.0), mx.minimum(delta_mean / mx.maximum(mx.array(low), eps),
                                                mx.array(high) / mx.maximum(delta_mean, eps)))


def _trajectory_scale_torch(delta_mean, cfg, torch):
    low, high = cfg["traj_clip"]
    eps = torch.tensor(1e-6, device=delta_mean.device, dtype=delta_mean.dtype)
    one = torch.tensor(1.0, device=delta_mean.device, dtype=delta_mean.dtype)
    return torch.minimum(one, torch.minimum(delta_mean / torch.clamp(torch.tensor(low, device=delta_mean.device, dtype=delta_mean.dtype), min=1e-6),
                                            torch.tensor(high, device=delta_mean.device, dtype=delta_mean.dtype) / torch.maximum(delta_mean, eps)))


def _safe_total_memory_bytes(hardware) -> int:
    cap = getattr(hardware, "recommended_working_set_bytes", None)
    return min(hardware.total_memory_bytes, cap) if cap else hardware.total_memory_bytes


def _preload_limit_bytes(hardware, reserve_bytes: int) -> int:
    if hardware.unified_memory:
        available = hardware.available_memory_bytes or hardware.total_memory_bytes
        safe_total = _safe_total_memory_bytes(hardware)
        safe_cap = min(
            max(available - reserve_bytes // 2, MIN_TRAINING_BUDGET),
            max(safe_total - reserve_bytes, MIN_TRAINING_BUDGET),
        )
        return int(safe_cap)
    available = hardware.available_memory_bytes or hardware.total_memory_bytes
    return max(MIN_TRAINING_BUDGET, min(int(hardware.total_memory_bytes * 0.85), int(available * 0.9)))


def _training_budget_bytes(hardware, reserve_bytes: int, model_bytes: int) -> int:
    if hardware.unified_memory:
        available = hardware.available_memory_bytes or hardware.total_memory_bytes
        free_budget = max(0, available - reserve_bytes)
        cap_budget = max(0, _safe_total_memory_bytes(hardware) - reserve_bytes - max(model_bytes, 0))
        return min(free_budget, cap_budget)

    device_reserve = max(1 * GB, int(hardware.total_memory_bytes * 0.10))
    device_available = hardware.available_memory_bytes or hardware.total_memory_bytes
    device_budget = max(0, device_available - device_reserve - max(model_bytes, 0))
    host_available = getattr(hardware, "system_available_memory_bytes", None) or getattr(hardware, "system_total_memory_bytes", 0)
    host_budget = max(0, host_available - reserve_bytes)
    return min(device_budget, host_budget) if host_budget else device_budget


def _next_adapter_version(conn) -> int:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM adapters").fetchone()
    return int(row["version"]) + 1


def _plan_strategy(config, hardware, model_bytes: int) -> TrainingPlan | None:
    reserve_bytes = _os_reserve_bytes(hardware.unified_memory)
    busy = _load_ratio() > float(config.get("idle_load_threshold", BACKGROUND_IDLE_LOAD))
    available = hardware.available_memory_bytes
    if available is not None and available < int(reserve_bytes * 1.25):
        busy = True

    budget = _training_budget_bytes(hardware, reserve_bytes, model_bytes)
    if budget < MIN_TRAINING_BUDGET:
        return None

    batch_cap = max(1, int(config.get("batch_size", config.get("batch_min", 24))))
    base_accum = max(1, config.get("grad_accum", 4))
    base_steps = max(1, config.get("steps", 8))
    preload = _preload_limit_bytes(hardware, reserve_bytes)

    if budget < 4 * GB:
        batch_cap = min(batch_cap, 1)
        grad_accum = 1
        steps = min(base_steps, 2)
        aggressive = True
    elif budget < 8 * GB:
        batch_cap = min(batch_cap, 4)
        grad_accum = min(base_accum, 2)
        steps = min(base_steps, 3)
        aggressive = True
    else:
        batch_cap = min(batch_cap, 8 if budget < 12 * GB else batch_cap)
        grad_accum = base_accum
        steps = base_steps
        aggressive = False

    floor = model_bytes + min(budget, MIN_TRAINING_BUDGET)
    limit = max(floor, min(preload, model_bytes + max(MIN_TRAINING_BUDGET, int(budget * 0.9))))
    return TrainingPlan(
        effective_batch_size=max(1, batch_cap),
        grad_accum=max(1, grad_accum),
        steps=max(1, steps),
        memory_limit_bytes=max(limit, MIN_TRAINING_BUDGET),
        training_budget_bytes=budget,
        resident_model_bytes=max(0, model_bytes),
        aggressive_checkpointing=aggressive,
        busy=busy,
    )


def _degrade_plan(plan: TrainingPlan | None) -> TrainingPlan | None:
    if plan is None:
        return None
    if plan.effective_batch_size == 1 and plan.grad_accum == 1:
        return None
    floor = plan.resident_model_bytes + MIN_TRAINING_BUDGET
    return replace(
        plan,
        effective_batch_size=max(1, plan.effective_batch_size // 2),
        grad_accum=max(1, plan.grad_accum // 2),
        steps=max(1, min(plan.steps, 2)),
        memory_limit_bytes=max(floor, int(plan.memory_limit_bytes * 0.9)),
        aggressive_checkpointing=True,
        busy=True,
    )


def _build_batch(conn, batch_size: int, replay_ratio: float, source: str | None = None):
    batch_size = max(1, int(batch_size))
    replay_target = min(batch_size, max(0, int(round(batch_size * max(0.0, float(replay_ratio))))))
    fresh_target = max(0, batch_size - replay_target)
    all_fresh = db.get_untrained(conn, limit=batch_size, source=source)
    all_replay = db.get_replay(conn, limit=batch_size, source=source) if replay_target else []
    fresh = all_fresh[:fresh_target] if fresh_target else []
    replay = all_replay[:replay_target] if replay_target else []
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        fresh.extend(all_fresh[len(fresh):len(fresh) + missing])
    missing = batch_size - (len(fresh) + len(replay))
    if missing > 0:
        replay.extend(all_replay[len(replay):len(replay) + missing])
    return fresh + replay, [item["id"] for item in fresh], fresh


def _release_backend_memory(backend) -> None:
    backend.clear_all()
    gc.collect()
    try:
        backend.synchronize()
    except Exception:
        pass
    if backend.name == "mlx":
        time.sleep(0.25)


def _lora_target(config) -> str:
    return str(config.get("lora_target", "attention")).lower()


_ATTENTION_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "c_attn", "c_proj"}
_MLP_LEAVES = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}


def _strict_target_selection(target: str) -> bool:
    # This is intentionally strict for all non-"all" LoRA targets.
    # It matters most for MoE models because falling back to every linear layer can
    # accidentally LoRA expert/router-adjacent weights, but the same silent fallback
    # is also unsafe on dense models.
    return target != "all"


def _mlx_lora_keys(model, target: str):
    if target == "all":
        return None
    allowed = _ATTENTION_LEAVES if target == "attention" else _ATTENTION_LEAVES | _MLP_LEAVES
    keys = set()
    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None) or []
    for layer in layers:
        for path, _module in layer.named_modules():
            if path.rsplit(".", 1)[-1] in allowed:
                keys.add(path)
    return keys


def _apply_lora(model, rank: int, target: str = "attention"):
    _ensure_mlx()
    model.freeze()
    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None) or []
    cfg = {"rank": rank, "scale": rank, "dropout": 0.0}
    keys = _mlx_lora_keys(model, target)
    if _strict_target_selection(target) and not keys:
        raise RuntimeError(
            f"no_{target}_modules_found_for_lora:strict_targeting_enabled"
        )
    if keys is not None:
        cfg["keys"] = keys
    linear_to_lora_layers(
        model,
        len(layers),
        cfg,
    )
    return model


def _disable_lora(model):
    saved = {
        name: param
        for name, param in nn.utils.tree_flatten(model.trainable_parameters())
        if "lora_b" in name.lower()
    }
    if not saved:
        raise RuntimeError("lora_disable_failed:no_lora_b_weights")
    zeros = {key: mx.zeros_like(value) for key, value in saved.items()}
    model.load_weights(list(zeros.items()), strict=False)
    mx.eval(model.parameters())
    return saved


def _enable_lora(model, saved_weights):
    if saved_weights:
        model.load_weights(list(saved_weights.items()), strict=False)
        mx.eval(model.parameters())


def _enable_grad_checkpoint_mlx(model):
    if not hasattr(mx, "checkpoint"):
        return lambda: None
    try:
        layers = getattr(getattr(model, "model", None), "layers", None) or getattr(model, "layers", None)
        if not layers:
            return lambda: None
        layer_cls = type(layers[0])
        original_call = getattr(layer_cls, "__reinforceclaw_original_call__", layer_cls.__call__)
        if getattr(layer_cls, "_reinforceclaw_checkpointed", False):
            return lambda: None

        def checkpointed_call(self, *args, **kwargs):
            def inner(params, *inner_args, **inner_kwargs):
                self.update(params)
                return original_call(self, *inner_args, **inner_kwargs)

            return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

        layer_cls.__reinforceclaw_original_call__ = original_call
        layer_cls.__call__ = checkpointed_call
        layer_cls._reinforceclaw_checkpointed = True
        def restore():
            if getattr(layer_cls, "_reinforceclaw_checkpointed", False):
                layer_cls.__call__ = layer_cls.__reinforceclaw_original_call__
                layer_cls._reinforceclaw_checkpointed = False
        return restore
    except Exception:
        return lambda: None


def _build_chat_messages(ctx, prompt):
    msgs = ctx.get("messages") if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list) else None
    if msgs is not None:
        return msgs
    msgs = []
    if isinstance(ctx, dict) and "system" in ctx:
        msgs.append({"role": "system", "content": ctx["system"]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _chat_text_pair(tokenizer, item):
    """Return (prompt_text, full_text) for an item, or None if tokenizer has no chat template."""
    prompt, response = item["prompt"], item["response"]
    ctx = _context_dict(item)
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = _build_chat_messages(ctx, prompt)
        prompt_text = _apply_chat_template(tokenizer, msgs, add_generation_prompt=True, tokenize=False)
        full_text = _apply_chat_template(
            tokenizer, msgs + [{"role": "assistant", "content": response}],
            add_generation_prompt=False, tokenize=False,
        )
        return prompt_text, full_text
    if isinstance(ctx, dict) and isinstance(ctx.get("messages"), list):
        prompt_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ctx["messages"])
        return prompt_text, prompt_text + "\nAssistant: " + response
    return prompt, prompt + response


def _tokenize_mlx(tokenizer, item):
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    prompt_text, full_text = _chat_text_pair(tokenizer, item)
    prompt_ids = _plain_token_ids(tokenizer, prompt_text)
    full_ids = _plain_token_ids(tokenizer, full_text)
    offset = max(0, len(full_ids) - max_seq_len)
    full_ids = full_ids[offset:]
    return {
        "input_ids": mx.array(full_ids),
        "response_start": max(1, len(prompt_ids) - offset),
        "rating": item["rating"],
        "id": item["id"],
    }


def _compute_logprobs_mlx(model, input_ids):
    logits = model(input_ids[None, :]).squeeze(0)
    lp = nn.log_softmax(logits, axis=-1)
    tok_lp = mx.take_along_axis(lp[:-1], input_ids[1:, None], axis=-1).squeeze(-1)
    mx.eval(tok_lp)
    return tok_lp


def _fallback_chat_text(messages, *, add_generation_prompt=False, tokenizer=None):
    parts = [f"{str(m.get('role', 'user')).capitalize()}: {str(m.get('content', ''))}" for m in messages]
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts)


def _apply_chat_template(tokenizer, messages, *, add_generation_prompt=False, tokenize=True):
    kwargs = {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize}
    if add_generation_prompt:
        kwargs["enable_thinking"] = False
    def _missing_template(exc: Exception) -> bool:
        return "chat_template is not set" in str(exc)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except ValueError as exc:
            if not _missing_template(exc):
                raise
        except ImportError:
            pass
    except ValueError as exc:
        if not _missing_template(exc):
            raise
    except ImportError:
        pass
    fallback = _fallback_chat_text(messages, add_generation_prompt=add_generation_prompt, tokenizer=tokenizer)
    return _plain_token_ids(tokenizer, fallback) if tokenize else fallback


def _context_dict(item):
    context = item.get("rollout_context") or item.get("context")
    if not context:
        return None
    try:
        return json.loads(context)
    except (json.JSONDecodeError, TypeError):
        return None


def _plain_token_ids(tokenizer, text):
    if isinstance(text, (list, tuple)):
        return [int(tok) for tok in text]
    if hasattr(tokenizer, "__call__"):
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids", encoded)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(tok) for tok in ids]
        except Exception:
            pass
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        return [int(tok) for tok in ids]
    raise TypeError("tokenizer cannot produce plain token ids")


def _behavior_logprobs(item, length: int, xp_name: str, torch_module=None):
    ctx = _context_dict(item)
    if not isinstance(ctx, dict):
        return None
    values = ctx.get("behavior_logprobs")
    if not isinstance(values, list):
        return None
    if len(values) != length:
        return None
    if xp_name == "mlx":
        return mx.array(values)
    device = item["input_ids"].device
    return torch_module.tensor(values, device=device, dtype=torch_module.float32)


def _effective_steps(planned_steps: int, batch_size: int, grad_accum: int, max_passes: float) -> int:
    budgeted = math.ceil(max(1, batch_size) * max(max_passes, 0.25) / max(1, grad_accum))
    return max(1, min(planned_steps, budgeted))


def _scalar_advantage(rating: int, ema_mean: float, cfg) -> float:
    adv = _raw_advantage(rating, ema_mean, cfg)
    clip = float(cfg.get("adv_clip", 2.0))
    return max(-clip, min(clip, adv))


def _raw_advantage(rating: int, ema_mean: float, cfg) -> float:
    adv = float(rating - ema_mean)
    return adv * float(cfg.get("pos_weight", 1.0)) if rating > 0 else adv


def _attach_advantages(items, ema_mean, cfg):
    vals = [_raw_advantage(item["rating"], ema_mean, cfg) for item in items]
    if len(vals) > 1 and cfg.get("adv_norm", True):
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        vals = [(v - mean) / (std + 1e-6) for v in vals]
    clip = float(cfg.get("adv_clip", 2.0))
    for item, value in zip(items, vals):
        item["advantage"] = max(-clip, min(clip, value))
    return items


def _aggregate_diag(samples: list[dict]) -> dict:
    """Average per-sample diagnostics into per-step scalars for logging."""
    if not samples:
        return {}
    keys = samples[0].keys()
    out = {}
    for k in keys:
        vals = [s[k] for s in samples]
        out[k] = round(sum(vals) / len(vals), 6)
    # advantage_std is meaningful only across a batch — override the per-sample 0s.
    adv = [s["advantage_mean"] for s in samples]
    if len(adv) > 1:
        m = sum(adv) / len(adv)
        out["advantage_std"] = round((sum((v - m) ** 2 for v in adv) / len(adv)) ** 0.5, 6)
    return out


def _make_loss_fn_mlx(model, tokenized, ref_logprobs, cfg, ema_mean, diag_sink=None):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()
    traj_lo, traj_hi = cfg["traj_clip"]

    def loss_fn(idx: int):
        item = tokenized[idx]
        ids, start, rating = item["input_ids"], item["response_start"], item["rating"]
        ref = ref_logprobs[idx]
        logits = model(ids[None, :]).squeeze(0)
        new_lp = mx.take_along_axis(
            nn.log_softmax(logits, axis=-1)[:-1],
            ids[1:, None],
            axis=-1,
        ).squeeze(-1)
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        if int(new_r.shape[0]) == 0 or int(ref_r.shape[0]) == 0:
            return mx.sum(logits) * 0.0
        adv_scalar = item.get("advantage", _scalar_advantage(rating, ema_mean, cfg))
        adv = mx.array(adv_scalar, dtype=new_r.dtype)

        # Diagnostics (no-grad, scalar floats) — captured via sink so the trainer
        # can aggregate them per step without changing loss_fn's signature.
        if diag_sink is not None:
            delta = new_r - ref_r
            traj_val = float(mx.exp(mx.mean(delta)).item())
            traj_pass = 1.0 if traj_lo <= traj_val <= traj_hi else 0.0
            # token_mask_rate = fraction of ratio values inside [tc_lo, tc_hi].
            ratio_for_diag = mx.exp(delta)
            inside = (ratio_for_diag >= tc_lo) & (ratio_for_diag <= tc_hi)
            token_mask_rate = float(mx.mean(inside.astype(new_r.dtype)).item())
            diag_sink.append({
                "policy_kl": float(mx.mean(delta).item()),
                "ratio_mean": float(mx.mean(ratio_for_diag).item()),
                "ratio_p5": float(mx.min(ratio_for_diag).item()),   # approx — mlx has no percentile
                "ratio_p95": float(mx.max(ratio_for_diag).item()),  # approx
                "advantage_mean": adv_scalar,
                "advantage_std": 0.0,  # scalar per sample — std computed at batch level
                "pct_zero_adv": 1.0 if adv_scalar == 0.0 else 0.0,
                "traj_gate_pass_rate": traj_pass,
                "token_mask_rate": token_mask_rate,
            })

        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = mx.stop_gradient(adv - kl_c * per_token_kl)
            return -mx.mean(new_r * adjusted)
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "mlx") if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = ref_r

        ratio = mx.exp(new_r - behavior_r)
        clipped_ratio = mx.clip(ratio, tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -mx.mean(mx.minimum(surr_1, surr_2) if rating > 0 else mx.maximum(surr_1, surr_2))
        kl = mx.exp(ref_r - new_r) - (ref_r - new_r) - 1
        traj = mx.exp(mx.mean(new_r - ref_r))
        return actor * _trajectory_scale_mlx(traj, cfg) + kl_c * mx.mean(kl)

    return loss_fn


def _torch_stack():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(f"missing_cuda_training_dependency:{exc}") from exc

    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model


def _optimizer_file(adapter_path: str | Path) -> Path:
    return Path(adapter_path).parent / "optimizer.safetensors"


def _save_mlx_optimizer(opt, adapter_path) -> None:
    mx.save_safetensors(str(_optimizer_file(adapter_path)), dict(nn.utils.tree_flatten(opt.state)))


def _load_mlx_optimizer(opt, adapter_path) -> None:
    path = _optimizer_file(adapter_path)
    if path.exists():
        opt.state = tree_unflatten(list(mx.load(str(path)).items()))


def _save_torch_optimizer(torch, opt, adapter_path) -> None:
    torch.save(opt.state_dict(), str(_optimizer_file(adapter_path).with_suffix(".pt")))


def _load_torch_optimizer(torch, opt, adapter_path) -> None:
    path = _optimizer_file(adapter_path).with_suffix(".pt")
    if not path.exists():
        return
    try:
        opt.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    except (TypeError, RuntimeError, ValueError) as exc:
        _log_event("optimizer_resume_skipped", backend="cuda", detail=str(exc)[:200])


def _save_partial_cuda(model, config, run_id: str, torch=None, optimizer=None) -> str:
    target_dir = _resume_checkpoint_dir(config, run_id)
    model.save_pretrained(target_dir, safe_serialization=True)
    adapter = str(next((path for path in target_dir.glob("*.safetensors")), target_dir / "adapter_model.bin"))
    if torch is not None and optimizer is not None:
        _save_torch_optimizer(torch, optimizer, adapter)
    return adapter


def _load_torch_adapter_weights(torch, model, adapter_path: str | Path) -> None:
    adapter_path = Path(adapter_path)
    if adapter_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(adapter_path), device="cpu")
    else:
        root = (Path.home() / ".reinforceclaw" / "adapters").resolve()
        if root not in adapter_path.resolve().parents:
            raise RuntimeError("Refusing non-safetensors adapter outside ~/.reinforceclaw/adapters")
        try:
            state_dict = torch.load(str(adapter_path), map_location="cpu", weights_only=True)
        except TypeError:
            raise RuntimeError("Non-safetensors adapters require PyTorch with weights_only=True")
    try:
        from peft.utils.save_and_load import set_peft_model_state_dict

        try:
            set_peft_model_state_dict(model, state_dict, adapter_name="default", ignore_mismatched_sizes=True)
            return
        except TypeError:
            set_peft_model_state_dict(model, state_dict, adapter_name="default")
            return
        except RuntimeError as exc:
            _log_event("adapter_load_failed", detail=str(exc)[:300], path=str(adapter_path))
            raise
    except ImportError:
        pass
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    if not compatible:
        raise RuntimeError(f"adapter_no_compatible_weights:{adapter_path}")
    if len(compatible) != len(state_dict):
        _log_event("adapter_partial_load", path=str(adapter_path), loaded=len(compatible), total=len(state_dict))
    model.load_state_dict(compatible, strict=False)


def _torch_target_modules(torch, model, target: str = "attention"):
    common = set()
    exact = set()
    exact_preferred = set()
    seen = set()
    # Handle "all" and "all_linear" to include ALL linear modules (including experts)
    if target in ("all", "all_linear"):
        preferred = None  # None means all linear modules
    elif target == "attention":
        preferred = _ATTENTION_LEAVES
    else:
        preferred = _ATTENTION_LEAVES | _MLP_LEAVES
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        leaf = parts[-1]
        parent_leaf = parts[-2] if len(parts) >= 2 else ""
        seen.add(leaf)
        if parent_leaf:
            seen.add(parent_leaf)
        if any(part in {"vision", "vision_tower"} for part in parts):
            continue
        # If preferred is None (target="all" or "all_linear"), accept all linear modules
        if preferred is None:
            exact.add(name)
        elif leaf in preferred:
            common.add(leaf)
            exact.add(name)
        elif leaf == "linear" and parent_leaf in preferred:
            exact_preferred.add(name)
    target_modules = exact_preferred or exact or common
    if _strict_target_selection(target) and not target_modules:
        raise RuntimeError(
            f"no_{target}_modules_found_for_lora:strict_targeting_enabled:{','.join(sorted(seen))}"
        )
    return sorted(target_modules)


def _tokenize_torch(tokenizer, item, torch_device):
    max_seq_len = max(128, int(item.get("max_seq_len") or 2048))
    prompt_text, full_text = _chat_text_pair(tokenizer, item)
    prompt_batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    full_batch = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_batch["input_ids"][0]
    full_ids = full_batch["input_ids"][0]
    offset = max(0, int(full_ids.shape[0]) - max_seq_len)
    full_ids = full_ids[offset:]
    tokenized = {
        "input_ids": full_ids,
        "response_start": max(1, int(prompt_ids.shape[0]) - offset),
        "rating": item["rating"],
        "id": item["id"],
    }
    for key, value in full_batch.items():
        if key == "input_ids" or not hasattr(value, "shape"):
            continue
        tokenized[key] = value[0][offset:]
    return tokenized


def _torch_move_item(item, device):
    moved = {}
    for key, value in item.items():
        moved[key] = value if getattr(value, "device", None) == device else (
            value.to(device, non_blocking=True) if hasattr(value, "to") else value
        )
    return moved


def _torch_model_type(model):
    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ]
    for cfg in candidates:
        model_type = getattr(cfg, "model_type", None)
        if model_type:
            return str(model_type).lower()
    return None


def _torch_forward_kwargs(model, item, torch):
    ids = item["input_ids"].unsqueeze(0)
    kwargs = {"input_ids": ids}
    attention_mask = item.get("attention_mask")
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.unsqueeze(0)
    for key, value in item.items():
        if not key.startswith("mm_") or key in kwargs or not hasattr(value, "unsqueeze"):
            continue
        kwargs[key] = value.unsqueeze(0)
    if (_torch_model_type(model) or "").startswith("gemma") and "mm_token_type_ids" not in kwargs:
        kwargs["mm_token_type_ids"] = torch.zeros_like(ids)
    return kwargs


def _compute_logprobs_torch(model, item, torch):
    with torch.no_grad():
        device = next(model.parameters()).device
        device_item = _torch_move_item(item, device)
        logits = model(**_torch_forward_kwargs(model, device_item, torch)).logits.squeeze(0)
        lp = torch.log_softmax(logits, dim=-1)
        ids = device_item["input_ids"]
        return lp[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)



def load_model(model_name, lora_rank=16, adapter_path=None):
    _ensure_mlx()
    model, tokenizer = mlx_load(model_name)
    target = "attention"
    if adapter_path:
        cfg_path = Path(adapter_path).with_name("adapter_config.json")
        if cfg_path.exists():
            try:
                saved_target = json.loads(cfg_path.read_text()).get("target_modules", target)
                target = saved_target if isinstance(saved_target, str) else "all"
            except Exception:
                pass
    model = _apply_lora(model, rank=lora_rank, target=target)
    if adapter_path and Path(adapter_path).exists():
        model.load_weights(adapter_path, strict=False)
    model.eval()
    return model, tokenizer


def publish_gate(config, adapter_path):
    return {"ok": True, "reason": "no_eval_gate"}


def _write_mlx_adapter_dir(target_dir: Path, model, config) -> str:
    _ensure_mlx()
    adapter_file = target_dir / "adapter.safetensors"
    portable_file = target_dir / "adapter_model.safetensors"
    lora_weights = {name: value for name, value in nn.utils.tree_flatten(model.trainable_parameters()) if "lora" in name.lower()}
    mx.save_safetensors(str(adapter_file), lora_weights)
    mx.save_safetensors(str(portable_file), lora_weights)
    _rank = config.get("lora_rank", 8)
    (target_dir / "adapter_config.json").write_text(json.dumps({
        "r": _rank,
        "lora_alpha": config.get("lora_alpha", _rank),
        "base_model_name_or_path": config["model"],
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": _lora_target(config),
        "base_model": config["model"],
    }))
    return str(adapter_file)


def _save_partial_mlx(model, config, run_id: str, optimizer=None) -> str:
    adapter = _write_mlx_adapter_dir(_resume_checkpoint_dir(config, run_id), model, config)
    if optimizer is not None:
        _save_mlx_optimizer(optimizer, adapter)
    return adapter


def _make_loss_fn_torch(model, tokenized, ref_logprobs, cfg, ema_mean, torch, diag_sink=None):
    tc_lo, tc_hi = cfg["token_clip"]
    kl_c = cfg["kl_coeff"]
    loss_name = str(cfg.get("loss_fn", "mis-po")).lower()
    traj_lo, traj_hi = cfg["traj_clip"]
    device = next(model.parameters()).device

    def loss_fn(idx: int):
        item = _torch_move_item(tokenized[idx], device)
        ids, start, rating = item["input_ids"], item["response_start"], item["rating"]
        ref = ref_logprobs[idx]
        logits = model(**_torch_forward_kwargs(model, item, torch)).logits.squeeze(0)
        new_lp = torch.log_softmax(logits, dim=-1)[:-1].gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1)
        new_r, ref_r = new_lp[start - 1:], ref[start - 1:]
        if new_r.numel() == 0 or ref_r.numel() == 0:
            return logits.sum() * 0.0
        adv_scalar = item.get("advantage", _scalar_advantage(rating, ema_mean, cfg))
        adv = torch.tensor(adv_scalar, device=ids.device, dtype=new_r.dtype)

        if diag_sink is not None:
            with torch.no_grad():
                delta = new_r - ref_r
                ratio_for_diag = torch.exp(delta)
                traj_val = float(torch.exp(delta.mean()).item())
                traj_pass = 1.0 if traj_lo <= traj_val <= traj_hi else 0.0
                inside = ((ratio_for_diag >= tc_lo) & (ratio_for_diag <= tc_hi)).to(new_r.dtype)
                try:
                    p5 = float(torch.quantile(ratio_for_diag.float(), 0.05).item())
                    p95 = float(torch.quantile(ratio_for_diag.float(), 0.95).item())
                except Exception:
                    p5, p95 = float(ratio_for_diag.min().item()), float(ratio_for_diag.max().item())
                diag_sink.append({
                    "policy_kl": float(delta.mean().item()),
                    "ratio_mean": float(ratio_for_diag.mean().item()),
                    "ratio_p5": p5,
                    "ratio_p95": p95,
                    "advantage_mean": adv_scalar,
                    "advantage_std": 0.0,
                    "pct_zero_adv": 1.0 if adv_scalar == 0.0 else 0.0,
                    "traj_gate_pass_rate": traj_pass,
                    "token_mask_rate": float(inside.mean().item()),
                })

        if loss_name == "reinforce++":
            per_token_kl = new_r - ref_r
            adjusted = (adv - kl_c * per_token_kl).detach()
            return -(new_r * adjusted).mean()
        behavior_r = _behavior_logprobs(item, int(new_r.shape[0]), "torch", torch_module=torch) if loss_name == "mis-po" else ref_r
        if behavior_r is None:
            behavior_r = ref_r

        ratio = torch.exp(new_r - behavior_r)
        clipped_ratio = ratio.clamp(tc_lo, tc_hi)
        surr_1 = ratio * adv
        surr_2 = clipped_ratio * adv
        actor = -(torch.minimum(surr_1, surr_2) if rating > 0 else torch.maximum(surr_1, surr_2)).mean()
        kl = torch.exp(ref_r - new_r) - (ref_r - new_r) - 1
        traj = torch.exp((new_r - ref_r).mean())
        return actor * _trajectory_scale_torch(traj, cfg, torch) + kl_c * kl.mean()

    return loss_fn


def _attempt_train_mlx(config, conn, backend, hardware, attempt: int):
    _ensure_mlx()
    reserve_bytes = _os_reserve_bytes(True)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn)
    resume = _resume_state(conn, config)
    guard.check("before_model_load")
    try:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        restore_checkpoint = lambda: None
        model, tokenizer = mlx_load(config["model"])
        model = _apply_lora(model, rank=config.get("lora_rank", 8), target=_lora_target(config))
        if resume and resume.get("checkpoint_path"):
            model.load_weights(resume["checkpoint_path"], strict=False)
        elif latest and Path(latest["path"]).exists():
            model.load_weights(latest["path"], strict=False)
        model.eval()

        model_bytes = max(backend.current_memory_bytes(), backend.active_memory_bytes())
        hardware = _settle_backend_hardware(backend, backend.hardware(), rounds=2, delay=0.5)
        plan = _plan_strategy(config, hardware, model_bytes)
        for _ in range(attempt):
            plan = _degrade_plan(plan)
        if plan is None:
            _log_event("skip", backend="mlx", reason="insufficient_budget", model_gb=round(model_bytes / 1e9, 3))
            return _skip("insufficient_budget", backend="mlx", model_gb=round(model_bytes / 1e9, 3))

        guard = AdaptiveMemoryGuard(backend, plan.memory_limit_bytes)
        guard.log_step("post_load")

        batch, fresh_ids, fresh = _load_batch(conn, config, plan, resume)
        if not batch:
            return _skip("empty_batch", backend="mlx")

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        tokenized = _attach_advantages(
            [_tokenize_mlx(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}) for item in batch],
            db.get_ema(conn)[0], cfg,
        )
        saved_lora = _disable_lora(model)
        ref_lps = [_compute_logprobs_mlx(model, item["input_ids"]) for item in tokenized]
        mx.eval(*ref_lps)
        _enable_lora(model, saved_lora)
        restore_checkpoint = _enable_grad_checkpoint_mlx(model) if plan.aggressive_checkpointing else (lambda: None)
        model.train()

        try:
            ema_mean, ema_count = db.get_ema(conn)
            total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
            remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
            cfg["steps"] = min(remaining_steps, _slice_steps(config, remaining_steps))
            diag_sink: list[dict] = []
            loss_fn = _make_loss_fn_mlx(model, tokenized, ref_lps, cfg, ema_mean, diag_sink=diag_sink)
            if config.get("mlx_compile"):
                try:
                    loss_fn = mx.compile(loss_fn)
                    _log_event("mlx_compile_applied")
                except Exception as exc:
                    _log_event("mlx_compile_skipped", detail=str(exc)[:200])
            vg = nn.value_and_grad(model, loss_fn)
            opt = optim.Adam(learning_rate=cfg["lr"])
            if resume and resume.get("checkpoint_path"):
                _load_mlx_optimizer(opt, resume["checkpoint_path"])
            backend.reset_peak_memory()

            total_loss = 0.0
            total_steps = 0
            for step in range(cfg["steps"]):
                for retry in range(_pressure_retry_limit(config) + 1):
                    try:
                        current = backend.hardware()
                        block = _background_block_reason(config, backend, current)
                        if block:
                            if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                                _pressure_cooldown(backend, config, step, retry, reason=block)
                                continue
                            _log_event("skip", backend="mlx", reason=block, step=step)
                            return _skip(block, backend="mlx", step=step)
                        if not _has_minimum_headroom(current):
                            raise MemoryError("memory pressure while training")
                        guard.check(f"before_step_{step}")
                        acc_grads = None
                        step_loss = 0.0
                        diag_sink.clear()
                        for micro in range(cfg["grad_accum"]):
                            idx = (step * cfg["grad_accum"] + micro) % len(tokenized)
                            loss, grads = vg(idx)
                            mx.eval(loss)
                            step_loss += loss.item()
                            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)

                        acc_grads = tree_map(lambda grad: grad / cfg["grad_accum"], acc_grads)
                        flat = [grad for _, grad in nn.utils.tree_flatten(acc_grads)]
                        if flat:
                            norm = mx.sqrt(sum(mx.sum(grad * grad) for grad in flat))
                            scale = mx.minimum(mx.array(cfg["grad_clip"]) / (norm + 1e-6), mx.array(1.0))
                            acc_grads = tree_map(lambda grad: grad * scale, acc_grads)
                            grad_norm = float(norm.item())
                        else:
                            grad_norm = 0.0

                        opt.update(model, acc_grads)
                        mx.eval(model.parameters(), opt.state)
                        total_loss += step_loss / cfg["grad_accum"]
                        total_steps += 1
                        diag_fields = _aggregate_diag(diag_sink)
                        _log_event(
                            "opt_step",
                            backend="mlx",
                            step=step,
                            loss=round(step_loss / cfg["grad_accum"], 6),
                            grad_norm=round(grad_norm, 6),
                            **diag_fields,
                        )
                        backend.clear_cache()
                        guard.log_step("step", step)
                        guard.check(f"after_step_{step}")
                        break
                    except MemoryError as exc:
                        if retry >= _pressure_retry_limit(config):
                            raise
                        _pressure_cooldown(backend, config, step, retry, reason="memory_pressure", detail=str(exc))
        finally:
            restore_checkpoint()

        remaining_steps = max(0, remaining_steps - total_steps)
        if remaining_steps > 0:
            if total_steps == 0:
                return _skip("no_steps_completed", backend="mlx")
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}"
            checkpoint_path = _save_partial_mlx(model, config, run_id, opt)
            parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
            db.save_training_state(conn, _training_state_payload(
                config, cfg, backend.name, batch, fresh_ids, resume, latest,
                run_id, checkpoint_path, remaining_steps, total_target_steps,
            ))
            return _paused_result("mlx", cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_v)

        ema_mean, ema_count = _update_ema_from_fresh(ema_mean, ema_count, fresh, cfg["ema_decay"])
        parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        try:
            _write_mlx_adapter_dir(temp_dir, model, config)
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return _finalize_training(
            conn, config, backend, cfg, batch, fresh_ids,
            str(save_dir / "adapter.safetensors"),
            ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
            resume.get("checkpoint_path") if resume else None,
        )
    finally:
        model = tokenizer = tokenized = ref_lps = saved_lora = None
        _release_backend_memory(backend)


def _attempt_train_cuda(config, conn, backend, hardware, attempt: int):
    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, PeftModel, get_peft_model = _torch_stack()
    reserve_bytes = _os_reserve_bytes(False)
    guard = AdaptiveMemoryGuard(backend, _preload_limit_bytes(hardware, reserve_bytes))
    latest = db.latest_adapter(conn)
    resume = _resume_state(conn, config)

    guard.check("before_model_load")
    try:
        model = base = tokenizer = tokenized = ref_lps = None
        _warn_remote_code(config)
        model_source = resolve_hf_model_source(config["model"])
        local_only = model_source != config["model"]
        tokenizer_source = _prepare_tokenizer_source(model_source, config["model"])
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **_hf_tokenizer_kwargs(config, local_files_only=local_only),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=_trust_remote_code(config),
            torch_dtype=backend.preferred_dtype(),
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        ).to(backend.device)
        if hasattr(base, "config") and hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        target_modules = _torch_target_modules(torch, base, _lora_target(config))
        _rank = config.get("lora_rank", 8)
        model = get_peft_model(
            base,
            LoraConfig(
                r=_rank,
                lora_alpha=config.get("lora_alpha", _rank),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
                **_lora_init_kwargs(config),
            ),
        )
        adapter_path = None
        if resume and resume.get("checkpoint_path") and Path(resume["checkpoint_path"]).exists():
            adapter_path = resume["checkpoint_path"]
        elif latest and Path(latest["path"]).exists():
            adapter_path = latest["path"]
        if adapter_path:
            _load_torch_adapter_weights(torch, model, adapter_path)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if not hasattr(model, "disable_adapter"):
            return _skip("adapter_disable_unavailable", backend="cuda")

        model_bytes = max(backend.current_memory_bytes(), backend.active_memory_bytes())
        plan = _plan_strategy(config, hardware, model_bytes)
        for _ in range(attempt):
            plan = _degrade_plan(plan)
        if plan is None:
            _log_event("skip", backend="cuda", reason="insufficient_budget", model_gb=round(model_bytes / 1e9, 3))
            return _skip("insufficient_budget", backend="cuda", model_gb=round(model_bytes / 1e9, 3))

        guard = AdaptiveMemoryGuard(backend, plan.memory_limit_bytes)
        if plan.aggressive_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        batch, fresh_ids, fresh = _load_batch(conn, config, plan, resume)
        if not batch:
            return _skip("empty_batch", backend="cuda")

        cfg = _tighten_small_batch_cfg(_build_train_cfg(config, plan), len(batch))
        tokenized = _attach_advantages(
            [_tokenize_torch(tokenizer, {**item, "max_seq_len": config.get("max_seq_len", 2048)}, backend.device) for item in batch],
            db.get_ema(conn)[0], cfg,
        )
        # Reference logprobs: eval mode (no dropout) with LoRA disabled, so the
        # reference is the base-model distribution and is deterministic.
        model.eval()
        with model.disable_adapter():
            ref_lps = [_compute_logprobs_torch(model, item, torch) for item in tokenized]
        model.train()

        _maybe_apply_liger(model, config)
        model = _maybe_torch_compile(model, config)
        ema_mean, ema_count = db.get_ema(conn)
        total_target_steps = int(resume["total_steps"]) if resume else int(cfg["steps"])
        remaining_steps = int(resume["remaining_steps"]) if resume else total_target_steps
        cfg["steps"] = min(remaining_steps, _slice_steps(config, remaining_steps))
        diag_sink: list[dict] = []
        loss_fn = _make_loss_fn_torch(model, tokenized, ref_lps, cfg, ema_mean, torch, diag_sink=diag_sink)
        optimizer = _build_optimizer_torch(torch, model, cfg, config)
        if resume and resume.get("checkpoint_path"):
            _load_torch_optimizer(torch, optimizer, resume["checkpoint_path"])
        backend.reset_peak_memory()

        total_loss = 0.0
        total_steps = 0
        for step in range(cfg["steps"]):
            for retry in range(_pressure_retry_limit(config) + 1):
                try:
                    current = backend.hardware()
                    block = _background_block_reason(config, backend, current)
                    if block:
                        if block in _TRANSIENT_RESOURCE_BLOCKS and retry < _pressure_retry_limit(config):
                            _pressure_cooldown(backend, config, step, retry, reason=block)
                            continue
                        _log_event("skip", backend="cuda", reason=block, step=step)
                        return _skip(block, backend="cuda", step=step)
                    if not _has_minimum_headroom(current):
                        raise MemoryError("memory pressure while training")
                    guard.check(f"before_step_{step}")
                    optimizer.zero_grad(set_to_none=True)
                    step_loss = 0.0
                    diag_sink.clear()
                    for micro in range(cfg["grad_accum"]):
                        idx = (step * cfg["grad_accum"] + micro) % len(tokenized)
                        loss = loss_fn(idx) / cfg["grad_accum"]
                        loss.backward()
                        step_loss += float(loss.detach().item())
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).detach().item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    backend.synchronize()
                    backend.clear_cache()
                    diag_fields = _aggregate_diag(diag_sink)
                    _log_event(
                        "opt_step",
                        backend="cuda",
                        step=step,
                        loss=round(step_loss, 6),
                        grad_norm=round(grad_norm, 6),
                        **diag_fields,
                    )
                    guard.log_step("step", step)
                    guard.check(f"after_step_{step}")
                    total_loss += step_loss
                    total_steps += 1
                    break
                except MemoryError as exc:
                    if retry >= _pressure_retry_limit(config):
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    _pressure_cooldown(backend, config, step, retry, reason="memory_pressure", detail=str(exc))

        remaining_steps = max(0, remaining_steps - total_steps)
        if remaining_steps > 0:
            if total_steps == 0:
                return _skip("no_steps_completed", backend="cuda")
            run_id = resume["run_id"] if resume else f"{int(time.time())}-{os.getpid()}"
            checkpoint_path = _save_partial_cuda(model, config, run_id, torch, optimizer)
            parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
            db.save_training_state(conn, _training_state_payload(
                config, cfg, backend.name, batch, fresh_ids, resume, latest,
                run_id, checkpoint_path, remaining_steps, total_target_steps,
            ))
            return _paused_result("cuda", cfg, batch, total_steps, remaining_steps, config, checkpoint_path, parent_v)

        ema_mean, ema_count = _update_ema_from_fresh(ema_mean, ema_count, fresh, cfg["ema_decay"])
        parent_v = resume.get("parent_version") if resume else (latest["version"] if latest else None)
        new_v, temp_dir, save_dir = _stage_adapter_dir(_next_adapter_version(conn), config)
        try:
            model.save_pretrained(temp_dir, safe_serialization=True)
            _commit_adapter_dir(temp_dir, save_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        adapter_file = next((str(path) for path in save_dir.glob("*.safetensors")), None)
        if not adapter_file:
            raise RuntimeError("adapter_save_missing_safetensors")
        return _finalize_training(
            conn, config, backend, cfg, batch, fresh_ids, adapter_file,
            ema_mean, ema_count, total_loss, total_steps, new_v, parent_v,
            resume.get("checkpoint_path") if resume else None,
        )
    finally:
        model = base = tokenizer = tokenized = ref_lps = None
        _release_backend_memory(backend)


def _attempt_train(config, conn, backend, hardware, attempt: int):
    if backend.name == "mlx":
        return _attempt_train_mlx(config, conn, backend, hardware, attempt)
    return _attempt_train_cuda(config, conn, backend, hardware, attempt)


def _is_retryable_memory_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(
        token in name or token in msg
        for token in (
            "outofmemory",
            "cuda out of memory",
            "memory pressure",
            "out of memory",
            "insufficient memory",
            "unable to allocate",
            "working set",
        )
    )


def train_result(config, conn):
    lock_fd = None
    if not (config.get("_skip_lock_once") and config.get("_fresh_process_retry_done")):
        lock_fd = _acquire_lock()
        if lock_fd is None:
            _log_event("skip", reason="train_lock_held")
            return _skip("train_lock_held")

    caffeinate = _keep_awake() if config.get("_background") else None
    try:
        if config.get("low_priority", True):
            _set_low_priority()

        resume = _resume_state(conn, config)
        trainable = db.count_trainable_untrained(conn, source=_feedback_source(config))
        if not resume and trainable < config.get("batch_min", 32):
            return _skip("below_threshold", trainable=trainable, batch_min=config.get("batch_min", 32))

        compat = model_compatibility(config)
        if not compat["ok"]:
            _log_event("skip", reason=compat["reason"], detail=compat.get("detail"), backend=compat.get("backend"))
            return _skip(compat["reason"], detail=compat.get("detail"), backend=compat.get("backend"))

        try:
            backend = _select_backend(config)
        except Exception as exc:
            _log_event("skip", reason="backend_unavailable", error=type(exc).__name__, detail=str(exc))
            return _skip("backend_unavailable", detail=str(exc))
        hardware = _settle_backend_hardware(backend, backend.hardware())
        if not _has_minimum_headroom(hardware):
            avail_gb, _ = _hardware_gbs(hardware)
            _log_event("skip", backend=backend.name, reason="insufficient_headroom", available_gb=avail_gb)
            return _skip("insufficient_headroom", backend=backend.name, available_gb=avail_gb)
        block = _background_block_reason(config, backend, hardware)
        if block:
            avail_gb, host_gb = _hardware_gbs(hardware)
            _log_event("skip", backend=backend.name, reason=block, available_gb=avail_gb, host_available_gb=host_gb)
            return _skip(block, backend=backend.name, available_gb=avail_gb, host_available_gb=host_gb)
        avail_gb, host_gb = _hardware_gbs(hardware)
        _log_event(
            "train_start",
            backend=backend.name,
            device=getattr(hardware, "device_name", backend.name),
            total_gb=round(hardware.total_memory_bytes / 1e9, 3),
            available_gb=avail_gb,
            host_available_gb=host_gb,
        )

        max_attempts = max(2, int(config.get("cuda_degrade_attempts", 5)))
        for attempt in range(max_attempts):
            try:
                hardware = hardware if attempt == 0 else _settle_backend_hardware(backend, backend.hardware())
                if not _has_minimum_headroom(hardware):
                    avail_gb, _ = _hardware_gbs(hardware)
                    _log_event("skip", backend=backend.name, attempt=attempt, reason="insufficient_headroom", available_gb=avail_gb)
                    return _skip("insufficient_headroom", backend=backend.name, attempt=attempt)
                block = _background_block_reason(config, backend, hardware)
                if block:
                    avail_gb, host_gb = _hardware_gbs(hardware)
                    _log_event("skip", backend=backend.name, attempt=attempt, reason=block, available_gb=avail_gb, host_available_gb=host_gb)
                    return _skip(block, backend=backend.name, attempt=attempt)
                result = _attempt_train(config, conn, backend, hardware, attempt)
                if (
                    backend.name == "mlx"
                    and result.get("status") == "skipped"
                    and result.get("reason") == "insufficient_budget"
                ):
                    retried = _fresh_process_train_retry(config, conn)
                    if retried is not None:
                        return retried
                if result.get("status") == "trained":
                    _log_event("train_done", attempt=attempt, **{k: v for k, v in result.items() if k != "status"})
                    return result
                if (
                    result.get("status") == "skipped"
                    and result.get("reason") in {"insufficient_budget", "memory_pressure"}
                    and attempt + 1 < max_attempts
                ):
                    _release_backend_memory(backend)
                    continue
                return result
            except Exception as exc:
                _release_backend_memory(backend)
                detail = _scrub_secret(str(exc))
                _log_event("train_error", attempt=attempt, error=type(exc).__name__, detail=detail)
                if "missing_cuda_training_dependency:" in str(exc):
                    return _skip("missing_cuda_training_dependency", detail=detail)
                if not _is_retryable_memory_error(exc):
                    return _skip("train_error", detail=detail, backend=backend.name, attempt=attempt)
                if attempt == 0:
                    continue
                return _skip("memory_pressure", detail=detail, backend=backend.name, attempt=attempt)
        return _skip("no_training_result")
    finally:
        if caffeinate is not None:
            caffeinate.terminate()
        _release_lock(lock_fd)


def train(config, conn):
    result = train_result(config, conn)
    if result.get("status") == "trained":
        return {k: v for k, v in result.items() if k != "status"}
    return None


def _adapter_dir(config=None):
    root = Path((config or {}).get("adapter_root", Path.home() / ".reinforceclaw" / "adapters"))
    return db.secure_private_dir(root)


def _resume_dir(config=None):
    root = _adapter_dir(config) / ".resume"
    return db.secure_private_dir(root)


def _resume_checkpoint_dir(config, run_id: str) -> Path:
    path = _resume_dir(config) / run_id
    return db.secure_private_dir(path)


def _stage_adapter_dir(version: int, config=None) -> tuple[int, Path, Path]:
    root = _adapter_dir(config)
    current = version
    while True:
        final_dir = root / f"v{current}"
        temp_dir = root / f".v{current}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
        if not final_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
            db.secure_private_dir(temp_dir)
            return current, temp_dir, final_dir
        current += 1


def _commit_adapter_dir(temp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        raise FileExistsError(f"adapter dir already exists: {final_dir}")
    temp_dir.replace(final_dir)


def _convert_to_gguf(adapter_dir):
    import subprocess

    safetensors_file = Path(adapter_dir) / "adapter.safetensors"
    if not safetensors_file.exists():
        safetensors_file = Path(adapter_dir) / "adapter_model.safetensors"
    gguf_file = Path(adapter_dir) / "adapter.gguf"
    if gguf_file.exists():
        return str(gguf_file)
    if not safetensors_file.exists():
        return None
    for cmd in ["convert-lora-to-gguf", "python3 -m llama_cpp.convert_lora"]:
        try:
            result = subprocess.run(
                cmd.split() + [str(safetensors_file), "--outfile", str(gguf_file)],
                capture_output=True,
                timeout=120,
                env=_child_env(),
            )
            if result.returncode == 0 and gguf_file.exists():
                return str(gguf_file)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def load_adapter(server_type, adapter_path, model_name):
    import requests

    adapter_file = Path(adapter_path)
    adapter_dir = str(adapter_file.parent)
    if server_type == "ollama":
        model_name = str(model_name or "").strip()
        if not model_name or "\n" in model_name or "\r" in model_name:
            return None
        adapter_ref = str(adapter_file) if adapter_file.is_file() else _convert_to_gguf(adapter_dir)
        if not adapter_ref:
            print(f"Ollama adapter not prepared; use the saved adapter manually: {adapter_path}")
            return None
        modelfile = f"FROM {model_name}\nADAPTER {adapter_ref}\n"
        out_name = f"{model_name.split('/')[-1]}-reinforceclaw"
        modelfile_path = Path(adapter_dir) / "Modelfile.reinforceclaw"
        try:
            modelfile_path.write_text(modelfile, encoding="utf-8")
            db.secure_private_file(modelfile_path)
            result = subprocess.run(
                ["ollama", "create", out_name, "-f", str(modelfile_path)],
                capture_output=True, text=True, timeout=120, env=_child_env(),
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            modelfile_path.unlink(missing_ok=True)
        try:
            response = requests.post(
                "http://localhost:11434/api/create",
                json={"model": out_name, "modelfile": modelfile, "stream": False},
                timeout=60,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    if server_type == "lmstudio":
        try:
            response = requests.post(
                "http://localhost:1234/v1/lora/load",
                json={"path": adapter_dir},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            print(f"Load adapter in LM Studio from: {adapter_dir}")
            return None
    if server_type == "vllm":
        try:
            response = requests.post(
                "http://localhost:8000/v1/load_lora_adapter",
                json={"lora_name": "reinforceclaw", "lora_path": adapter_dir},
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    print(f"Restart your server with adapter: {adapter_path}")
    return None
