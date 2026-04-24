"""CLI. argparse. No Click, no Typer. Setup wizard + all commands."""
# two entry points: `reinforceclaw <cmd>` from terminal, `/rl <cmd>` from inside agents.
# both hit the same functions. wizard is `reinforceclaw init`.

import argparse
import secrets
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from . import db, trainer, presets, profile
from .hooks._common import pop_pending

console = Console()
CONFIG_PATH = Path.home() / ".reinforceclaw" / "config.json"
ADAPTER_ROOT = Path.home() / ".reinforceclaw" / "adapters"
OPENCLAW_BRIDGE_CONFIG = Path.home() / ".reinforceclaw" / "openclaw_bridge.json"
TRAIN_RETRY_PATH = Path.home() / ".reinforceclaw" / "train.retry"
RESET_MARK_PATH = Path.home() / ".reinforceclaw" / "reset.marker"

PRESETS = {
    "careful": "Safest updates. Highest KL, slowest drift.",
    "balanced": "Stable MIS-PO default. Sweep winner.",
    "aggressive": "Faster drift. More overfit risk.",
}

DEFAULTS = {
    "loss_fn": "mis-po",
    "lora_target": "attention",
    "token_clip": [0.5, 2.0], "kl_coeff": 0.001, "lora_rank": 16,
    "grad_accum": 2, "grad_clip": 1.0, "batch_min": 32, "batch_size": 4,
    "replay_ratio": 0.0, "ema_decay": 0.99, "pos_weight": 1.2,
    "adv_clip": 2.0, "max_passes": 1.0,
    "pressure_retry_limit": 2, "pressure_cooldown_s": 3.0,
    "background_slice_steps": 2,
    "adapter_keep": 0,  # 0 = keep all adapters forever
    "train_schedule": "auto",
    "schedule_window_minutes": 180,
    # speed-up knobs (opt-in; 0/None = disabled so validated baseline stands).
    "lora_plus_ratio": 0.0,    # B-matrix LR multiplier; ~16 roughly doubles convergence
    "use_liger": False,        # Liger-Kernel fused CE + RMSNorm (CUDA)
    "compile_backend": "none", # "reduce-overhead" | "max-autotune" | "default" (CUDA)
    "mlx_compile": False,      # mx.compile on MLX loss_fn
    "lora_init": "default",    # "pissa" | "olora" | "loftq" | "eva" — PEFT init recipe
    "sdpa_backend": "auto",    # PyTorch SDPA hint; "auto" lets torch pick FA4 on Blackwell
    "trust_remote_code": False,
}
_PROFILE_CACHE = {}

from .models import MODELS  # model catalog lives in models.py

LOGO = r"""
[bold green]
    _   __          __
   / | / /__  ____/ /___ ____
  /  |/ / / / / __  / __ `/ _ \
 / /|  / /_/ / /_/ / /_/ /  __/
/_/ |_/\__,_/\__,_/\__, /\___/
                  /____/
[/bold green]"""


def _read_json(path, default=None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _read_config():
    return _read_json(CONFIG_PATH)


_TUNED_KEYS = ("lr", "traj_clip", "steps", "kl_coeff", "lora_rank", "lora_alpha",
               "lora_target", "batch_min", "batch_size", "grad_accum", "pos_weight",
               "replay_ratio", "token_clip")


def _auto_tuned_values(cfg: dict) -> dict:
    model, preset = cfg.get("model", ""), cfg.get("preset", "balanced")
    mp = cfg.get("model_profile") if cfg.get("model_profile_model") == model else None
    try:
        if isinstance(mp, dict):
            prof = profile.ModelProfile(**mp)
        else:
            prof = _PROFILE_CACHE.get(model) or profile.detect(model)
            _PROFILE_CACHE[model] = prof
    except TypeError:
        prof = _PROFILE_CACHE.get(model) or profile.detect(model)
        _PROFILE_CACHE[model] = prof
    tuned = presets.pick(prof, preset)
    tuned["model_profile_model"] = model
    return tuned


def _resolve_config(cfg: dict) -> dict:
    if not cfg or cfg.get("tuning_mode") == "custom" or not cfg.get("model"):
        return cfg
    tuned = _auto_tuned_values(cfg)
    resolved = dict(cfg)
    for key in _TUNED_KEYS:
        if key in tuned:
            resolved[key] = tuned[key]
    resolved["model_profile"] = tuned["model_profile"]
    resolved["model_profile_model"] = tuned["model_profile_model"]
    resolved["tuning_mode"] = "auto"
    return resolved


def load_config():
    cfg = _read_config()
    resolved = _resolve_config(cfg)
    if resolved != cfg and resolved.get("tuning_mode") == "auto":
        save_config(resolved)
    return resolved


def save_config(cfg):
    _write_json_atomic(CONFIG_PATH, cfg)


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


def _load_model_cfg():
    cfg = load_config()
    if cfg.get("model"):
        return cfg
    console.print("[red]Run 'reinforceclaw init' first.[/red]")
    return None


def _trainable_untrained(conn):
    return db.count_trainable_untrained(conn)


def _swap_latest(cfg, conn):
    latest = db.latest_adapter(conn)
    return latest and trainer.load_adapter(cfg.get("server", "ollama"), latest["path"], cfg.get("serve_model") or cfg["model"])


def _record_background_event(conn, kind):
    db.record_background_event(conn, kind, datetime.now().hour)


def _next_retry_delay(conn, base_delay=900):
    history = db.background_history(conn)
    if not history:
        return base_delay
    now = datetime.now()
    best = base_delay
    best_score = None
    for hours_ahead in range(0, 24):
        candidate = now + timedelta(hours=hours_ahead)
        delay = max(base_delay, int((candidate.replace(minute=0, second=0, microsecond=0) - now).total_seconds()))
        hour = candidate.hour
        stats = history.get(hour, {"pressure_count": 0, "success_count": 0})
        score = stats["pressure_count"] - (0.5 * stats["success_count"])
        if best_score is None or score < best_score:
            best_score, best = score, delay
    return best


def _set_panel(enabled):
    cfg = load_config()
    cfg["panel_enabled"] = enabled
    save_config(cfg)
    console.print("[green]Panel on.[/green]" if enabled else "[yellow]Panel off.[/yellow] Use /rl good or /rl bad.")


def _write_json_atomic(path, payload, *, backup=False):
    db.secure_private_dir(path.parent)
    if backup and path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.bak"))
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{os.urandom(4).hex()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _write_text_atomic(path, text, *, backup=False):
    db.secure_private_dir(path.parent)
    if backup and path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.bak"))
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{os.urandom(4).hex()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_private_json(path, payload):
    _write_json_atomic(path, payload)


def reset_state():
    lock_fd = trainer._acquire_lock()
    if lock_fd is None:
        raise RuntimeError("training is running; try again when the current update finishes")
    try:
        conn = db.connect()
        try:
            db.reset_all(conn)
        finally:
            conn.close()
        shutil.rmtree(ADAPTER_ROOT, ignore_errors=True)
        shutil.rmtree(Path.home() / ".reinforceclaw" / "pending", ignore_errors=True)
        TRAIN_RETRY_PATH.unlink(missing_ok=True)
        _write_private_json(RESET_MARK_PATH, {"reset_at": datetime.now().isoformat()})
    finally:
        trainer._release_lock(lock_fd)


def rollback_adapter(conn, cfg, version=None):
    lock_fd = trainer._acquire_lock()
    if lock_fd is None:
        return None, "training is running; try again when the current update finishes"
    try:
        prev = db.rollback_to(conn, version) if version is not None else db.rollback(conn)
        if prev:
            trainer.load_adapter(cfg.get("server", "ollama"), prev["path"], cfg.get("serve_model") or cfg.get("model", ""))
        return prev, None
    finally:
        trainer._release_lock(lock_fd)


# -- setup wizard --

def cmd_init(_args):
    console.print(LOGO)
    console.print(Panel("Hands-off reinforcement learning. Rate responses, your model learns the rest.",
                        title="Welcome to ReinforceClaw", border_style="green"))

    # agents
    agents = []
    console.print("\n[bold]Which agents do you use?[/bold]")
    for name in ["Claude Code", "Codex", "OpenClaw"]:
        if Confirm.ask(f"  {name}", default=(name == "Claude Code")):
            agents.append(name.lower().replace(" ", "_"))
    agents = agents or ["claude_code"]

    # model
    console.print("\n[bold]Pick your local model:[/bold]")
    companies = list(MODELS.keys()) + ["Other (HuggingFace ID)"]
    for i, c in enumerate(companies, 1):
        console.print(f"  [green]{i}[/green]. {c}")

    choice = _clamp(IntPrompt.ask("Company", default=1), 1, len(companies))
    if choice <= len(MODELS):
        company = list(MODELS.keys())[choice - 1]
        models = MODELS[company]
        console.print(f"\n[bold]{company} models:[/bold]")
        for i, m in enumerate(models, 1):
            console.print(f"  [green]{i}[/green]. {m}")
        mc = _clamp(IntPrompt.ask("Model", default=1), 1, len(models))
        model_name = models[mc - 1]
    else:
        model_name = Prompt.ask("HuggingFace model ID")

    # MoE models work fine, no warning needed — user picked what they want

    # preset
    console.print("\n[bold]Training preset:[/bold]")
    for i, (name, desc) in enumerate(PRESETS.items(), 1):
        console.print(f"  [green]{i}[/green]. [bold]{name}[/bold] — {desc}")
    console.print(f"  [green]4[/green]. [bold]custom[/bold] — set your own learning rate and steps")
    pc = _clamp(IntPrompt.ask("Preset", default=2), 1, 4)
    if pc <= 3:
        preset_name = list(PRESETS.keys())[pc - 1]
        tuning_mode = "auto"
        custom_overrides = {}
    else:
        preset_name = "balanced"
        lr = float(Prompt.ask("Learning rate", default="5e-6"))
        steps = IntPrompt.ask("Steps per round", default=32)
        tuning_mode = "custom"
        custom_overrides = {"lr": lr, "steps": steps, "traj_clip": [0.996, 1.001]}

    # server
    servers = ["Ollama", "LM Studio", "vLLM", "Other"]
    console.print("\n[bold]Inference server:[/bold]")
    for i, s in enumerate(servers, 1):
        console.print(f"  [green]{i}[/green]. {s}")
    sc = _clamp(IntPrompt.ask("Server", default=1), 1, len(servers))
    server = ["ollama", "lmstudio", "vllm", "other"][sc - 1]
    serve_model = None
    if server == "ollama":
        serve_model = Prompt.ask(
            "Ollama base model/tag to attach the adapter to",
            default=model_name,
        ).strip()

    cfg = {"model": model_name, "server": server, "preset": preset_name,
           "agents": agents, "panel_enabled": True,
           "tuning_mode": tuning_mode, **DEFAULTS, **custom_overrides}
    if serve_model:
        cfg["serve_model"] = serve_model
    if "openclaw" in agents:
        cfg["openclaw_secret"] = secrets.token_urlsafe(24)
    resolved_cfg = _resolve_config(cfg)
    save_config(resolved_cfg)
    db.init()
    hook_results = _install_hooks(cfg)
    if "openclaw" in agents and not hook_results.get("openclaw", False):
        resolved_cfg.pop("openclaw_secret", None)
        save_config(resolved_cfg)

    # set up training schedule
    from reinforceclaw import scheduler
    schedule = cfg.get("train_schedule", "auto")
    if scheduler.install(schedule, cfg.get("schedule_window_minutes", 180)):
        if schedule not in ("manual", "auto"):
            console.print(f"[green]Training scheduled daily at {schedule}[/green]")
    else:
        resolved_cfg["train_schedule"] = "manual"
        save_config(resolved_cfg)
        detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
        console.print(f"[yellow]Scheduler install failed. Training left in manual mode.{detail}[/yellow]")

    console.print("\n")
    console.print(Panel(
        "[bold red]Rate BOTH good AND bad responses.[/bold red]\n\n"
        "If you only rate bad, your model will get WORSE — it learns what to avoid\n"
        "but has no idea what you actually want. It needs both signals to improve.\n"
        "Bad only = broken model. Good only = weak model. Both = the goal.\n\n"
        "[dim]Your adapter only works on this exact model. Switching models = fresh start.[/dim]",
        title="Important", border_style="red"))
    console.print(Panel(
        f"Model: [bold]{model_name}[/bold] | Preset: [bold]{preset_name}[/bold] | Server: [bold]{server}[/bold]\n"
        f"Config: [dim]{CONFIG_PATH}[/dim]\n\n"
        f"1. Use your AI agent normally\n"
        f"2. Rate responses ([bold]/rl good[/bold], [bold]/rl bad[/bold], or ignore)\n"
        f"3. Once you hit {resolved_cfg['batch_min']} ratings, training runs automatically in the background\n"
        f"4. [bold]reinforceclaw status[/bold] to check progress | [bold]reinforceclaw history[/bold] to fix ratings",
        title="Setup complete", border_style="green"))


def _install_hooks(cfg):
    """Install hooks for each selected agent."""
    hook_dir = Path(__file__).parent / "hooks"
    results = {}

    if "claude_code" in cfg.get("agents", []):
        _install_claude_code_hooks(hook_dir)
        results["claude_code"] = True
    if "codex" in cfg.get("agents", []):
        _install_codex_hooks(hook_dir)
        results["codex"] = True
    if "openclaw" in cfg.get("agents", []):
        results["openclaw"] = _install_openclaw_plugin(cfg)
    return results


def _is_reinforceclaw_hook(entry, script_name):
    for hook in entry.get("hooks", []) if isinstance(entry, dict) else []:
        command = hook.get("command", "") if isinstance(hook, dict) else ""
        if "reinforceclaw" in command and script_name in command:
            return True
    return False


def _replace_reinforceclaw_hook(existing, entry, script_name):
    existing = existing if isinstance(existing, list) else []
    kept = [item for item in existing if not _is_reinforceclaw_hook(item, script_name)]
    kept.append(entry)
    return kept


def _install_json_hooks(path, script_name, command):
    cfg = _read_json(path)
    hooks = cfg.get("hooks") if isinstance(cfg.get("hooks"), dict) else {}
    for event, arg in (("Stop", "stop"), ("UserPromptSubmit", "prompt")):
        existing = hooks.get(event, [])
        entry = {"hooks": [{"type": "command", "command": f"{command} {arg}", "timeout": 30}]}
        hooks[event] = _replace_reinforceclaw_hook(existing, entry, script_name)
    cfg["hooks"] = hooks
    _write_json_atomic(path, cfg, backup=True)


def _install_claude_code_hooks(hook_dir):
    settings_path = Path.home() / ".claude" / "settings.json"
    script = str(hook_dir / "claude_code.py")
    _install_json_hooks(settings_path, "claude_code.py", f"{shlex.quote(sys.executable)} {shlex.quote(script)}")
    console.print(f"[green]Claude Code hooks installed:[/green] {settings_path}")


def _install_codex_hooks(hook_dir):
    """Install hooks into Codex CLI's hooks.json."""
    # codex hooks live at ~/.codex/hooks.json — same protocol as claude code
    hooks_path = Path.home() / ".codex" / "hooks.json"
    script = str(hook_dir / "codex.py")
    _install_json_hooks(hooks_path, "codex.py", f"{shlex.quote(sys.executable)} {shlex.quote(script)}")
    _enable_codex_hooks_feature()
    console.print(f"[green]Codex hooks installed:[/green] {hooks_path}")


def _enable_codex_hooks_feature():
    config_path = Path.home() / ".codex" / "config.toml"
    text = config_path.read_text() if config_path.exists() else ""
    db.secure_private_dir(config_path.parent)
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib:
        try:
            if tomllib.loads(text or "").get("features", {}).get("codex_hooks") is True:
                return
        except tomllib.TOMLDecodeError:
            pass
    lines = text.splitlines()
    in_features = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_features:
                lines.insert(idx, "codex_hooks = true")
                _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
                return
            in_features = stripped == "[features]"
            continue
        if in_features and stripped.startswith("codex_hooks"):
            lines[idx] = "codex_hooks = true"
            _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
            return
    if in_features:
        lines.append("codex_hooks = true")
        _write_text_atomic(config_path, "\n".join(lines).rstrip() + "\n", backup=True)
        return
    _write_text_atomic(config_path, text.rstrip() + ("\n\n" if text.strip() else "") + "[features]\ncodex_hooks = true\n", backup=True)


def _install_openclaw_plugin(cfg):
    """Install the OpenClaw plugin into the user's existing gateway."""
    plugin_dir = Path(__file__).parent / "openclaw_plugin"
    if not plugin_dir.exists():
        console.print("[yellow]Bundled OpenClaw plugin not found. Skipping.[/yellow]")
        return False
    secret = cfg.get("openclaw_secret")
    host = "http://127.0.0.1:8420"
    import subprocess
    try:
        result = subprocess.run(
            ["openclaw", "plugins", "install", str(plugin_dir)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            if secret:
                _write_private_json(OPENCLAW_BRIDGE_CONFIG, {"host": host, "secret": secret, "python": sys.executable})
            configured = True
            for path, value in (
                ("plugins.entries.reinforceclaw-feedback.config.reinforceclawHost", host),
                ("plugins.entries.reinforceclaw-feedback.config.reinforceclawPython", sys.executable),
            ):
                if not value:
                    continue
                configured = subprocess.run(
                    ["openclaw", "config", "set", path, value],
                    capture_output=True, text=True, timeout=15,
                ).returncode == 0 and configured
            enabled = subprocess.run(["openclaw", "plugins", "enable", "reinforceclaw-feedback"], capture_output=True, text=True, timeout=15).returncode == 0
            if configured and enabled:
                console.print("[green]OpenClaw plugin installed — use /rl good, /rl bad, /rl undo, or /rl status in any connected channel[/green]")
                return True
            console.print("[yellow]OpenClaw plugin installed but could not be fully enabled/configured. Run reinforceclaw init again after checking OpenClaw.[/yellow]")
            OPENCLAW_BRIDGE_CONFIG.unlink(missing_ok=True)
            return False
        else:
            console.print(f"[yellow]OpenClaw plugin install failed: {result.stderr.strip()}[/yellow]")
        return False
    except FileNotFoundError:
        console.print("[yellow]openclaw not found. Install it first, then run reinforceclaw init again.[/yellow]")
        return False
    except subprocess.TimeoutExpired:
        console.print("[yellow]OpenClaw plugin install timed out.[/yellow]")
        return False


# -- rating commands --

def cmd_rate(_args, rating=None):
    # rate the last response
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    pending = pop_pending("claude_code") or pop_pending("codex")
    if not pending:
        console.print(
            "[yellow]No captured response to rate. Use /rl good or /rl bad in your agent, "
            "or keep the panel on.[/yellow]"
        )
        conn.close()
        return
    db.add_feedback(
        conn, pending["model"], pending["prompt"], pending["response"], rating,
        context=pending.get("context"), source=pending.get("source", "cli"),
        event_id=pending.get("key"), rollout_context=pending.get("rollout_context"),
    )
    label = "[green]good[/green]" if rating == 1 else "[red]bad[/red]"
    console.print(f"Rated: {label}")
    _maybe_train(cfg, conn)
    conn.close()



def cmd_undo(_args):
    conn = db.connect()
    removed = db.remove_last(conn)
    if removed:
        console.print(f"[yellow]Removed last rating ({'good' if removed['rating']==1 else 'bad'})[/yellow]")
    else:
        console.print("[dim]Nothing to undo.[/dim]")
    conn.close()


def cmd_train(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    background = bool(getattr(_args, "background", False))
    cfg = {**cfg, "_background": background}
    conn = db.connect()
    n = _trainable_untrained(conn)
    batch_min = cfg.get("batch_min", 8)
    resume = trainer._resume_state(conn, cfg)
    if background:
        TRAIN_RETRY_PATH.unlink(missing_ok=True)
    if n < batch_min and not resume:
        console.print(f"[yellow]Only {n} rated responses. Need at least {batch_min}.[/yellow]")
        conn.close()
        return
    # only ask for confirmation when human is at the keyboard
    if sys.stdin.isatty() and not background:
        if not Confirm.ask(f"{n} ratings ready. Train now?"):
            conn.close()
            return
    console.print("[bold]Training...[/bold]" if not background else "[dim]Background training...[/dim]")
    result = trainer.train_result(cfg, conn)
    if result.get("status") == "trained":
        metrics = {k: v for k, v in result.items() if k != "status"}
        gate = trainer.publish_gate(cfg, metrics["path"])
        if gate.get("ok"):
            db.activate_training_round(conn, metrics["version"], metrics["ema_mean"], metrics["ema_count"], metrics["feedback_ids"])
            console.print(f"[green]Done![/green] Loss: {metrics['avg_loss']:.4f}, "
                           f"Batch: {metrics['batch_size']}, EMA: {metrics['ema_mean']:.3f}")
            if background:
                _record_background_event(conn, "success")
            ok = _swap_latest(cfg, conn)
            if ok is True:
                console.print("[green]Prepared adapter/model loaded.[/green]")
            elif ok is False:
                console.print("[yellow]Could not load adapter automatically. Restart or repoint your server manually.[/yellow]")
            elif ok is None:
                console.print("[yellow]Adapter saved. Manual server reload may still be needed.[/yellow]")
        else:
            db.reject_adapter(conn, metrics["version"])
            console.print("[yellow]Trained candidate rejected by publish gate.[/yellow]")
            if gate.get("reason"):
                console.print(f"[dim]Gate: {gate['reason']}[/dim]")
    else:
        reason = result.get("reason", "unknown")
        transient = {
            "high_cpu_load", "memory_busy", "gpu_busy", "gpu_memory_busy",
            "host_memory_busy", "low_free_vram", "outside_schedule_window",
            "missing_cuda_idle_telemetry", "memory_pressure", "insufficient_headroom",
        }
        if background and reason == "resume_pending":
            from reinforceclaw.hooks._common import queue_training
            queue_training(delay_seconds=1)
        elif background and reason in transient:
            _record_background_event(conn, "pressure")
            from reinforceclaw.hooks._common import queue_training
            queue_training(delay_seconds=_next_retry_delay(conn))
        console.print(
            "[yellow]Skipped for now.[/yellow]"
            if background and reason not in ("ready", "backend_unavailable", "below_threshold")
            else "[yellow]Training failed.[/yellow]"
        )
        if background or reason != "ready":
            console.print(f"[dim]Reason: {reason}[/dim]")
        if result.get("detail"):
            console.print(f"[dim]{result['detail']}[/dim]")
    conn.close()


def cmd_smoke(_args):
    cfg = _load_model_cfg()
    if not cfg:
        return
    conn = db.connect()
    try:
        status = trainer.smoke_status(cfg, conn)
    finally:
        conn.close()
    title = "[green]Would train[/green]" if status["would_train"] else "[yellow]Would skip[/yellow]"
    t = Table(title=f"Background Smoke: {title}", border_style="cyan")
    t.add_column("Key", style="bold")
    t.add_column("Value")
    for key in ("reason", "backend", "schedule", "trainable", "batch_min", "available_gb", "host_available_gb", "detail"):
        if key in status and status[key] is not None:
            t.add_row(key, str(status[key]))
    console.print(t)


def cmd_status(_args):
    cfg = load_config()
    conn = db.connect()
    counts = db.count(conn)
    ema_mean, ema_count = db.get_ema(conn)
    latest = db.latest_adapter(conn)

    t = Table(title="ReinforceClaw Status", border_style="cyan")
    t.add_column("", style="bold")
    t.add_column("")
    t.add_row("Model", cfg.get("model", "not set"))
    t.add_row("Preset", cfg.get("preset", "balanced"))
    from .profile import detect as _detect_profile
    mp = cfg.get("model_profile")
    if not mp:
        _p = _detect_profile(cfg.get("model", ""))
        mp = {"kind": _p.kind, "size_bucket": _p.size_bucket}
    t.add_row("Profile", f"{mp.get('kind', '?')} / {mp.get('size_bucket', mp.get('scale', '?'))}")
    t.add_row("Tuning", cfg.get("tuning_mode", "auto"))
    t.add_row("Server", cfg.get("server", "ollama"))
    t.add_row("Adapter", f"v{latest['version']}" if latest else "none (base model)")
    t.add_row("Ratings", f"{counts['total']} total ({counts['good']}+ {counts['bad']}-)")
    t.add_row("Untrained", str(_trainable_untrained(conn)))
    t.add_row("EMA", f"{ema_mean:.3f} ({ema_count} updates)")
    t.add_row("Panel", "on" if cfg.get("panel_enabled", True) else "off")
    console.print(t)
    conn.close()


def cmd_rollback(_args):
    cfg = load_config()
    conn = db.connect()
    adapters = db.list_adapters(conn)
    if not adapters:
        console.print("[yellow]No adapters to roll back to.[/yellow]")
        conn.close()
        return
    t = Table(title="Adapters", border_style="green")
    t.add_column("#", style="dim")
    t.add_column("Status")
    t.add_column("Created")
    for a in adapters:
        status = "[green]active[/green]" if a["status"] == "active" else "[dim]rolled back[/dim]"
        t.add_row(f"v{a['version']}", status, a["created_at"])
    console.print(t)
    pick = IntPrompt.ask("Roll back to version", default=adapters[0]["version"])
    prev, error = rollback_adapter(conn, cfg, pick)
    if prev:
        console.print(f"[green]Now on v{prev['version']}[/green]")
    elif error:
        console.print(f"[yellow]{error}[/yellow]")
    conn.close()


def cmd_reset(_args):
    if not Confirm.ask("[red]Delete all ratings, adapters, and start fresh?[/red]"):
        return
    try:
        reset_state()
        console.print("[green]Reset. Clean slate.[/green]")
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")


def cmd_on(_a):
    _set_panel(True)

def cmd_off(_a):
    _set_panel(False)


def _maybe_train(cfg, conn):
    if cfg.get("train_schedule", "auto") != "auto":
        return
    if _trainable_untrained(conn) >= cfg.get("batch_min", 8):
        from reinforceclaw.hooks._common import queue_training
        console.print("[dim]Batch ready, training queued in background...[/dim]")
        queue_training()


def cmd_history(_args):
    """Show recent ratings. Use 'reinforceclaw rate <id> good|bad|delete' to edit one."""
    conn = db.connect()
    rows = db.recent(conn)
    if not rows:
        console.print("[dim]No ratings yet.[/dim]")
        conn.close()
        return
    t = Table(title="Recent Ratings", border_style="green")
    t.add_column("ID", style="dim")
    t.add_column("Prompt")
    t.add_column("Rating")
    t.add_column("Source")
    t.add_column("When")
    for r in rows:
        label = {"1": "[green]good[/green]", "-1": "[red]bad[/red]", "0": "[dim]unrated[/dim]"}
        prompt = r["prompt"][:47] + "..." if len(r["prompt"]) > 50 else r["prompt"]
        t.add_row(str(r["id"]), prompt, label.get(str(r["rating"]), "?"),
                  r["source"], r["created_at"])
    console.print(t)
    console.print("[dim]To change a rating: reinforceclaw rate <id> <good|bad|delete>[/dim]")
    conn.close()


def cmd_rerate(_args):
    """Change or delete a specific rating: reinforceclaw rate 42 good|bad|delete"""
    rating = {"good": 1, "bad": -1, "delete": 0, "ignore": 0}[_args.value]
    conn = db.connect()
    changed = db.revise_feedback_rating(conn, _args.id, rating)
    if changed:
        label = "[green]good[/green]" if rating == 1 else "[red]bad[/red]" if rating == -1 else "[dim]deleted[/dim]"
        console.print(f"Rating #{_args.id} changed to {label}")
    else:
        console.print(f"[yellow]No rating found with ID {_args.id}.[/yellow]")
    conn.close()


def cmd_schedule(_args):
    """Set training schedule: reinforceclaw schedule 03:00 / reinforceclaw schedule auto / reinforceclaw schedule manual"""
    from reinforceclaw import scheduler
    cfg = load_config()
    if hasattr(_args, 'time') and _args.time:
        val = _args.time
        try:
            if val not in ("auto", "manual"):
                scheduler._parse_time(val)
        except ValueError:
            console.print("[red]Use HH:MM format, 'auto', or 'manual'[/red]")
            return
        cfg["train_schedule"] = val
        if scheduler.install(val, cfg.get("schedule_window_minutes", 180)):
            save_config(cfg)
            console.print(f"[green]Schedule set: {val}[/green]")
        else:
            detail = f" {escape(scheduler.LAST_ERROR)}" if getattr(scheduler, "LAST_ERROR", "") else ""
            console.print(f"[red]Could not install scheduler. Leaving existing schedule unchanged.{detail}[/red]")
    else:
        console.print(f"Current: {cfg.get('train_schedule', 'auto')}")
        console.print("[dim]reinforceclaw schedule 03:00 / auto / manual[/dim]")


COMMANDS = {
    "init": cmd_init, "good": lambda a: cmd_rate(a, 1), "bad": lambda a: cmd_rate(a, -1),
    "undo": cmd_undo, "train": cmd_train, "status": cmd_status,
    "rollback": cmd_rollback, "reset": cmd_reset, "on": cmd_on, "off": cmd_off,
    "history": cmd_history, "schedule": cmd_schedule, "smoke": cmd_smoke,
}


def main():
    parser = argparse.ArgumentParser(prog="reinforceclaw", description="Personal RL for AI agents")
    sub = parser.add_subparsers(dest="command")
    for name in COMMANDS:
        if name in ("rate", "schedule", "train"):
            continue  # these have custom args, added below
        sub.add_parser(name)

    train_p = sub.add_parser("train", help="Run training now")
    train_p.add_argument("--background", action="store_true")

    # reinforceclaw rate <id> <good|bad|delete>
    rate_p = sub.add_parser("rate", help="Change/delete a rating: reinforceclaw rate 42 good|bad|delete")
    rate_p.add_argument("id", type=int)
    rate_p.add_argument("value", choices=["good", "bad", "delete", "ignore"])

    # reinforceclaw schedule <time>
    sched_p = sub.add_parser("schedule", help="Set training schedule")
    sched_p.add_argument("time", nargs="?")

    args = parser.parse_args()
    if args.command == "rate":
        cmd_rerate(args)
        return 0
    elif args.command in COMMANDS:
        COMMANDS[args.command](args)
        return 0
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
