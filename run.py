"""
run.py — Entry point for local development and Streamlit Cloud deployment.

Local  (python run.py):
  Spawns FastAPI (:8000) and Streamlit (:8501) as subprocesses.

Cloud  (Streamlit Cloud runs this as the app file):
  Starts FastAPI in a background daemon thread, then delegates UI to
  frontend/app.py via runpy — no subprocess spawning, no signal handlers.

Secret precedence (highest → lowest):
  1. Streamlit Cloud Secrets UI  /  .streamlit/secrets.toml
  2. .env
  3. existing environment variables
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Detect whether we are inside the Streamlit script runner
# ---------------------------------------------------------------------------

def _running_under_streamlit() -> bool:
    """True when this script is executed by Streamlit (cloud or `streamlit run`)."""
    # Streamlit Cloud: __file__ lives under /mount/src/
    try:
        if "/mount/src" in os.path.abspath(__file__):
            return True
    except Exception:
        pass
    # Fallback: /mount/src directory exists on the filesystem
    if os.path.exists("/mount/src"):
        return True
    # Running via `streamlit run` — streamlit appears in argv[0]
    if sys.argv and "streamlit" in sys.argv[0]:
        return True
    # Not the OS main thread → Streamlit script-runner thread
    if threading.current_thread() is not threading.main_thread():
        return True
    # Active Streamlit runtime (local `streamlit run`)
    try:
        import streamlit.runtime
        return streamlit.runtime.exists()
    except Exception:
        return False


# Evaluate once at import time so both branches can reference it
_IS_CLOUD: bool = _running_under_streamlit()


# ---------------------------------------------------------------------------
# Secret / env helpers
# ---------------------------------------------------------------------------


def _load_toml_secrets(base_env: dict) -> dict:
    """Parse .streamlit/secrets.toml and overlay its values onto base_env."""
    secrets_path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        return base_env
    env = dict(base_env)
    _scalar = re.compile(
        r'^([A-Z][A-Z0-9_]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^#\n\[]+))\s*(?:#.*)?$'
    )
    with open(secrets_path, encoding="utf-8") as fh:
        for line in fh:
            m = _scalar.match(line.strip())
            if m:
                key = m.group(1)
                value = (m.group(2) or m.group(3) or m.group(4) or "").strip()
                if value and value != "sk-...":
                    env[key] = value
    return env


def _inject_streamlit_secrets() -> None:
    """On Streamlit Cloud, copy st.secrets into os.environ."""
    try:
        import streamlit as st
        for k, v in st.secrets.items():
            if isinstance(v, str) and v not in ("sk-...", ""):
                os.environ.setdefault(k, v)
    except Exception:
        pass
    # Also honour .env if present (lower priority)
    try:
        from dotenv import dotenv_values
        for k, v in dotenv_values(os.path.join(ROOT, ".env")).items():
            if v:
                os.environ.setdefault(k, v)
    except Exception:
        pass


def _start_fastapi_thread() -> None:
    """Start uvicorn in a daemon thread (used in cloud / Streamlit-runner mode)."""
    import uvicorn
    os.environ.setdefault("PYTHONPATH", ROOT)
    config = uvicorn.Config(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="error",
    )
    server = uvicorn.Server(config)
    # MUST disable — signal.signal() only works in the OS main thread
    server.install_signal_handlers = False
    threading.Thread(target=server.run, daemon=True, name="fastapi").start()
    time.sleep(2)   # give uvicorn time to bind


# ---------------------------------------------------------------------------
# Cloud / Streamlit-runner execution path  (mutually exclusive with main())
# ---------------------------------------------------------------------------

if _IS_CLOUD:
    _inject_streamlit_secrets()
    # Guard against re-starting on every Streamlit rerun
    if not os.environ.get("_FASTAPI_THREAD_STARTED"):
        _start_fastapi_thread()
        os.environ["_FASTAPI_THREAD_STARTED"] = "1"
    # Delegate rendering to the actual frontend app
    import runpy as _runpy
    _runpy.run_path(
        os.path.join(ROOT, "frontend", "app.py"),
        run_name="__main__",
    )


# ---------------------------------------------------------------------------
# Local development path  (python run.py) — skipped entirely on cloud
# ---------------------------------------------------------------------------

elif __name__ == "__main__":
    import signal
    import subprocess
    from dotenv import dotenv_values

    print("Starting Adaptive RAG Configurator…\n")

    env_file = os.path.join(ROOT, ".env")
    base_env = {**os.environ, **dotenv_values(env_file)}
    merged_env = _load_toml_secrets(base_env)
    merged_env["PYTHONPATH"] = ROOT

    provider = merged_env.get("LLM_PROVIDER", "ollama")
    _model_key = {
        "openai": "OPENAI_MODEL", "groq": "GROQ_MODEL",
        "groq-fallback": "GROQ_MODEL", "custom": "CUSTOM_MODEL",
    }.get(provider, "OLLAMA_MODEL")
    model = merged_env.get(_model_key, "?")
    print(f"  LLM provider : {provider}  |  model : {model}")

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT, env=merged_env,
    )
    time.sleep(2)

    st_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         os.path.join(ROOT, "frontend", "app.py"),
         "--server.port", "8501", "--server.address", "0.0.0.0",
         "--server.headless", "true"],
        cwd=ROOT, env=merged_env,
    )

    print("\n" + "=" * 60)
    print("  FastAPI  →  http://localhost:8000")
    print("  API docs →  http://localhost:8000/docs")
    print("  Streamlit→  http://localhost:8501")
    print("  Press Ctrl+C to stop both services")
    print("=" * 60 + "\n")

    def _shutdown(signum, frame):
        print("\nShutting down…")
        api_proc.terminate()
        st_proc.terminate()
        api_proc.wait()
        st_proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        if api_proc.poll() is not None:
            print("FastAPI exited unexpectedly. Stopping Streamlit.")
            st_proc.terminate()
            break
        if st_proc.poll() is not None:
            print("Streamlit exited unexpectedly. Stopping FastAPI.")
            api_proc.terminate()
            break
        time.sleep(1)
