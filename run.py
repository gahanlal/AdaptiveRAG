"""
run.py — Single entry point.
Starts FastAPI (uvicorn :8000) and Streamlit (:8501) as subprocesses.
Press Ctrl+C to stop both.

Secret precedence (highest → lowest):
  1. .streamlit/secrets.toml  (your API keys — never commit this file)
  2. .env                     (provider/model defaults — safe to commit)
  3. existing environment variables
"""
import subprocess
import sys
import os
import re
import signal
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_toml_secrets(base_env: dict) -> dict:
    """
    Parse .streamlit/secrets.toml and overlay its values onto base_env.
    Handles simple scalar lines:  KEY = "value"  |  KEY = 'value'  |  KEY = value
    Silently skips sections ([headers]), comments, and complex values.
    """
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
                if value and value != "sk-...":   # skip unfilled placeholders
                    env[key] = value
    return env


def main() -> None:
    print("Starting Adaptive RAG Configurator…\n")

    # Load .env first (low priority), then overlay .streamlit/secrets.toml
    from dotenv import dotenv_values
    env_file = os.path.join(ROOT, ".env")
    base_env = {**os.environ, **dotenv_values(env_file)}
    merged_env = _load_toml_secrets(base_env)
    merged_env["PYTHONPATH"] = ROOT

    provider = merged_env.get("LLM_PROVIDER", "ollama")
    model = merged_env.get(
        {"openai": "OPENAI_MODEL", "groq": "GROQ_MODEL", "custom": "CUSTOM_MODEL"}.get(provider, "OLLAMA_MODEL"),
        "?"
    )
    print(f"  LLM provider : {provider}  |  model : {model}")

    # FastAPI backend
    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=ROOT,
        env=merged_env,
    )

    # Brief pause so API is up before Streamlit renders
    time.sleep(2)

    # Streamlit frontend
    st_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(ROOT, "frontend", "app.py"),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
        ],
        cwd=ROOT,
        env=merged_env,
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

    # Wait for either process to exit unexpectedly
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


if __name__ == "__main__":
    main()
