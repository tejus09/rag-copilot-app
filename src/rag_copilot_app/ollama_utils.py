"""Friendly health checks for the local Ollama daemon, so the UI can explain
what's wrong instead of surfacing a raw connection error."""

from __future__ import annotations

import requests

DEFAULT_HOST = "http://localhost:11434"


def is_ollama_running(host: str = DEFAULT_HOST) -> bool:
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_pulled_models(host: str = DEFAULT_HOST) -> list[str]:
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException:
        return []


def is_model_pulled(model: str, host: str = DEFAULT_HOST) -> bool:
    return model in list_pulled_models(host)
