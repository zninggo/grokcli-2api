"""Model list helpers + upstream sync."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config import (
    CLI_VERSION,
    CLIENT_IDENTIFIER,
    CLIENT_SURFACE,
    DEFAULT_MODEL,
    MODEL_ALIASES,
    MODELS_CACHE,
    UPSTREAM_BASE,
)


def resolve_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    m = model.strip()
    # grok-search always routes to default model with web search enabled
    if m.lower() in ("grok-search", "web-search"):
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(m, MODEL_ALIASES.get(m.lower(), m))


def load_models_from_cache(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or MODELS_CACHE
    models: list[dict[str, Any]] = []
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            bucket = raw.get("models") or {}
            for mid, meta in bucket.items():
                info = (meta or {}).get("info") or {}
                if info.get("hidden"):
                    continue
                models.append(
                    {
                        "id": info.get("id") or mid,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "xai",
                        "name": info.get("name"),
                        "description": info.get("description"),
                        "context_window": info.get("context_window"),
                        "supports_reasoning_effort": info.get(
                            "supports_reasoning_effort"
                        ),
                    }
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    if not models:
        models = [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai",
            },
            {
                "id": "grok-build",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai",
            },
            {
                "id": "grok-search",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai",
                "description": "Grok with web search enabled",
            },
        ]
    # stable order: default first
    models.sort(key=lambda m: (0 if m.get("id") == DEFAULT_MODEL else 1, m.get("id") or ""))
    return models


def _upstream_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-XAI-Token-Auth": "xai-grok-cli",
        "x-grok-client-version": CLI_VERSION,
        "x-grok-client-surface": CLIENT_SURFACE,
        "x-grok-client-identifier": CLIENT_IDENTIFIER,
        "User-Agent": f"grok-cli/{CLI_VERSION}",
        "Accept": "application/json",
    }


def sync_models_from_upstream(path: Path | None = None) -> dict[str, Any]:
    """
    GET cli-chat-proxy /v1/models and write models_cache.json.
    Uses any live pool account.
    """
    import httpx

    from auth import AuthError
    import account_pool

    path = path or MODELS_CACHE
    try:
        creds = account_pool.acquire()
    except AuthError as e:
        return {"ok": False, "error": str(e)}

    url = f"{UPSTREAM_BASE}/models"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=_upstream_headers(creds.token))
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"network: {e}"}

    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"upstream {resp.status_code}: {(resp.text or '')[:300]}",
            "status_code": resp.status_code,
        }

    try:
        payload = resp.json()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"parse: {e}"}

    data_list = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data_list, list):
        return {"ok": False, "error": "unexpected models payload"}

    bucket: dict[str, Any] = {}
    for item in data_list:
        if not isinstance(item, dict):
            continue
        mid = item.get("id") or item.get("model")
        if not mid:
            continue
        info = {
            "id": mid,
            "model": mid,
            "name": item.get("name") or mid,
            "description": item.get("description"),
            "context_window": item.get("context_window"),
            "supports_reasoning_effort": item.get("supports_reasoning_effort"),
            "hidden": bool(item.get("hidden")),
            "owned_by": item.get("owned_by") or "xai",
        }
        # merge extra known fields
        for k in (
            "max_completion_tokens",
            "reasoning_effort",
            "reasoning_efforts",
            "auto_compact_threshold_percent",
            "supported_in_api",
        ):
            if item.get(k) is not None:
                info[k] = item[k]
        bucket[str(mid)] = {"info": info, "api_key": None, "env_key": None}

    if not bucket:
        return {"ok": False, "error": "no models in upstream response"}

    cache_obj = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grok_version": CLI_VERSION,
        "auth_method": "session",
        "origin": url,
        "models": bucket,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        return {"ok": False, "error": f"write cache failed: {e}"}

    models = load_models_from_cache(path)
    return {
        "ok": True,
        "count": len(models),
        "path": str(path),
        "fetched_via": creds.email or creds.auth_key,
        "models": models,
    }
