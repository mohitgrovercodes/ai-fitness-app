"""
Thin HTTP wrapper around the AI Fitness backend.

Responsibilities:
- Read the API base URL from .streamlit/secrets.toml, fall back to localhost.
- Inject the JWT bearer token from st.session_state on every call.
- Unwrap the {"status": True, "data": ...} envelope used by the backend's
  success() helper, returning just the payload to callers.
- Normalize all backend errors (4xx, 5xx, transport) into a single ApiError
  exception with a user-friendly message — no SQLAlchemy / stack text bubbles
  up to UI code.
"""
from typing import Any, Optional

import requests
import streamlit as st


class ApiError(Exception):
    """Raised when the backend returns a non-2xx response or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _base_url() -> str:
    """secrets.toml first, fall back to localhost so the app boots without a secrets file."""
    try:
        return st.secrets["API_BASE_URL"]
    except (FileNotFoundError, KeyError, AttributeError):
        return "http://localhost:8000"


def _headers(extra: Optional[dict] = None) -> dict:
    headers = {"Accept": "application/json"}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _handle(response: requests.Response) -> Any:
    """Unwrap success envelope and surface user-friendly errors."""
    try:
        body = response.json()
    except ValueError:
        body = None

    if not response.ok:
        # FastAPI HTTPException → {"detail": "..."}
        # legacy error() helper → {"status": False, "message": "..."}
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("message") or response.text
        else:
            detail = response.text or f"HTTP {response.status_code}"
        raise ApiError(detail, status_code=response.status_code)

    # success() envelope: {"status": True/False, "message": "...", "data": ...}
    if isinstance(body, dict) and "status" in body and "data" in body:
        if body["status"] is False:
            raise ApiError(
                body.get("message") or "Operation failed",
                status_code=response.status_code,
            )
        return body["data"]

    # Plain dict / list response (e.g. /health, future endpoints)
    return body


def get(path: str, **kwargs) -> Any:
    url = _base_url() + path
    try:
        r = requests.get(
            url,
            headers=_headers(kwargs.pop("headers", None)),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )
    except requests.RequestException as e:
        raise ApiError(f"Could not reach backend at {url}: {e}")
    return _handle(r)


def post(
    path: str,
    *,
    json: Optional[dict] = None,
    files: Optional[dict] = None,
    data: Optional[dict] = None,
    **kwargs,
) -> Any:
    url = _base_url() + path
    try:
        r = requests.post(
            url,
            json=json,
            files=files,
            data=data,
            headers=_headers(kwargs.pop("headers", None)),
            timeout=kwargs.pop("timeout", 120),
            **kwargs,
        )
    except requests.RequestException as e:
        raise ApiError(f"Could not reach backend at {url}: {e}")
    return _handle(r)


def patch(path: str, *, json: Optional[dict] = None, **kwargs) -> Any:
    url = _base_url() + path
    try:
        r = requests.patch(
            url,
            json=json,
            headers=_headers(kwargs.pop("headers", None)),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )
    except requests.RequestException as e:
        raise ApiError(f"Could not reach backend at {url}: {e}")
    return _handle(r)


def delete(path: str, *, json: Optional[dict] = None, **kwargs) -> Any:
    url = _base_url() + path
    try:
        r = requests.delete(
            url,
            json=json,
            headers=_headers(kwargs.pop("headers", None)),
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )
    except requests.RequestException as e:
        raise ApiError(f"Could not reach backend at {url}: {e}")
    return _handle(r)
