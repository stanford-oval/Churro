"""Low-level Docker SDK access and streaming helpers (internal)."""

from __future__ import annotations

import contextlib
from queue import Queue
import threading
from typing import TYPE_CHECKING

from .errors import DockerError


try:  # Local import aliasing to avoid hard failure at import time
    import docker  # type: ignore
except Exception as _e:  # pragma: no cover - host/env dependent
    docker = None  # type: ignore
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


if TYPE_CHECKING:  # pragma: no cover - typing only
    from docker import DockerClient as _DockerClient
    from docker.models.containers import Container as _DockerContainer
else:  # pragma: no cover - runtime fallback when typing info unavailable
    _DockerClient = object
    _DockerContainer = object


def ensure_docker_sdk() -> _DockerClient:
    """Return connected Docker client or raise DockerError with guidance."""
    if _IMPORT_ERROR is not None or docker is None:
        raise DockerError(
            "The 'docker' Python package is required. Install it with: pip install docker"
        ) from _IMPORT_ERROR
    try:
        client = docker.from_env()  # type: ignore[union-attr]
        client.ping()
        return client
    except FileNotFoundError as e:  # pragma: no cover
        raise DockerError(
            "Could not find the Docker socket. Is the Docker daemon running? "
            "Start it (e.g., 'systemctl start docker' or 'colima start' / 'docker desktop') and retry."
        ) from e
    except PermissionError as e:  # pragma: no cover
        raise DockerError(
            "Permission denied accessing the Docker socket. Add your user to the 'docker' group or run with appropriate permissions."
        ) from e
    except Exception as e:  # pragma: no cover
        raise DockerError(
            "Failed to connect to Docker daemon via SDK. Ensure the daemon is running."
        ) from e


def make_device_requests(gpus: str | None) -> object | None:
    """Translate a CLI-style GPU selection string to SDK device requests.

    Supported formats (case-insensitive):
        "all"            -> All available GPUs
        "device=0,1"     -> Explicit device IDs
        "<number>"       -> Request N GPUs (count)
        None / ""        -> No GPU scheduling (returns None)

    Falls back to requesting all GPUs when an unrecognized non-empty string is provided.
    """
    if not gpus:
        return None
    s = gpus.strip().lower()
    from docker.types import DeviceRequest  # type: ignore

    if s == "all":
        return [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    if s.startswith("device="):
        ids = s.split("=", 1)[1].strip()
        dev_ids = [part.strip() for part in ids.split(",") if part.strip()]
        return [DeviceRequest(device_ids=dev_ids, capabilities=[["gpu"]])]
    if s.isdigit():
        return [DeviceRequest(count=int(s), capabilities=[["gpu"]])]
    return [DeviceRequest(count=-1, capabilities=[["gpu"]])]


def container_is_running(container: _DockerContainer) -> bool:
    """Return True if docker container object status is 'running'."""
    with contextlib.suppress(Exception):
        container.reload()
        return getattr(container, "status", None) == "running"
    return False


def stream_logs(
    container: _DockerContainer, line_queue: Queue[str], stop_event: threading.Event
) -> None:
    """Follow container logs and enqueue decoded lines until stopped."""
    try:
        for chunk in container.logs(stream=True, follow=True):
            if stop_event.is_set():
                break
            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                text = str(chunk)
            frames = text.split("\r")
            last_frame = frames[-1]
            for ln in last_frame.splitlines():
                line_queue.put(ln)
    except Exception:
        pass
