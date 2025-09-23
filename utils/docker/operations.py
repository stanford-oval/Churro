"""Core container lifecycle operations: start, wait for readiness."""

from __future__ import annotations

from collections import deque
import contextlib
import os
from queue import Empty, Queue
import re
import threading
import time
from typing import Any, Mapping, Optional, Pattern, Sequence

from utils.log_utils import logger

from .container import DockerContainer
from .errors import DockerError
from .logging_utils import format_prefix, strip_ansi
from .sdk import (
    ensure_docker_sdk,
    make_device_requests,
    stream_logs,
)


__all__ = [
    "start_container",
    "wait_for_readiness",
    "start_and_wait_ready",
]


def start_container(
    *,
    image: str,
    name: Optional[str] = None,
    gpus: Optional[str] = None,
    volumes: Optional[Mapping[str, str]] = None,
    ports: Optional[Mapping[int, int]] = None,
    env: Optional[Mapping[str, str]] = None,
    ipc: Optional[str] = None,
    shm_size: Optional[str] = None,
    network: Optional[str] = None,
    user: Optional[str] = None,
    auto_remove: bool = True,
    cmd: Optional[Sequence[str]] = None,
    force_replace: bool = False,
    pull: bool = True,
) -> DockerContainer:
    """Start a detached Docker container and return a handle.

    Args:
        image: Image reference (``repository[:tag]``) to run.
        name: Optional explicit container name.
        gpus: GPU selection string (e.g. ``"all"``, ``"device=0,1"``, ``"1"``) or ``None``.
        volumes: Host path -> container path mappings (rw mode).
        ports: Host port -> container internal port mappings (tcp assumed).
        env: Environment variables to inject.
        ipc: IPC mode (e.g. ``host``).
        shm_size: Shared memory size (e.g. ``"4g"``) useful for some workloads.
        network: Optional network name.
        user: Run container processes as this user string (``UID[:GID]`` form).
        auto_remove: If True, Docker auto-removes the container on exit.
        cmd: Override default image command with this sequence.
        force_replace: If True and ``name`` is provided, attempt to stop/remove
            any existing container with that name before starting.
        pull: If True, ensure the image is present locally (pull when missing).

    Returns:
        DockerContainer: Lightweight wrapper handle for further operations.

    Raises:
        DockerError: On image pull failure or container creation failure.
    """
    client = ensure_docker_sdk()

    if pull:  # ensure image present
        try:
            try:
                client.images.get(image)
            except Exception:
                logger.info(
                    f"Pulling Docker image '{image}' ... this may take a while on first use"
                )
                api_client = getattr(client, "api", None)
                if api_client and hasattr(api_client, "pull"):
                    api_client.pull(image)  # type: ignore[arg-type]
                else:  # pragma: no cover
                    client.images.pull(image)
        except DockerError:
            raise
        except Exception as e:  # pragma: no cover
            raise DockerError(
                f"Failed to pull image '{image}'. Check network, image name, or registry auth."
            ) from e

    if force_replace and name:
        for attempt in range(5):
            try:
                existing = client.containers.list(all=True, filters={"name": name})
            except Exception:
                break
            removed_any = False
            for c in existing:
                if getattr(c, "name", "") == name:
                    removed_any = True
                    with contextlib.suppress(Exception):
                        logger.debug(
                            f"Stopping existing container '{name}' (attempt {attempt + 1})"
                        )
                        c.stop(timeout=5)
                    with contextlib.suppress(Exception):
                        logger.debug(
                            f"Removing existing container '{name}' (attempt {attempt + 1})"
                        )
                        c.remove(force=True)
            if not removed_any:
                break
            time.sleep(0.4)

    ports_map: dict[str, int] | None = None
    if ports:
        ports_map = {f"{container}/tcp": host for host, container in ports.items()}

    volumes_map: dict[str, dict[str, str]] | None = None
    if volumes:
        volumes_map = {os.path.expanduser(h): {"bind": c, "mode": "rw"} for h, c in volumes.items()}

    device_requests = make_device_requests(gpus)

    def _run_container() -> Any:
        return client.containers.run(
            image=image,
            name=name,
            environment=dict(env) if env else None,
            volumes=volumes_map,
            ports=ports_map,
            ipc_mode=ipc,
            shm_size=shm_size,
            network=network,
            user=user,
            remove=auto_remove,
            detach=True,
            command=list(cmd) if cmd else None,
            device_requests=device_requests,
        )

    try:
        container = _run_container()
    except Exception as e:
        conflict_msg = str(e)
        if force_replace and name and "Conflict" in conflict_msg:
            with contextlib.suppress(Exception):
                logger.debug(f"Retrying container create after conflict on name '{name}'")
                existing = client.containers.list(all=True, filters={"name": name})
                for c in existing:
                    if getattr(c, "name", "") == name:
                        with contextlib.suppress(Exception):
                            c.stop(timeout=3)
                        with contextlib.suppress(Exception):
                            c.remove(force=True)
                time.sleep(0.6)
            try:
                container = _run_container()
            except Exception as e2:  # pragma: no cover
                raise DockerError(
                    f"Failed to start Docker container after force_replace retry (name='{name}')."
                ) from e2
        else:
            raise DockerError("Failed to start Docker container via SDK.") from e

    resolved_name = getattr(container, "name", name or "") or (name or "")
    return DockerContainer(
        id=getattr(container, "id", ""),
        name=resolved_name,
        image=image,
        auto_remove=auto_remove,
        _container=container,
    )


def wait_for_readiness(
    container: DockerContainer,
    *,
    ready_pattern: str | Pattern[str],
    ready_timeout: float = 120.0,
    check_interval: float = 0.5,
    capture_tail_lines: int = 4000,
    log_prefix: Optional[str] = None,
) -> None:
    """Block until container logs match regex or fail.

    Args:
        container: Running container handle.
        ready_pattern: Regex pattern or string to signal readiness.
        ready_timeout: Inactivity timeout in seconds (time since last log line).
        check_interval: Poll interval for waiting on new log lines.
        capture_tail_lines: Maximum number of log lines retained internally.
        log_prefix: Optional prefix prepended to each log line when printing.

    Behavior:
        Uses an *inactivity timeout* semantic: the timer resets on every new
        log line. If no new line arrives for ``ready_timeout`` seconds before
        the readiness pattern is detected, a ``TimeoutError`` is raised.

    Raises:
        TimeoutError: If inactivity timeout elapses first.
        DockerError: If container exits prematurely before readiness.
    """
    pattern: Pattern[str] = (
        re.compile(ready_pattern) if isinstance(ready_pattern, str) else ready_pattern
    )

    line_queue: Queue[str] = Queue(maxsize=10000)
    stop_event = threading.Event()
    tail = deque(maxlen=capture_tail_lines)

    t = threading.Thread(
        target=stream_logs,
        args=(container._container, line_queue, stop_event),
        daemon=True,
    )
    t.start()

    # Track time of last log line (or start time if none yet). If we exceed
    # ready_timeout since the last log activity, we fail.
    last_log_time = time.time()
    try:
        while True:
            try:
                line = line_queue.get(timeout=check_interval)
                sanitized = strip_ansi(line)
                tail.append(sanitized)
                logger.info(f"{format_prefix(log_prefix)}{sanitized}")
                if pattern.search(sanitized):
                    stop_event.set()
                    t.join(timeout=1.0)
                    return
                # Update last activity time after processing the line
                last_log_time = time.time()
            except Empty:
                if not container.is_running():
                    stop_event.set()
                    t.join(timeout=1.0)
                    with contextlib.suppress(Exception):
                        container._container.remove(force=True)
                    raise DockerError("Container exited before readiness was detected.")
                # No new line; check inactivity timeout
                if time.time() - last_log_time > ready_timeout:
                    stop_event.set()
                    t.join(timeout=1.0)
                    with contextlib.suppress(Exception):
                        container._container.remove(force=True)
                    raise TimeoutError(
                        "No new container logs received within inactivity timeout before readiness pattern matched."
                    )
    finally:
        pass


def start_and_wait_ready(
    *,
    image: str,
    name: Optional[str] = None,
    gpus: Optional[str] = None,
    volumes: Optional[Mapping[str, str]] = None,
    ports: Optional[Mapping[int, int]] = None,
    env: Optional[Mapping[str, str]] = None,
    ipc: Optional[str] = None,
    shm_size: Optional[str] = None,
    network: Optional[str] = None,
    user: Optional[str] = None,
    auto_remove: bool = True,
    cmd: Optional[Sequence[str]] = None,
    ready_pattern: (
        str | Pattern[str]
    ) = r"Ready|started|listening|Uvicorn running|Application startup complete",
    ready_timeout: float = 180.0,
    check_interval: float = 0.5,
    log_prefix: Optional[str] = None,
    pull: bool = True,
) -> DockerContainer:
    """Convenience helper to start a container then wait for readiness.

    Note: ``ready_timeout`` is an inactivity timeout (see ``wait_for_readiness``).
    """
    container = start_container(
        image=image,
        name=name,
        gpus=gpus,
        volumes=volumes,
        ports=ports,
        env=env,
        ipc=ipc,
        shm_size=shm_size,
        network=network,
        user=user,
        auto_remove=auto_remove,
        cmd=cmd,
        pull=pull,
    )
    try:
        wait_for_readiness(
            container,
            ready_pattern=ready_pattern,
            ready_timeout=ready_timeout,
            check_interval=check_interval,
            log_prefix=log_prefix,
        )
    except Exception:
        with contextlib.suppress(Exception):
            container.stop()
        raise
    return container
