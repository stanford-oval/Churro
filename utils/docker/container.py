"""Dataclass wrapper for a Docker container instance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
from typing import Any

from .errors import DockerError
from .sdk import container_is_running


@dataclass(slots=True)
class DockerContainer:
    """A lightweight handle to a running Docker container.

    Attributes:
        id: Container ID (hash string).
        name: Container name.
        image: Image reference used to create it.
        auto_remove: Whether container auto-removes itself on exit.
    """

    id: str
    name: str
    image: str
    _container: Any = field(repr=False)
    auto_remove: bool = True

    def is_running(self) -> bool:
        """Return whether the container is still running."""
        return container_is_running(self._container)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the container and remove it if not auto_remove."""
        with contextlib.suppress(Exception):
            self._container.stop(timeout=int(timeout))
        if not self.auto_remove:
            with contextlib.suppress(Exception):
                self._container.remove(force=True)

    def logs(self, tail: int | None = None, since: str | None = None) -> str:
        """Return combined stdout/stderr logs as text snapshot."""
        try:
            logs = self._container.logs(tail=tail, since=since, stdout=True, stderr=True)
            if isinstance(logs, bytes | bytearray):
                return logs.decode("utf-8", errors="replace")
            return str(logs)
        except Exception as e:  # pragma: no cover
            raise DockerError("Failed to retrieve container logs.") from e

    def exec(
        self,
        command: Sequence[str] | str,
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: Mapping[str, str] | None = None,
        demux: bool = False,
    ) -> tuple[int, str]:
        """Execute a command inside the container and return (exit_code, output)."""
        try:
            res = self._container.exec_run(
                cmd=command,
                user=user,
                workdir=workdir,
                environment=dict(environment) if environment else None,
                demux=demux,
            )
        except Exception as e:  # pragma: no cover
            raise DockerError("Failed to exec inside Docker container.") from e

        if hasattr(res, "exit_code") and hasattr(res, "output"):
            exit_code, output = res.exit_code, res.output
        else:  # Docker SDK variant fallback
            try:
                exit_code, output = res  # type: ignore[misc]
            except Exception:  # pragma: no cover
                exit_code, output = 1, b""

        if demux and isinstance(output, tuple):
            out_b = b"" if output[0] is None else output[0]
            err_b = b"" if output[1] is None else output[1]
            combined = out_b + err_b
            text = combined.decode("utf-8", errors="replace")
        else:
            if isinstance(output, bytes | bytearray):
                text = output.decode("utf-8", errors="replace")
            else:
                text = str(output)
        return exit_code, text


__all__ = ["DockerContainer"]
