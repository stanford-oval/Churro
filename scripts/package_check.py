"""Build, inspect, and smoke-test churro-ocr publish artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email import message_from_string
from email.message import Message
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
EGG_INFO_DIR = ROOT / "src" / "churro_ocr.egg-info"
EXPECTED_PROJECT_URLS = {
    "Homepage": "https://github.com/stanford-oval/Churro",
    "Documentation": "https://stanford-oval.github.io/Churro/",
    "Repository": "https://github.com/stanford-oval/Churro",
    "Issues": "https://github.com/stanford-oval/Churro/issues",
}
EXPECTED_EXTRAS = {
    "all",
    "azure",
    "hf",
    "llm",
    "local",
    "mistral",
    "pdf",
}
FORBIDDEN_ARTIFACT_SEGMENTS = ("/tests/", "/tooling/", "/scripts/")
FORBIDDEN_ARTIFACT_SUFFIXES = ("PYPI_AUDIT.md",)
REQUIREMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+")
ALLOWED_LICENSE_TOKENS = (
    "apache",
    "bsd",
    "isc",
    "mit",
    "mozilla public license",
    "mpl-2.0",
    "python software foundation",
)
INCOMPATIBLE_LICENSE_TOKENS = (
    "agpl",
    "commercial",
    "gnu affero",
    "gpl",
    "lgpl",
)


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args),
        check=True,
        cwd=cwd or ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def _clean_build_artifacts() -> None:
    for path in (BUILD_DIR, DIST_DIR, EGG_INFO_DIR):
        _remove_if_exists(path)


def _build_distributions() -> tuple[Path, Path]:
    _run(sys.executable, "-m", "build", cwd=ROOT)
    wheel = next(DIST_DIR.glob("*.whl"), None)
    sdist = next(DIST_DIR.glob("*.tar.gz"), None)
    if wheel is None or sdist is None:
        raise RuntimeError("Expected both wheel and sdist artifacts in dist/.")
    return wheel, sdist


def _read_wheel_metadata(wheel: Path) -> tuple[Message, str]:
    with zipfile.ZipFile(wheel) as zip_file:
        metadata_name = next(name for name in zip_file.namelist() if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(
            name for name in zip_file.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        metadata_message = message_from_string(zip_file.read(metadata_name).decode("utf-8"))
        entry_points_text = zip_file.read(entry_points_name).decode("utf-8")
    return metadata_message, entry_points_text


def _twine_check(wheel: Path, sdist: Path) -> None:
    _run(sys.executable, "-m", "twine", "check", str(wheel), str(sdist), cwd=ROOT)


def _assert_metadata(metadata_message: Message, entry_points_text: str) -> None:
    name = metadata_message["Name"]
    if name != "churro-ocr":
        raise RuntimeError(f"Unexpected package name {name!r}.")
    if metadata_message["Requires-Python"] != ">=3.12":
        raise RuntimeError("Requires-Python metadata no longer matches the documented support policy.")

    project_urls: dict[str, str] = {}
    for raw_value in metadata_message.get_all("Project-URL", []):
        label, value = raw_value.split(", ", maxsplit=1)
        project_urls[label] = value
    if project_urls != EXPECTED_PROJECT_URLS:
        raise RuntimeError(f"Project URLs do not match the expected package repository: {project_urls!r}.")

    provides_extra = set(metadata_message.get_all("Provides-Extra", []))
    if provides_extra != EXPECTED_EXTRAS:
        raise RuntimeError(f"Unexpected extras set: {sorted(provides_extra)!r}.")

    _assert_local_runtime_packaging_policy(metadata_message)

    if (
        "[console_scripts]" not in entry_points_text
        or "churro-ocr = churro_ocr.cli:main" not in entry_points_text
    ):
        raise RuntimeError("Console script entry point is missing or incorrect.")


def _iter_requirements_for_extra(metadata_message: Message, extra: str) -> list[Requirement]:
    requirements: list[Requirement] = []
    for requirement_text in metadata_message.get_all("Requires-Dist", []):
        parsed = Requirement(requirement_text)
        marker_text = str(parsed.marker) if parsed.marker is not None else ""
        if f'extra == "{extra}"' in marker_text:
            requirements.append(parsed)
    return requirements


def _assert_local_runtime_packaging_policy(metadata_message: Message) -> None:
    disallowed_runtime_reqs: list[str] = []
    for extra in EXPECTED_EXTRAS:
        for requirement in _iter_requirements_for_extra(metadata_message, extra):
            normalized_name = requirement.name.replace("_", "-").lower()
            if normalized_name in {"torch", "torchvision", "vllm"}:
                disallowed_runtime_reqs.append(f"{extra}:{requirement}")
                continue
            if normalized_name == "transformers" and "torch" in requirement.extras:
                disallowed_runtime_reqs.append(f"{extra}:{requirement}")
    if disallowed_runtime_reqs:
        formatted = ", ".join(sorted(disallowed_runtime_reqs))
        raise RuntimeError(
            "PyPI extras for active-environment runtimes must not pin local PyTorch or vLLM runtimes. "
            f"Found disallowed requirements: {formatted}."
        )


def _assert_runtime_only_artifacts(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as zip_file:
        wheel_names = zip_file.namelist()
    for name in wheel_names:
        normalized = f"/{name}"
        if any(segment in normalized for segment in FORBIDDEN_ARTIFACT_SEGMENTS):
            raise RuntimeError(f"Wheel unexpectedly includes repo-only content: {name}")
        if any(normalized.endswith(suffix) for suffix in FORBIDDEN_ARTIFACT_SUFFIXES):
            raise RuntimeError(f"Wheel unexpectedly includes repo-only documentation: {name}")

    with tarfile.open(sdist) as tar_file:
        sdist_names = tar_file.getnames()
    for name in sdist_names:
        normalized = f"/{name}"
        if any(segment in normalized for segment in FORBIDDEN_ARTIFACT_SEGMENTS):
            raise RuntimeError(f"sdist unexpectedly includes repo-only content: {name}")
        if any(normalized.endswith(suffix) for suffix in FORBIDDEN_ARTIFACT_SUFFIXES):
            raise RuntimeError(f"sdist unexpectedly includes repo-only documentation: {name}")


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _smoke_install(requirement: str, *, label: str, import_check: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"churro-{label}-") as temp_dir:
        temp_path = Path(temp_dir)
        venv_dir = temp_path / "venv"
        workspace_dir = temp_path / "workspace"
        workspace_dir.mkdir()

        _run(sys.executable, "-m", "venv", str(venv_dir), cwd=workspace_dir)
        python = _venv_python(venv_dir)
        _run(str(python), "-m", "pip", "install", "--upgrade", "pip", cwd=workspace_dir)
        _run(str(python), "-m", "pip", "install", requirement, cwd=workspace_dir)
        _run(str(python), "-c", import_check, cwd=workspace_dir)
        if (workspace_dir / "debug.log").exists():
            raise RuntimeError(f"{label} created an unexpected debug.log file.")
        _run(str(python), "-m", "churro_ocr", "--help", cwd=workspace_dir)
        if (workspace_dir / "debug.log").exists():
            raise RuntimeError(f"{label} CLI help created an unexpected debug.log file.")


def _requirement_name(requirement: str) -> str | None:
    match = REQUIREMENT_NAME_PATTERN.match(requirement)
    if match is None:
        return None
    return match.group(0).replace("_", "-").lower()


def _audited_requirement(requirement: str) -> tuple[str, bool] | None:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        name = _requirement_name(requirement)
        if name is None:
            return None
        return name, True

    marker = parsed.marker
    marker_text = str(marker) if marker is not None else ""
    if marker is not None and "extra" not in marker_text and not marker.evaluate():
        return None

    is_optional_extra = marker is not None and "extra" in marker_text
    return parsed.name.replace("_", "-").lower(), not is_optional_extra


def _direct_dependencies_to_audit(metadata_message: Message) -> dict[str, bool]:
    direct_dependencies: dict[str, bool] = {}
    for requirement in metadata_message.get_all("Requires-Dist", []):
        parsed = _audited_requirement(requirement)
        if parsed is None:
            continue

        name, is_required = parsed
        direct_dependencies[name] = direct_dependencies.get(name, False) or is_required
    return direct_dependencies


def _read_distribution_license_text(distribution: metadata.Distribution) -> str:
    metadata_message = distribution.metadata
    parts: list[str] = []

    for key in ("License-Expression", "License"):
        parts.extend(value for value in metadata_message.get_all(key, []) if value)
    parts.extend(
        classifier
        for classifier in metadata_message.get_all("Classifier", [])
        if classifier.startswith("License ::")
    )

    for license_file in metadata_message.get_all("License-File", []):
        path = distribution.locate_file(license_file)
        if path.exists():
            parts.append(path.read_text(errors="ignore")[:4_000])

    if parts:
        return "\n".join(parts).lower()

    for file_path in distribution.files or []:
        lowered = str(file_path).lower()
        if "license" not in lowered and "copying" not in lowered:
            continue
        path = distribution.locate_file(file_path)
        if path.exists():
            parts.append(path.read_text(errors="ignore")[:4_000])
    return "\n".join(parts).lower()


def _audit_dependency_licenses(metadata_message: Message) -> None:
    direct_dependencies = _direct_dependencies_to_audit(metadata_message)
    incompatible: list[str] = []
    unknown: list[str] = []
    for dependency_name in sorted(direct_dependencies):
        is_required = direct_dependencies[dependency_name]
        try:
            distribution = metadata.distribution(dependency_name)
        except metadata.PackageNotFoundError:
            if not is_required:
                continue
            unknown.append(f"{dependency_name} (not installed in the Pixi audit environment)")
            continue

        license_text = _read_distribution_license_text(distribution)
        if any(token in license_text for token in INCOMPATIBLE_LICENSE_TOKENS):
            incompatible.append(f"{dependency_name}=={distribution.version}")
            continue
        if any(token in license_text for token in ALLOWED_LICENSE_TOKENS):
            continue
        unknown.append(f"{dependency_name}=={distribution.version}")

    if incompatible:
        raise RuntimeError(
            "Incompatible direct dependency licenses detected: " + ", ".join(incompatible) + "."
        )
    if unknown:
        raise RuntimeError("Unknown direct dependency licenses detected: " + ", ".join(unknown) + ".")


def main() -> int:
    print("==> Cleaning build artifacts")
    _clean_build_artifacts()

    print("==> Building wheel and sdist")
    wheel, sdist = _build_distributions()

    print("==> Running twine check")
    _twine_check(wheel, sdist)

    print("==> Validating wheel metadata")
    metadata_message, entry_points_text = _read_wheel_metadata(wheel)
    _assert_metadata(metadata_message, entry_points_text)

    print("==> Validating artifact contents")
    _assert_runtime_only_artifacts(wheel, sdist)

    print("==> Smoke-testing base wheel install")
    _smoke_install(
        wheel.resolve().as_uri(),
        label="wheel",
        import_check=(
            "import churro_ocr; import churro_ocr.providers as providers; "
            "assert hasattr(providers, 'OCRBackendSpec')"
        ),
    )

    print("==> Smoke-testing base sdist install")
    _smoke_install(
        sdist.resolve().as_uri(),
        label="sdist",
        import_check=(
            "import churro_ocr; import churro_ocr.providers as providers; "
            "assert hasattr(providers, 'OCRBackendSpec')"
        ),
    )

    print("==> Smoke-testing lightweight extras")
    wheel_uri = wheel.resolve().as_uri()
    _smoke_install(
        f"churro-ocr[local] @ {wheel_uri}",
        label="wheel-local",
        import_check="import churro_ocr; import litellm",
    )
    _smoke_install(
        f"churro-ocr[pdf] @ {wheel_uri}",
        label="wheel-pdf",
        import_check="import churro_ocr; import pypdfium2",
    )

    print("==> Auditing direct dependency licenses")
    _audit_dependency_licenses(metadata_message)

    print("==> package-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
