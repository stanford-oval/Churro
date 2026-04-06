from email.message import Message
from importlib import metadata as importlib_metadata
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_package_check_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "package_check.py"
    spec = spec_from_file_location("package_check", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_message(*requirements: str) -> Message:
    message = Message()
    for requirement in requirements:
        message.add_header("Requires-Dist", requirement)
    return message


package_check = _load_package_check_module()


def test_license_audit_skips_missing_optional_extra_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _always_missing(_: str):
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(package_check.metadata, "distribution", _always_missing)

    package_check._audit_dependency_licenses(_metadata_message('mistralai<2,>=1.6.0; extra == "mistral"'))


def test_license_audit_fails_for_missing_base_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _always_missing(_: str):
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(package_check.metadata, "distribution", _always_missing)

    with pytest.raises(RuntimeError, match="pillow \\(not installed in the Pixi audit environment\\)"):
        package_check._audit_dependency_licenses(_metadata_message("Pillow<12,>=10.4.0"))


def test_local_runtime_packaging_policy_rejects_direct_torch_runtime_pin() -> None:
    metadata_message = _metadata_message(
        'transformers[torch]>=5,<6; extra == "hf"',
        'torchvision; extra == "all"',
    )

    with pytest.raises(RuntimeError, match="must not pin local PyTorch"):
        package_check._assert_local_runtime_packaging_policy(metadata_message)
