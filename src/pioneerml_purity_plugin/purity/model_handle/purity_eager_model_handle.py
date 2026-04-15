from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pioneerml.integration.pytorch.model_handles import BaseModelHandle
from pioneerml.integration.pytorch.model_handles import MODEL_HANDLE_REGISTRY

from ..model import PurityModel


@MODEL_HANDLE_REGISTRY.register("purity_eager")
@MODEL_HANDLE_REGISTRY.register("purity_state_dict")
class PurityEagerModelHandle(BaseModelHandle):
    """Loads PURITY eager weights/state and materializes a native `PurityModel`."""

    TYPE = "purity_eager"

    @staticmethod
    def _candidate_bundle_paths(path: Path) -> list[Path]:
        candidates: list[Path] = []

        derived = PurityModel.state_bundle_path_for(path)
        candidates.append(derived)
        candidates.append(path)

        name = path.name
        if name.endswith("_torchscript.pt"):
            prefix = name[: -len("_torchscript.pt")]
            candidates.append(path.with_name(f"{prefix}_eager_state.pt"))
            candidates.append(path.with_name(f"{prefix}_eager.pt"))

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(resolved)
        return unique

    def _load_bundle(self) -> tuple[Path, Mapping[str, Any]]:
        search_paths = self._candidate_bundle_paths(self.path)
        attempted: list[str] = []
        errors: list[str] = []

        for bundle_path in search_paths:
            attempted.append(str(bundle_path))
            if not bundle_path.exists():
                continue
            try:
                payload = torch.load(str(bundle_path), map_location="cpu")
            except Exception as exc:  # pragma: no cover - defensive path for non-bundle files.
                errors.append(f"{bundle_path}: {exc.__class__.__name__}: {exc}")
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
                return bundle_path, payload

        message = "No PURITY eager state bundle found. Expected one of: " + ", ".join(attempted)
        if errors:
            message = message + " | load errors: " + " ; ".join(errors)
        raise FileNotFoundError(message)

    def load(self, *, device: torch.device):
        _, bundle = self._load_bundle()
        raw_arch = bundle.get("architecture")
        architecture = dict(raw_arch) if isinstance(raw_arch, Mapping) else {}

        model = PurityModel(**architecture)
        state_dict = bundle.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("Invalid PURITY state bundle: 'state_dict' must be a mapping.")

        load_result = model.load_state_dict(state_dict, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                "Failed to restore PURITY state dict strictly. "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )

        model.eval()
        model.to(device)
        return model
