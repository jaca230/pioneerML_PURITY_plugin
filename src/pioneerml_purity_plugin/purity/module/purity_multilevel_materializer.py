from __future__ import annotations

from pathlib import Path

import torch
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

from .purity_multilevel_module import PurityMultiLevelLightningModule


class PurityMultiLevelLightningModuleMaterializer(BaseMaterializer):
    """ZenML materializer for PURITY LightningModule artifacts."""

    SKIP_REGISTRATION = False
    ASSOCIATED_TYPES = (PurityMultiLevelLightningModule,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: type):
        _ = data_type
        path = Path(self.uri) / "module.pt"
        return torch.load(path, weights_only=False, map_location="cpu")

    def save(self, module: PurityMultiLevelLightningModule) -> None:
        path = Path(self.uri)
        path.mkdir(parents=True, exist_ok=True)
        try:
            module = module.to("cpu")
        except Exception:
            pass
        torch.save(module, path / "module.pt")
