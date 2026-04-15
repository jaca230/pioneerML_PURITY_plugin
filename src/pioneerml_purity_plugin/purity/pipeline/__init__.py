from __future__ import annotations

import json
from pathlib import Path

from pioneerml.pipeline.pipelines.inference import inference_pipeline

from .training import training_pipeline

MODEL_KEY = "purity"


def load_config() -> dict:
    cfg_path = Path(__file__).with_name("config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing PURITY pipeline config: {cfg_path}")
    return dict(json.loads(cfg_path.read_text(encoding="utf-8")))


__all__ = ["MODEL_KEY", "training_pipeline", "inference_pipeline", "load_config"]
