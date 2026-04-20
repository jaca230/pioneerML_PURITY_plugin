from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pioneerml.integration.pytorch.trainers import TrainerFactory
from pioneerml.pipeline.steps import BaseFullTrainingStep
from pioneerml.pipeline.steps.step_types.model_runner.utils import (
    log_loader_diagnostics,
    merge_nested_dicts,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PurityPhase:
    name: str
    max_epochs: int | None
    task_weights: dict[str, float] | None
    freeze_all: bool
    trainable_param_patterns: tuple[str, ...]
    freeze_param_patterns: tuple[str, ...]
    unfreeze_param_patterns: tuple[str, ...]
    trainer_kwargs: dict[str, Any] | None
    early_stopping: dict[str, Any] | None


@dataclass(frozen=True)
class PurityStagedTrainingConfig:
    enabled: bool
    phases: tuple[PurityPhase, ...]
    strict_param_pattern_match: bool
    reset_task_weights_after_training: bool
    restore_param_trainability_after_training: bool


class PurityStagedTrainingStep(BaseFullTrainingStep):
    step_key = "train"

    def _execute(self):
        self.apply_warning_filter()
        module = self.runtime_state.get("module")
        trainer = self.runtime_state.get("trainer")
        train_loader = self.runtime_state.get("train_loader")
        val_loader = self.runtime_state.get("val_loader")
        train_provider = self.runtime_state.get("train_provider")
        val_provider = self.runtime_state.get("val_provider")
        train_params = self.runtime_state.get("train_params")
        val_params = self.runtime_state.get("val_params")
        training_context = self.runtime_state.get("training_context")
        hpo_params = self.runtime_state.get("hpo_params")
        upstream_payloads = self.runtime_state.get("upstream_payloads")

        if module is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'module'.")
        if trainer is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing 'trainer'.")
        if train_loader is None or val_loader is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val loaders.")
        if train_provider is None or val_provider is None:
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val providers.")
        if not isinstance(train_params, dict) or not isinstance(val_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing train/val params.")
        if not isinstance(training_context, str) or training_context == "":
            raise RuntimeError(f"{self.__class__.__name__} runtime_state missing valid 'training_context'.")
        if hpo_params is not None and not isinstance(hpo_params, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'hpo_params'.")
        if upstream_payloads is not None and not isinstance(upstream_payloads, dict):
            raise RuntimeError(f"{self.__class__.__name__} runtime_state has invalid 'upstream_payloads'.")

        staged_cfg = self._resolve_staged_training_config()
        phase_summaries: list[dict[str, Any]] = []
        phase_loss_histories: list[dict[str, Any]] = []
        if not staged_cfg.enabled or len(staged_cfg.phases) == 0:
            trainer.fit(
                model=module,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
        else:
            original_trainability = self._snapshot_trainability(module=module)
            try:
                for phase_index, phase in enumerate(staged_cfg.phases, start=1):
                    # Omar parity note:
                    # unified_reco training is staged by task weights, so we keep
                    # explicit phase boundaries and preserve per-phase loss slices
                    # for debugging/validation and notebook visualization.
                    train_hist_before = len(list(getattr(module, "train_epoch_loss_history", []) or []))
                    val_hist_before = len(list(getattr(module, "val_epoch_loss_history", []) or []))
                    trainability = self._apply_phase_trainability(
                        module=module,
                        phase=phase,
                        strict_match=staged_cfg.strict_param_pattern_match,
                    )
                    self._apply_phase_task_weights(module=module, task_weights=phase.task_weights)
                    phase_trainer = self._build_phase_trainer(phase=phase)
                    LOGGER.info(
                        "[purity_staged_training] phase=%s configured_max_epochs=%s",
                        phase.name,
                        int(phase.max_epochs) if phase.max_epochs is not None else self._resolved_base_max_epochs(),
                    )
                    phase_trainer.fit(
                        model=module,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                    )
                    train_hist_full = list(getattr(module, "train_epoch_loss_history", []) or [])
                    val_hist_full = list(getattr(module, "val_epoch_loss_history", []) or [])
                    phase_train_losses = [float(v) for v in train_hist_full[train_hist_before:]]
                    phase_val_losses = [float(v) for v in val_hist_full[val_hist_before:]]
                    LOGGER.info(
                        "[purity_staged_training] phase=%s train_loss_points=%s val_loss_points=%s",
                        phase.name,
                        len(phase_train_losses),
                        len(phase_val_losses),
                    )
                    phase_loss_histories.append(
                        {
                            "index": int(phase_index),
                            "name": str(phase.name),
                            "train_losses": phase_train_losses,
                            "val_losses": phase_val_losses,
                        }
                    )
                    phase_summaries.append(
                        {
                            "index": int(phase_index),
                            "name": str(phase.name),
                            "max_epochs": (
                                int(phase.max_epochs)
                                if phase.max_epochs is not None
                                else self._resolved_base_max_epochs()
                            ),
                            "task_weights": dict(phase.task_weights or {}),
                            "trainable_parameters": int(trainability["trainable_parameters"]),
                            "frozen_parameters": int(trainability["frozen_parameters"]),
                            "train_loss_points": len(phase_train_losses),
                            "val_loss_points": len(phase_val_losses),
                        }
                    )
            finally:
                if staged_cfg.reset_task_weights_after_training:
                    self._apply_phase_task_weights(module=module, task_weights=None)
                if staged_cfg.restore_param_trainability_after_training:
                    self._restore_trainability(module=module, snapshot=original_trainability)
        if phase_loss_histories:
            # Persist on module so downstream notebook/debug tooling can render
            # one curve per staged phase without re-running training.
            setattr(module, "staged_phase_loss_histories", phase_loss_histories)

        if bool(train_params.get("log_diagnostics", False)):
            log_loader_diagnostics(label="train", loader_provider=train_provider)
        if bool(val_params.get("log_diagnostics", False)):
            log_loader_diagnostics(label="val", loader_provider=val_provider)

        payload = self.build_payload(
            module=module,
            training_context=training_context,
            hpo_params=hpo_params,
            upstream_payloads=upstream_payloads,
        )
        if phase_summaries:
            payload.with_extra_info(
                staged_training={
                    "enabled": True,
                    "num_phases": len(phase_summaries),
                    "phases": phase_summaries,
                }
            )
        return payload

    def _resolve_staged_training_config(self) -> PurityStagedTrainingConfig:
        raw = self.config_json.get("staged_training")
        if raw is None:
            return PurityStagedTrainingConfig(
                enabled=False,
                phases=(),
                strict_param_pattern_match=False,
                reset_task_weights_after_training=True,
                restore_param_trainability_after_training=True,
            )
        if not isinstance(raw, Mapping):
            raise TypeError("training.train.staged_training must be a mapping when provided.")

        enabled = bool(raw.get("enabled", False))
        strict_patterns = bool(raw.get("strict_param_pattern_match", False))
        reset_weights = bool(raw.get("reset_task_weights_after_training", True))
        restore_trainability = bool(raw.get("restore_param_trainability_after_training", True))
        raw_phases = raw.get("phases", [])
        if raw_phases is None:
            raw_phases = []
        if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes, bytearray)):
            raise TypeError("training.train.staged_training.phases must be a list.")

        phases = tuple(self._normalize_phase(idx=index, raw_phase=phase) for index, phase in enumerate(raw_phases))
        return PurityStagedTrainingConfig(
            enabled=enabled,
            phases=phases,
            strict_param_pattern_match=strict_patterns,
            reset_task_weights_after_training=reset_weights,
            restore_param_trainability_after_training=restore_trainability,
        )

    @staticmethod
    def _normalize_phase(*, idx: int, raw_phase: Any) -> PurityPhase:
        context = f"training.train.staged_training.phases[{idx}]"
        if not isinstance(raw_phase, Mapping):
            raise TypeError(f"{context} must be a mapping.")
        phase = dict(raw_phase)

        raw_name = phase.get("name")
        if raw_name is None:
            name = f"phase_{idx + 1}"
        else:
            name = str(raw_name).strip()
            if name == "":
                raise ValueError(f"{context}.name must be non-empty when provided.")

        max_epochs = phase.get("max_epochs")
        if max_epochs is None:
            parsed_epochs = None
        else:
            parsed_epochs = int(max_epochs)
            if parsed_epochs <= 0:
                raise ValueError(f"{context}.max_epochs must be > 0 when provided.")

        task_weights = PurityStagedTrainingStep._normalize_optional_float_mapping(
            phase.get("task_weights"),
            context=f"{context}.task_weights",
        )
        trainer_kwargs = PurityStagedTrainingStep._normalize_optional_mapping(
            phase.get("trainer_kwargs"),
            context=f"{context}.trainer_kwargs",
        )
        early_stopping = PurityStagedTrainingStep._normalize_optional_mapping(
            phase.get("early_stopping"),
            context=f"{context}.early_stopping",
        )
        trainable_patterns = PurityStagedTrainingStep._normalize_optional_string_list(
            phase.get("trainable_param_patterns"),
            context=f"{context}.trainable_param_patterns",
        )
        freeze_patterns = PurityStagedTrainingStep._normalize_optional_string_list(
            phase.get("freeze_param_patterns"),
            context=f"{context}.freeze_param_patterns",
        )
        unfreeze_patterns = PurityStagedTrainingStep._normalize_optional_string_list(
            phase.get("unfreeze_param_patterns"),
            context=f"{context}.unfreeze_param_patterns",
        )

        return PurityPhase(
            name=name,
            max_epochs=parsed_epochs,
            task_weights=task_weights,
            freeze_all=bool(phase.get("freeze_all", False)),
            trainable_param_patterns=tuple(trainable_patterns),
            freeze_param_patterns=tuple(freeze_patterns),
            unfreeze_param_patterns=tuple(unfreeze_patterns),
            trainer_kwargs=trainer_kwargs,
            early_stopping=early_stopping,
        )

    @staticmethod
    def _normalize_optional_mapping(raw: Any, *, context: str) -> dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context} must be a mapping when provided.")
        return dict(raw)

    @staticmethod
    def _normalize_optional_string_list(raw: Any, *, context: str) -> list[str]:
        if raw is None:
            return []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TypeError(f"{context} must be a list of strings when provided.")
        out: list[str] = []
        for index, value in enumerate(raw):
            if not isinstance(value, str):
                raise TypeError(f"{context}[{index}] must be a string.")
            pattern = value.strip()
            if pattern == "":
                raise ValueError(f"{context}[{index}] must be non-empty.")
            out.append(pattern)
        return out

    @staticmethod
    def _normalize_optional_float_mapping(raw: Any, *, context: str) -> dict[str, float] | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context} must be a mapping when provided.")
        out: dict[str, float] = {}
        for key, value in dict(raw).items():
            key_str = str(key).strip()
            if key_str == "":
                raise ValueError(f"{context} keys must be non-empty strings.")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{context}[{key_str!r}] must be finite.")
            out[key_str] = numeric
        return out

    @staticmethod
    def _snapshot_trainability(*, module) -> dict[str, bool]:
        return {name: bool(param.requires_grad) for name, param in module.named_parameters()}

    @staticmethod
    def _restore_trainability(*, module, snapshot: Mapping[str, bool]) -> None:
        for name, param in module.named_parameters():
            if name in snapshot:
                param.requires_grad = bool(snapshot[name])

    def _apply_phase_trainability(
        self,
        *,
        module,
        phase: PurityPhase,
        strict_match: bool,
    ) -> dict[str, int]:
        named_params = list(module.named_parameters())
        if len(named_params) == 0:
            return {"trainable_parameters": 0, "frozen_parameters": 0}

        if bool(phase.freeze_all):
            for _, param in named_params:
                param.requires_grad = False

        trainable_patterns = list(phase.trainable_param_patterns or [])
        if trainable_patterns:
            for _, param in named_params:
                param.requires_grad = False
            for pattern in trainable_patterns:
                matched = self._set_requires_grad_by_pattern(named_params=named_params, pattern=pattern, value=True)
                if strict_match and not matched:
                    raise RuntimeError(
                        f"Staged training pattern '{pattern}' matched no parameters in phase '{phase.name}'."
                    )

        for pattern in list(phase.freeze_param_patterns or []):
            matched = self._set_requires_grad_by_pattern(named_params=named_params, pattern=pattern, value=False)
            if strict_match and not matched:
                raise RuntimeError(
                    f"Staged training pattern '{pattern}' matched no parameters in phase '{phase.name}'."
                )

        for pattern in list(phase.unfreeze_param_patterns or []):
            matched = self._set_requires_grad_by_pattern(named_params=named_params, pattern=pattern, value=True)
            if strict_match and not matched:
                raise RuntimeError(
                    f"Staged training pattern '{pattern}' matched no parameters in phase '{phase.name}'."
                )

        trainable_parameters = sum(param.numel() for _, param in named_params if bool(param.requires_grad))
        frozen_parameters = sum(param.numel() for _, param in named_params if not bool(param.requires_grad))
        LOGGER.info(
            "[purity_staged_training] phase=%s freeze_all=%s trainable_patterns=%s freeze_patterns=%s "
            "unfreeze_patterns=%s trainable_parameters=%s frozen_parameters=%s",
            phase.name,
            bool(phase.freeze_all),
            len(list(phase.trainable_param_patterns or [])),
            len(list(phase.freeze_param_patterns or [])),
            len(list(phase.unfreeze_param_patterns or [])),
            trainable_parameters,
            frozen_parameters,
        )
        return {
            "trainable_parameters": int(trainable_parameters),
            "frozen_parameters": int(frozen_parameters),
        }

    @staticmethod
    def _set_requires_grad_by_pattern(*, named_params: list[tuple[str, Any]], pattern: str, value: bool) -> bool:
        regex = re.compile(pattern)
        matched = False
        for name, param in named_params:
            candidate_names = [name]
            if name.startswith("model."):
                candidate_names.append(name[len("model.") :])
            if any(regex.search(candidate) for candidate in candidate_names):
                param.requires_grad = bool(value)
                matched = True
        return matched

    @staticmethod
    def _apply_phase_task_weights(*, module, task_weights: Mapping[str, float] | None) -> None:
        setter = getattr(module, "set_task_weights", None)
        if callable(setter):
            setter(task_weights)
            return
        if task_weights:
            LOGGER.warning(
                "Module %s does not expose set_task_weights(); staged task_weights were ignored.",
                type(module).__name__,
            )

    def _build_phase_trainer(self, *, phase: PurityPhase):
        cfg = self.runtime_state.get("resolved_training_config")
        if not isinstance(cfg, Mapping):
            cfg = self.config_json
        trainer_block = dict(dict(cfg).get("trainer") or {})
        trainer_name = str(trainer_block.get("type") or "").strip()
        if trainer_name == "":
            raise RuntimeError("training.train.trainer.type must be set for staged training.")
        trainer_cfg = dict(trainer_block.get("config") or {})
        trainer_kwargs = dict(trainer_cfg.get("trainer_kwargs") or {})

        max_epochs = phase.max_epochs
        if max_epochs is not None:
            trainer_kwargs["max_epochs"] = int(max_epochs)
        phase_trainer_kwargs = phase.trainer_kwargs
        if isinstance(phase_trainer_kwargs, Mapping):
            trainer_kwargs = merge_nested_dicts(base=trainer_kwargs, override=phase_trainer_kwargs)

        early_stopping = dict(trainer_cfg.get("early_stopping") or {})
        phase_early_stopping = phase.early_stopping
        if isinstance(phase_early_stopping, Mapping):
            early_stopping = merge_nested_dicts(base=early_stopping, override=phase_early_stopping)

        return TrainerFactory(trainer_name=trainer_name).build(
            config={"trainer_kwargs": trainer_kwargs, "early_stopping_cfg": early_stopping}
        )

    def _resolved_base_max_epochs(self) -> int:
        cfg = self.runtime_state.get("resolved_training_config")
        if not isinstance(cfg, Mapping):
            cfg = self.config_json
        trainer_cfg = dict(dict(cfg).get("trainer") or {}).get("config")
        if isinstance(trainer_cfg, Mapping):
            trainer_kwargs = dict(trainer_cfg.get("trainer_kwargs") or {})
            if trainer_kwargs.get("max_epochs") is not None:
                return int(trainer_kwargs["max_epochs"])
            if trainer_cfg.get("max_epochs") is not None:
                return int(trainer_cfg["max_epochs"])
        return 1
