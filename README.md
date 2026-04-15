# pioneerML PURITY plugin

Plugin package that integrates the PURITY-style ATAR+LYSO event model into the pioneerML config-driven training/inference framework.

What this plugin registers:
- `architecture`: `purity`
- `loader`: `purity`
- `writer`: `purity`
- `loss`: `purity_event_bce`, `purity_unified` (alias: `purity_multitask`)

Main package:
- `plugins/PURITY/src/pioneerml_purity_plugin`

Manifest:
- `plugins/PURITY/plugin.json` declares plugin metadata for framework discovery/load order.

Pipeline config entrypoint:
- `pioneerml_purity_plugin.purity.pipeline.load_config()`

Model code layout:
- `pioneerml_purity_plugin/purity/model/purity_model_adapter.py` (framework adapter layer)
- `pioneerml_purity_plugin/purity/model/purity_hybrid_model.py` (core model port from `unified_reco/models.py`)
- `pioneerml_purity_plugin/purity/model/purity.py` (backward-compatible import shim)

Loss code layout:
- `pioneerml_purity_plugin/purity/losses/purity_unified_loss.py` (registered Omar-parity multitask loss)
- `pioneerml_purity_plugin/purity/losses/purity_event_bce_loss.py` (simple event-only BCE alternative)
- `pioneerml_purity_plugin/purity/losses/utils/unified_training_components.py` (internal loss components ported from `unified_reco/train_utils.py`)

Plugin notebooks:
- `pioneerml_purity_plugin/purity/notebooks/training.ipynb`
- `pioneerml_purity_plugin/purity/notebooks/inference.ipynb`
- `pioneerml_purity_plugin/purity/notebooks/validation.ipynb`

Expected input parquet columns for loader `purity`:
- Required features: `event_id`, `atar_x`, `atar_y`, `atar_z`, `atar_E`, `atar_t`, `atar_view`
- Slice id: `atar_slice` or `atar_slice_id`
- Optional LYSO: `lyso_x`, `lyso_y`, `lyso_z`, `lyso_E`, `lyso_t`, `lyso_slice`
- Required train/eval target: `truth_positron_energy`
- Optional train/eval target: `truth_is_signal`

Smoke scripts in main repo:
- `python artifacts/generate_purity_dummy_parquet.py`
- `python artifacts/run_purity_small.py`
- `python artifacts/run_purity_inference_small.py`

Staged training support:
- `pioneerml_purity_plugin.purity.pipeline.training_pipeline` now uses a PURITY-specific training step that can run multiple `trainer.fit(...)` phases from `training.train.staged_training`.
- Stage control is config-driven (`enabled`, `phases[*].task_weights`, per-phase epochs, optional parameter freeze/unfreeze regex patterns).
- HPO currently stays on the shared flow (`training.hpo`) and is not phase-aware yet.
