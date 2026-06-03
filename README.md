# pioneerML PURITY plugin

Plugin package that integrates the PURITY-style ATAR+LYSO event model into the pioneerML config-driven training/inference framework.

What this plugin registers:
- `architecture`: `purity`
- `loader`: `purity`
- `writer`: `purity`
- `loss`: `purity_unified` (alias: `purity_multitask`)

## Requirements

This is a plugin **to** the `pioneerML` framework — you need the framework checked out alongside it.
- Python 3.10+
- The framework's ML stack: `torch` (2.x, CUDA build), `torch_geometric`, `pytorch-lightning`, `zenml`, `pyarrow`, `numpy`. Match the versions the framework pins.

## Install

The framework loads plugins via their `plugin.json` manifest (it is **not** pip-installed). Place this repo where the framework scans for plugins and put both `src/` trees on the path:

```
<pioneerML>/plugins/PURITY/      # this repo
```
```bash
export PYTHONPATH="<pioneerML>/src:<pioneerML>/plugins/PURITY/src:$PYTHONPATH"
```

On load the framework reads `plugins/PURITY/plugin.json` (module `pioneerml_purity_plugin`) and registers the `purity` architecture / loader / writer and the `purity_unified` loss.

## Quickstart

Config-driven, staged training:
```python
from pioneerml_purity_plugin.purity.pipeline import load_config, training_pipeline
cfg = load_config()["training"]
# point cfg[...]["loader_manager"]["config"]["input_sources_spec"]["main_sources"]
# at your ML parquet shards, then:
training_pipeline.with_options(enable_cache=False)(pipeline_config=cfg)
```
Training exports a checkpoint under the configured `export` prefix (`*_state_dict.pt`); load it through the adapter for inference. See the notebooks below for full training / inference / validation walk-throughs, or the smoke scripts for a dummy-data end-to-end check.

Main package:
- `plugins/PURITY/src/pioneerml_purity_plugin`

Manifest:
- `plugins/PURITY/plugin.json` declares plugin metadata for framework discovery/load order.

Pipeline config entrypoint:
- `pioneerml_purity_plugin.purity.pipeline.load_config()`

Model code layout:
- `pioneerml_purity_plugin/purity/model/purity_model_adapter.py` (framework adapter layer)
- `pioneerml_purity_plugin/purity/model/purity_hybrid_model.py` (core model port from `unified_reco/models.py`)

Loss code layout:
- `pioneerml_purity_plugin/purity/losses/purity_unified_loss.py` (registered  multitask loss)
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
