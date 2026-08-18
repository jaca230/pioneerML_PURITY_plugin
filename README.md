# pioneerML PURITY Plugin

Current PURITY plugin for the ATAR+LYSO event model.

This plugin contains:

- PURITY model architecture and Lightning module
- PURITY parquet loader and writer
- training, inference, evaluation, and export pipeline pieces
- PURITY notebooks and default pipeline config

## Setup

Place this repository under a pioneerML checkout:

```text
plugins/PURITY/
```

Then include it on `PYTHONPATH` with the framework:

```bash
export PYTHONPATH="<pioneerML>/src:<pioneerML>/plugins/PURITY/src:$PYTHONPATH"
```

The plugin manifest is `plugin.json`; the Python module is
`pioneerml_purity_plugin`.
