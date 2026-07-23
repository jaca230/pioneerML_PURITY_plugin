# Native PURITY inference loader

The native loader accelerates the CPU-side preparation of PURITY inference
batches. It replaces only the PURITY graph-building stage; Parquet input,
generic loader stages, PyTorch conversion, model execution, and prediction
writing continue to use the normal pioneerML pipeline.

For each Parquet chunk, the C++ implementation:

- normalizes and interleaves ragged ATAR and LYSO hits in event order;
- constructs the ten node features consumed by the PURITY model;
- groups nodes into time slices;
- constructs node-to-slice and slice-to-graph mappings;
- constructs graph, node, and slice pointer/count arrays.

The extension returns owning NumPy arrays through pybind11. The existing Python
`BatchPackStage` converts these arrays with `torch.from_numpy()`, which is
zero-copy for the contiguous CPU arrays produced here. LibTorch and CUDA are
therefore intentionally absent from the native extension. This keeps it
independent of the installed PyTorch and CUDA versions.

## Selecting the loader

Set the PURITY loader option in an inference configuration:

```json
{
  "graph_builder_backend": "cpp"
}
```

The available backends are:

- `standard`: production Python graph builder and the default;
- `python_optimized`: inference-only vectorized NumPy graph builder;
- `cpp`: inference-only native graph builder.

The optimized backends reject training mode because they intentionally omit
training-target construction.

## Building

From the top-level PIONEER checkout:

```bash
source docker/setenv.sh
./setup.sh -m
```

The setup action validates the active Python environment and invokes
`build_native.py`. Developers can also run that script directly.

Build requirements:

- a C++20 compiler;
- Python development headers matching the active interpreter;
- `setuptools`, `wheel`, and `pybind11`;
- NumPy at runtime.

The PIONEER ML extension image supplies these through its base image and
`ml/requirements.txt`. The generated extension and intermediate build directory
are ignored by Git and should be rebuilt for each target Python environment.

## Expected performance

On a freshly simulated 10,000-event PURITY sample, batch size 128, the complete
loader took approximately:

| Backend | Complete loader time | Relative to standard |
|---|---:|---:|
| `standard` | 3.65 s | 1.0x |
| `python_optimized` | 0.43 s | 8.4x |
| `cpp` | 0.11 s | 33x |

These are complete-loader measurements, not isolated C++ kernel timings.
Absolute performance and speed-up depend on event complexity, storage, CPU,
batch size, and chunk layout. After native graph construction, Parquet reading,
batch slicing, and Python DataLoader orchestration become the largest remaining
loader costs.

An end-to-end smoke test confirmed identical event ordering, graph tensors,
slice metadata, model predictions, and serialized prediction columns for all
three backends.
