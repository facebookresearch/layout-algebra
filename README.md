<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/logo.svg" alt="tensor-layouts" width="520">
</p>

[![CI](https://github.com/facebookresearch/tensor-layouts/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/tensor-layouts/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tensor-layouts)](https://pypi.org/project/tensor-layouts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/facebookresearch/tensor-layouts/blob/main/LICENSE)

A pure-Python implementation of the [NVIDIA CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) layout algebra. **No GPU required.**

CuTe layouts describe how logical coordinates map to memory offsets on GPUs.
This library lets you construct, compose, and visualize those layouts using
plain Python — useful for understanding tensor core access patterns, debugging
swizzled shared memory, and prototyping tiled GPU kernels without compiling any CUDA.
The code in src/layouts.py is intended to be readable and helpful to learn and
understand layout algebra.
The visualization layer is also designed to be pedagogical: for example,
hierarchical layout views can explicitly show nested row/column coordinates and
the resulting offset for each displayed cell.

## Installation

```bash
pip install tensor-layouts
```

For visualization support:

```bash
pip install tensor-layouts[viz]
```

## Quick Start

```python
from tensor_layouts import Layout, compose, complement, logical_divide

# A 4x8 column-major layout: offset(i,j) = i + j*4
layout = Layout((4, 8), (1, 4))
print(layout)       # (4, 8) : (1, 4)
print(layout(2, 3)) # 14

# Compose two layouts
a = Layout((4, 2), (1, 4))
b = Layout((2, 4), (4, 1))
print(compose(a, b))

# Tile a layout into 2x4 blocks
tiler = Layout((2, 4))
print(logical_divide(layout, tiler))
```

## Core Concepts

A `Layout` is a function from logical coordinates to memory offsets, defined by
`(shape, stride)`:

| Layout | Description |
|--------|-------------|
| `Layout((4, 8), (8, 1))` | 4x8 row-major |
| `Layout((4, 8), (1, 4))` | 4x8 column-major |
| `Layout(((2,4), 8), ((1,16), 2))` | Hierarchical (tiled) |

The algebra provides four key operations:

- **`compose(A, B)`** — Function composition: apply B's indexing to A's codomain
- **`complement(L)`** — The "missing half" of a layout's codomain
- **`logical_divide(L, T)`** — Factor a layout into tiles of shape T
- **`logical_product(A, B)`** — Replicate A's pattern across B's domain

Plus `Swizzle(B, M, S)` for XOR-based bank conflict avoidance patterns.

## MMA Atoms

The library includes tensor core atom definitions for NVIDIA and AMD architectures.

### NVIDIA Atoms

```python
from tensor_layouts.atoms_nv import *

atom = SM90_64x64x16_F16F16F16_SS
print(atom.name)        # SM90_64x64x16_F16F16F16_SS
print(atom.shape_mnk)   # (64, 64, 16)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported architectures: SM70 (Volta), SM75 (Turing), SM80 (Ampere),
SM89 (Ada Lovelace), SM90 (Hopper GMMA), SM100 (Blackwell UMMA),
SM120 (Blackwell B200).

### AMD Atoms

```python
from tensor_layouts.atoms_amd import *

atom = CDNA3_32x32x16_F32F8F8_MFMA
print(atom.name)        # CDNA3_32x32x16_F32F8F8_MFMA
print(atom.shape_mnk)   # (32, 32, 16)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported architectures: CDNA1 (gfx908 / MI100), CDNA2 (gfx90a / MI200),
CDNA3 (gfx942 / MI300), CDNA3+ (gfx950).

## Visualization

With `pip install tensor-layouts[viz]`:

```python
from tensor_layouts import Layout, Swizzle
from tensor_layouts.viz import draw_layout, draw_swizzle

draw_layout(Layout((8, 8), (8, 1)), title="Row-Major 8x8", colorize=True)
draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), colorize=True)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/row_major_8x8.png" alt="Row-Major 8x8 layout" width="400">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/swizzle_8x8.png" alt="Swizzle(3, 0, 3) applied to row-major 8x8" width="800">
</p>

See [`examples/viz.ipynb`](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.ipynb) for a full
gallery of layout, swizzle, MMA atom, and tiled MMA visualizations.

## Documentation

- Example scripts assume `tensor-layouts` is installed.
  From a repo checkout, run `pip install -e .` first, or `pip install -e ".[viz]"`
  for visualization examples.
- [Layout Algebra API](https://github.com/facebookresearch/tensor-layouts/blob/main/docs/layout_api.md) — construction, querying, compose, complement, divide, product
- [Visualization API](https://github.com/facebookresearch/tensor-layouts/blob/main/docs/viz_api.md) — draw_layout, draw_swizzle, draw_mma_layout, and more
- [Layout Examples](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/layouts.py) — runnable script covering the full algebra (`python3 examples/layouts.py`)
- [Visualization Examples](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.py) — runnable script generating all visualization types (`python3 examples/viz.py`)
- [Visualization Notebook](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.ipynb) — Jupyter gallery

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

Oracle tests cross-validate against vendor reference implementations and are
skipped automatically if the corresponding tool is unavailable:

```bash
# NVIDIA (cross-validation against pycute)
pip install -e ".[test,oracle-nv]"
pytest tests/oracle_nv.py

# AMD (cross-validation against amd_matrix_instruction_calculator)
pip install -e ".[test,oracle-amd]"
pytest tests/oracle_amd.py
```

## References

- [CuTe Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [MMA Atom Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0t_mma_atom.md)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [AMD Matrix Cores Lab Notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)

## License

MIT License. See [LICENSE](https://github.com/facebookresearch/tensor-layouts/blob/main/LICENSE) for details.
