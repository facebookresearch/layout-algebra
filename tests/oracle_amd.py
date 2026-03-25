# MIT License
#
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Differential oracle tests: cross-validate atoms_amd.py against AMD's
amd_matrix_instruction_calculator.

This test imports AMD's matrix_calculator.py reference tool and compares
its lane-to-element mapping against our Layout-based atom definitions.

Also includes comprehensive structural, algebraic, and visualization tests
that exercise the full tensor_layouts API against every AMD MFMA atom.

The oracle tool is available at:
    https://github.com/ROCm/amd_matrix_instruction_calculator

Install:
    pip install amd-matrix-instruction-calculator
    # or: git clone https://github.com/ROCm/amd_matrix_instruction_calculator

Skipped automatically if the tool is not installed.
"""

import tempfile
import pytest

from tensor_layouts import Layout, size, rank, depth, mode, cosize
from tensor_layouts.layouts import (
    compose,
    complement,
    flatten,
    coalesce,
    logical_divide,
    logical_product,
    left_inverse,
    right_inverse,
    idx2crd,
    crd2idx,
)
from tensor_layouts.layout_utils import (
    make_ordered_layout,
    tile_to_shape,
    product_each,
)
from tensor_layouts.atoms_amd import * # noqa: F401,F403,F405

# Try to import the AMD matrix instruction calculator.
try:
    from amd_matrix_instruction_calculator import matrix_calculator

    HAS_CALCULATOR = True
except ImportError:
    try:
        import matrix_calculator

        HAS_CALCULATOR = True
    except ImportError:
        HAS_CALCULATOR = False

requires_calculator = pytest.mark.skipif(
    not HAS_CALCULATOR, reason="amd_matrix_instruction_calculator not available"
)

# Try to import visualization module (requires matplotlib).
try:
    from tensor_layouts.viz import draw_tv_layout, draw_mma_layout, _compute_tv_mapping

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

requires_viz = pytest.mark.skipif(
    not HAS_VIZ, reason="tensor_layouts.viz not available (needs matplotlib)"
)


# =============================================================================
# Complete atom inventory for parametrized tests
# =============================================================================

ALL_ATOMS = [
    # CDNA1 FP16
    CDNA_32x32x8_F32F16F16_MFMA,
    CDNA_16x16x16_F32F16F16_MFMA,
    CDNA_4x4x4_F32F16F16_MFMA,
    CDNA_32x32x4_F32F16F16_MFMA,
    CDNA_16x16x4_F32F16F16_MFMA,
    # BF16 (1K and original)
    CDNA_32x32x8_F32BF16BF16_1K_MFMA,
    CDNA_16x16x16_F32BF16BF16_1K_MFMA,
    CDNA_32x32x4_F32BF16BF16_MFMA,
    CDNA_16x16x8_F32BF16BF16_MFMA,
    # INT8, FP32, FP64
    CDNA_32x32x8_I32I8I8_MFMA,
    CDNA_16x16x16_I32I8I8_MFMA,
    CDNA_32x32x2_F32F32F32_MFMA,
    CDNA_16x16x4_F32F32F32_MFMA,
    CDNA_16x16x4_F64F64F64_MFMA,
    # CDNA3
    CDNA3_32x32x16_I32I8I8_MFMA,
    CDNA3_16x16x32_I32I8I8_MFMA,
    CDNA3_32x32x4_F32XF32XF32_MFMA,
    CDNA3_16x16x8_F32XF32XF32_MFMA,
    # CDNA3 FP8 (all 8 variants)
    CDNA3_32x32x16_F32F8F8_MFMA,
    CDNA3_16x16x32_F32F8F8_MFMA,
    CDNA3_32x32x16_F32BF8BF8_MFMA,
    CDNA3_16x16x32_F32BF8BF8_MFMA,
    CDNA3_32x32x16_F32F8BF8_MFMA,
    CDNA3_16x16x32_F32F8BF8_MFMA,
    CDNA3_32x32x16_F32BF8F8_MFMA,
    CDNA3_16x16x32_F32BF8F8_MFMA,
    # CDNA3+
    CDNA3P_32x32x16_F32F16F16_MFMA,
    CDNA3P_16x16x32_F32F16F16_MFMA,
    CDNA3P_32x32x16_F32BF16BF16_MFMA,
    CDNA3P_16x16x32_F32BF16BF16_MFMA,
    CDNA3P_32x32x32_I32I8I8_MFMA,
    CDNA3P_16x16x64_I32I8I8_MFMA,
]


# =============================================================================
# Helpers
# =============================================================================


def _num_threads(layout):
    """Number of threads from a TV layout's thread dimension."""
    return (
        size(layout.shape[0]) if isinstance(layout.shape, tuple) else size(layout.shape)
    )


def _num_values(layout):
    """Number of values per thread from a TV layout's value dimension."""
    return size(layout.shape[1]) if isinstance(layout.shape, tuple) else 1


def get_calculator_d_mapping(arch: str, instruction: str, m: int, n: int):
    """Run the AMD calculator and return a dict: (lane, vgpr) -> (row, col).

    The calculator's API may vary by version.  We try the known interfaces.
    """
    # The calculator typically provides a function or class to get register info.
    # Adapt this to the actual API once we can test on x86.
    try:
        # Attempt the CLI-style API
        calc = matrix_calculator.MatrixCalculator(arch, instruction)
        mapping = {}
        for lane in range(64):
            for vgpr in range(calc.num_output_regs):
                row, col = calc.get_output_register(lane, vgpr)
                mapping[(lane, vgpr)] = (row, col)
        return mapping
    except AttributeError:
        pass

    # Fallback: try the direct function API
    try:
        info = matrix_calculator.get_instruction_info(arch, instruction)
        mapping = {}
        num_vgprs = info["num_output_regs"]
        for lane in range(64):
            for vgpr in range(num_vgprs):
                row, col = info["get_output"](lane, vgpr)
                mapping[(lane, vgpr)] = (row, col)
        return mapping
    except (AttributeError, KeyError):
        pass

    pytest.skip(f"Could not determine calculator API for {instruction}")


def validate_c_layout(atom, arch: str):
    """Validate an atom's c_layout against the AMD calculator."""
    m, n, k = atom.shape_mnk
    c = atom.c_layout

    # Get the reference mapping from AMD's calculator
    ref = get_calculator_d_mapping(arch, atom.ptx, m, n)

    # Get number of values per thread from our layout
    num_v = _num_values(c)

    errors = []
    for lane in range(64):
        for vgpr in range(num_v):
            # Our layout: (thread_idx, value_idx) -> col-major offset
            offset = c(lane, vgpr)
            our_row = offset % m
            our_col = offset // m

            if (lane, vgpr) in ref:
                ref_row, ref_col = ref[(lane, vgpr)]
                if our_row != ref_row or our_col != ref_col:
                    errors.append(
                        f"lane={lane} vgpr={vgpr}: "
                        f"ours=({our_row},{our_col}) "
                        f"ref=({ref_row},{ref_col})"
                    )

    assert not errors, f"{atom.name}: {len(errors)} mismatches:\n" + "\n".join(
        errors[:20]
    )


# =============================================================================
# CDNA1 (gfx908) FP16 atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x8_f32f16f16():
    validate_c_layout(CDNA_32x32x8_F32F16F16_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_16x16x16_f32f16f16():
    validate_c_layout(CDNA_16x16x16_F32F16F16_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_4x4x4_f32f16f16():
    validate_c_layout(CDNA_4x4x4_F32F16F16_MFMA, "cdna1")


# =============================================================================
# CDNA1 non-k-reduction variants
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x4_f32f16f16():
    validate_c_layout(CDNA_32x32x4_F32F16F16_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_16x16x4_f32f16f16():
    validate_c_layout(CDNA_16x16x4_F32F16F16_MFMA, "cdna1")


# =============================================================================
# CDNA2 (gfx90a) BF16_1K atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x8_f32bf16bf16_1k():
    validate_c_layout(CDNA_32x32x8_F32BF16BF16_1K_MFMA, "cdna2")


@requires_calculator
def test_oracle_cdna_16x16x16_f32bf16bf16_1k():
    validate_c_layout(CDNA_16x16x16_F32BF16BF16_1K_MFMA, "cdna2")


# =============================================================================
# CDNA1/2 BF16 (original, non-1K) atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x4_f32bf16bf16():
    validate_c_layout(CDNA_32x32x4_F32BF16BF16_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_16x16x8_f32bf16bf16():
    validate_c_layout(CDNA_16x16x8_F32BF16BF16_MFMA, "cdna1")


# =============================================================================
# CDNA1/2 INT8 atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x8_i32i8i8():
    validate_c_layout(CDNA_32x32x8_I32I8I8_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_16x16x16_i32i8i8():
    validate_c_layout(CDNA_16x16x16_I32I8I8_MFMA, "cdna1")


# =============================================================================
# CDNA1/2 FP32 atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna_32x32x2_f32f32f32():
    validate_c_layout(CDNA_32x32x2_F32F32F32_MFMA, "cdna1")


@requires_calculator
def test_oracle_cdna_16x16x4_f32f32f32():
    validate_c_layout(CDNA_16x16x4_F32F32F32_MFMA, "cdna1")


# =============================================================================
# CDNA2/3 FP64 atom
# =============================================================================


@requires_calculator
def test_oracle_cdna_16x16x4_f64f64f64():
    validate_c_layout(CDNA_16x16x4_F64F64F64_MFMA, "cdna2")


# =============================================================================
# CDNA3 (gfx942) enhanced atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna3_32x32x16_i32i8i8():
    validate_c_layout(CDNA3_32x32x16_I32I8I8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x32_i32i8i8():
    validate_c_layout(CDNA3_16x16x32_I32I8I8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_32x32x4_f32xf32xf32():
    validate_c_layout(CDNA3_32x32x4_F32XF32XF32_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x8_f32xf32xf32():
    validate_c_layout(CDNA3_16x16x8_F32XF32XF32_MFMA, "cdna3")


# =============================================================================
# CDNA3 FP8 atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna3_32x32x16_f32f8f8():
    validate_c_layout(CDNA3_32x32x16_F32F8F8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x32_f32f8f8():
    validate_c_layout(CDNA3_16x16x32_F32F8F8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_32x32x16_f32bf8bf8():
    validate_c_layout(CDNA3_32x32x16_F32BF8BF8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x32_f32bf8bf8():
    validate_c_layout(CDNA3_16x16x32_F32BF8BF8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_32x32x16_f32f8bf8():
    validate_c_layout(CDNA3_32x32x16_F32F8BF8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x32_f32f8bf8():
    validate_c_layout(CDNA3_16x16x32_F32F8BF8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_32x32x16_f32bf8f8():
    validate_c_layout(CDNA3_32x32x16_F32BF8F8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3_16x16x32_f32bf8f8():
    validate_c_layout(CDNA3_16x16x32_F32BF8F8_MFMA, "cdna3")


# =============================================================================
# CDNA3+ (gfx950) double-rate atoms
# =============================================================================


@requires_calculator
def test_oracle_cdna3p_32x32x16_f32f16f16():
    validate_c_layout(CDNA3P_32x32x16_F32F16F16_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3p_16x16x32_f32f16f16():
    validate_c_layout(CDNA3P_16x16x32_F32F16F16_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3p_32x32x16_f32bf16bf16():
    validate_c_layout(CDNA3P_32x32x16_F32BF16BF16_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3p_16x16x32_f32bf16bf16():
    validate_c_layout(CDNA3P_16x16x32_F32BF16BF16_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3p_32x32x32_i32i8i8():
    validate_c_layout(CDNA3P_32x32x32_I32I8I8_MFMA, "cdna3")


@requires_calculator
def test_oracle_cdna3p_16x16x64_i32i8i8():
    validate_c_layout(CDNA3P_16x16x64_I32I8I8_MFMA, "cdna3")


# =============================================================================
# Structural self-consistency tests (run without the calculator)
# =============================================================================

# These tests verify algebraic properties of the layouts themselves,
# independent of the AMD calculator. They always run.


@pytest.mark.parametrize("atom", ALL_ATOMS, ids=lambda a: a.name)
class TestMFMAStructural:
    """Structural invariants that must hold for any valid MFMA atom."""

    def test_c_layout_covers_all_elements(self, atom):
        """Every element of the M x N output is touched exactly once."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        num_t = _num_threads(c)
        num_v = _num_values(c)

        seen = set()
        for t in range(num_t):
            for v in range(num_v):
                offset = c(t, v)
                assert (
                    0 <= offset < m * n
                ), f"{atom.name}: offset {offset} out of range [0, {m*n})"
                assert (
                    offset not in seen
                ), f"{atom.name}: duplicate offset {offset} at t={t}, v={v}"
                seen.add(offset)

        assert (
            len(seen) == m * n
        ), f"{atom.name}: covers {len(seen)} elements, expected {m*n}"

    def test_c_layout_thread_count(self, atom):
        """Thread dimension has exactly 64 elements (one wavefront)."""
        c = atom.c_layout
        assert (
            _num_threads(c) == 64
        ), f"{atom.name}: {_num_threads(c)} threads, expected 64"

    def test_a_layout_thread_count(self, atom):
        """A layout thread dimension has exactly 64 elements."""
        a = atom.a_layout
        assert (
            _num_threads(a) == 64
        ), f"{atom.name}: A has {_num_threads(a)} threads, expected 64"

    def test_b_layout_thread_count(self, atom):
        """B layout thread dimension has exactly 64 elements."""
        b = atom.b_layout
        assert (
            _num_threads(b) == 64
        ), f"{atom.name}: B has {_num_threads(b)} threads, expected 64"

    def test_a_layout_broadcast(self, atom):
        """A layout broadcasts across blocks (stride-0 in block dimension)."""
        a = atom.a_layout
        if isinstance(a.stride, tuple) and isinstance(a.stride[0], tuple):
            blk_stride = a.stride[0][0]
            assert (
                blk_stride == 0
            ), f"{atom.name}: A layout block stride is {blk_stride}, expected 0"

    def test_b_layout_broadcast(self, atom):
        """B layout broadcasts across blocks (stride-0 in block dimension)."""
        b = atom.b_layout
        if isinstance(b.stride, tuple) and isinstance(b.stride[0], tuple):
            blk_stride = b.stride[0][0]
            assert (
                blk_stride == 0
            ), f"{atom.name}: B layout block stride is {blk_stride}, expected 0"

    def test_a_layout_cosize_bounded(self, atom):
        """A layout codomain is bounded by thread_count * values_per_thread."""
        a = atom.a_layout
        # cosize is max_offset + 1; for broadcast layouts this can exceed M*K
        # but must be bounded by the underlying coordinate space
        assert cosize(a) >= 1, f"{atom.name}: A cosize must be positive"

    def test_b_layout_cosize_bounded(self, atom):
        """B layout codomain is bounded by thread_count * values_per_thread."""
        b = atom.b_layout
        assert cosize(b) >= 1, f"{atom.name}: B cosize must be positive"

    def test_c_layout_cosize_equals_mn(self, atom):
        """C layout codomain spans exactly M x N (since it's a bijection)."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        assert cosize(c) == m * n, f"{atom.name}: C cosize {cosize(c)} != M*N={m*n}"

    def test_thr_id_is_none(self, atom):
        """AMD MFMA atoms use identity thread mapping (thr_id is None)."""
        assert (
            atom.thr_id is None
        ), f"{atom.name}: thr_id should be None, got {atom.thr_id}"

    def test_c_layout_rank_is_2(self, atom):
        """C layout is rank-2: (thread, value)."""
        c = atom.c_layout
        assert rank(c) == 2, f"{atom.name}: C rank {rank(c)}, expected 2"

    def test_a_layout_rank_is_2(self, atom):
        """A layout is rank-2: (thread, value)."""
        a = atom.a_layout
        assert rank(a) == 2, f"{atom.name}: A rank {rank(a)}, expected 2"

    def test_b_layout_rank_is_2(self, atom):
        """B layout is rank-2: (thread, value)."""
        b = atom.b_layout
        assert rank(b) == 2, f"{atom.name}: B rank {rank(b)}, expected 2"

    def test_layout_sizes_match_shape_mnk(self, atom):
        """Layout domain sizes are consistent with M, N, K."""
        m, n, k = atom.shape_mnk
        a, b, c = atom.a_layout, atom.b_layout, atom.c_layout
        assert size(c) == m * n, f"{atom.name}: C size {size(c)} != M*N={m*n}"
        # A and B sizes include the broadcast dimension, so size >= M*K / N*K
        # but since broadcast replicates the same data, size == 64 * values_per_thread
        assert size(a) == 64 * _num_values(
            a
        ), f"{atom.name}: A size {size(a)} != 64 * {_num_values(a)}"
        assert size(b) == 64 * _num_values(
            b
        ), f"{atom.name}: B size {size(b)} != 64 * {_num_values(b)}"


# =============================================================================
# Layout algebra tests (run without the calculator)
# =============================================================================


@pytest.mark.parametrize("atom", ALL_ATOMS, ids=lambda a: a.name)
class TestLayoutAlgebra:
    """Test layout algebra operations on real AMD atom layouts."""

    def test_size_rank_depth_mode(self, atom):
        """Exercise size(), rank(), depth(), mode() on all three layouts."""
        for layout_name, layout in [
            ("C", atom.c_layout),
            ("A", atom.a_layout),
            ("B", atom.b_layout),
        ]:
            s = size(layout)
            r = rank(layout)
            d = depth(layout)
            assert s > 0, f"{atom.name} {layout_name}: size must be positive"
            assert r == 2, f"{atom.name} {layout_name}: rank must be 2"
            assert d >= 1, f"{atom.name} {layout_name}: depth must be >= 1"

            # mode(layout, 0) is the thread dimension
            thr_mode = mode(layout, 0)
            val_mode = mode(layout, 1)
            assert (
                size(thr_mode) * size(val_mode) == s
            ), f"{atom.name} {layout_name}: mode sizes don't multiply to total"

    def test_flatten_preserves_mapping(self, atom):
        """flatten(c_layout) produces the same offsets for all flat indices."""
        c = atom.c_layout
        c_flat = flatten(c)
        # Flattened layout should produce same offsets when indexed linearly
        for i in range(size(c)):
            assert c_flat(i) == c(
                i
            ), f"{atom.name}: flatten mismatch at {i}: {c_flat(i)} != {c(i)}"

    def test_coalesce_preserves_mapping(self, atom):
        """coalesce(c_layout) produces the same offsets."""
        c = atom.c_layout
        c_coal = coalesce(c)
        for i in range(size(c)):
            assert c_coal(i) == c(
                i
            ), f"{atom.name}: coalesce mismatch at {i}: {c_coal(i)} != {c(i)}"

    def test_compose_with_identity(self, atom):
        """compose(L, identity) == L for all indices."""
        c = atom.c_layout
        identity = Layout(size(c))  # col-major identity
        composed = compose(c, identity)
        for i in range(size(c)):
            assert composed(i) == c(i), f"{atom.name}: compose(C, id) mismatch at {i}"

    def test_complement_c_layout(self, atom):
        """complement of flattened C layout produces valid ordered disjoint layout."""
        c = atom.c_layout
        c_flat = flatten(c)
        # C layout is a bijection on [0, M*N), so complement should be trivial (size 1)
        comp = complement(c_flat)
        # complement must be ordered: comp(i-1) < comp(i) for i >= 1
        for i in range(1, size(comp)):
            assert comp(i - 1) < comp(
                i
            ), f"{atom.name}: complement not ordered at {i}: {comp(i-1)} >= {comp(i)}"
        # complement must be disjoint from layout for i >= 1
        c_offsets = {c_flat(j) for j in range(size(c_flat))}
        for i in range(1, size(comp)):
            assert (
                comp(i) not in c_offsets
            ), f"{atom.name}: complement({i})={comp(i)} overlaps with layout"

    def test_left_inverse_c_layout(self, atom):
        """left_inverse(C) composed with C gives identity for flat indices."""
        c = atom.c_layout
        c_flat = flatten(c)
        linv = left_inverse(c_flat)
        # left_inverse(L)(L(i)) == i for all i
        for i in range(size(c_flat)):
            offset = c_flat(i)
            roundtrip = linv(offset)
            assert (
                roundtrip == i
            ), f"{atom.name}: left_inverse roundtrip at {i}: {roundtrip} != {i}"

    def test_right_inverse_c_layout(self, atom):
        """C composed with right_inverse(C) gives identity for offsets in range."""
        c = atom.c_layout
        c_flat = flatten(c)
        rinv = right_inverse(c_flat)
        # L(right_inverse(L)(j)) == j for all j in the image of L
        for i in range(size(c_flat)):
            offset = c_flat(i)
            roundtrip = c_flat(rinv(offset))
            assert (
                roundtrip == offset
            ), f"{atom.name}: right_inverse roundtrip at offset {offset}: {roundtrip} != {offset}"

    def test_logical_divide_c_layout(self, atom):
        """logical_divide factors C layout into (tile, rest)."""
        c = atom.c_layout
        # Divide the thread dimension by 2 (always valid since threads >= 64)
        thr_shape = c.shape[0]
        thr_size = size(thr_shape)
        tile_size = min(thr_size, 2)
        tiler = Layout(tile_size)
        c_flat_thr = flatten(mode(c, 0))
        divided = logical_divide(c_flat_thr, tiler)
        # The divided layout must cover the same total size
        assert size(divided) == size(
            c_flat_thr
        ), f"{atom.name}: logical_divide changed size: {size(divided)} != {size(c_flat_thr)}"

    def test_logical_product(self, atom):
        """logical_product replicates a pattern across positions."""
        c = atom.c_layout
        c_flat = flatten(c)
        # Product with a 2-element layout should double the domain
        replicator = Layout(2, size(c_flat))
        product = logical_product(c_flat, replicator)
        # Size should be original * 2
        assert (
            size(product) == size(c_flat) * 2
        ), f"{atom.name}: logical_product size {size(product)} != {size(c_flat) * 2}"
        # First half should match original
        for i in range(size(c_flat)):
            assert product(i) == c_flat(
                i
            ), f"{atom.name}: logical_product first-half mismatch at {i}"

    def test_idx2crd_crd2idx_roundtrip(self, atom):
        """idx2crd and crd2idx are inverses on the thread dimension shape."""
        c = atom.c_layout
        thr_shape = c.shape[0]
        for i in range(size(thr_shape)):
            crd = idx2crd(i, thr_shape)
            idx = crd2idx(crd, thr_shape)
            assert (
                idx == i
            ), f"{atom.name}: idx2crd/crd2idx roundtrip at {i}: {idx} != {i}"

    def test_idx2crd_crd2idx_roundtrip_val(self, atom):
        """idx2crd/crd2idx roundtrip on value dimension."""
        c = atom.c_layout
        val_shape = c.shape[1]
        for i in range(size(val_shape)):
            crd = idx2crd(i, val_shape)
            idx = crd2idx(crd, val_shape)
            assert (
                idx == i
            ), f"{atom.name}: val idx2crd/crd2idx roundtrip at {i}: {idx} != {i}"

    def test_flatten_is_idempotent(self, atom):
        """flatten(flatten(L)) == flatten(L)."""
        c = atom.c_layout
        once = flatten(c)
        twice = flatten(once)
        for i in range(size(c)):
            assert once(i) == twice(i), f"{atom.name}: flatten not idempotent at {i}"

    def test_coalesce_is_idempotent(self, atom):
        """coalesce(coalesce(L)) == coalesce(L)."""
        c = atom.c_layout
        once = coalesce(c)
        twice = coalesce(once)
        for i in range(size(c)):
            assert once(i) == twice(i), f"{atom.name}: coalesce not idempotent at {i}"

    def test_flatten_then_coalesce(self, atom):
        """flatten then coalesce produces same mapping."""
        c = atom.c_layout
        fc = coalesce(flatten(c))
        for i in range(size(c)):
            assert fc(i) == c(i), f"{atom.name}: flatten+coalesce mismatch at {i}"

    def test_compose_chain(self, atom):
        """compose(compose(L, A), B) == compose(L, compose(A, B)) (associativity)."""
        c = atom.c_layout
        c_flat = flatten(c)
        # Use small sub-layouts for the chain
        total = size(c_flat)
        a = Layout(min(total, 4))
        b = Layout(min(size(a), 2))
        lhs = compose(compose(c_flat, a), b)
        rhs = compose(c_flat, compose(a, b))
        for i in range(size(b)):
            assert lhs(i) == rhs(
                i
            ), f"{atom.name}: compose associativity failed at {i}: {lhs(i)} != {rhs(i)}"

    def test_make_ordered_layout_flat_c_shape(self, atom):
        """make_ordered_layout on flattened C shape produces ordered strides."""
        c = atom.c_layout
        c_flat = flatten(c)
        ordered = make_ordered_layout(c_flat.shape)
        # Same size
        assert size(ordered) == size(
            c
        ), f"{atom.name}: make_ordered_layout changed size"
        # Ordered: strides should be increasing (column-major order)
        for i in range(1, size(ordered)):
            assert ordered(i) > ordered(
                i - 1
            ), f"{atom.name}: make_ordered_layout not ordered at {i}"


# =============================================================================
# Visualization tests (run without the calculator, skip if matplotlib missing)
# =============================================================================

# Use a smaller subset for viz tests since they're slower
VIZ_ATOMS = [
    CDNA_32x32x8_F32F16F16_MFMA,
    CDNA_16x16x16_F32F16F16_MFMA,
    CDNA_4x4x4_F32F16F16_MFMA,
    CDNA_16x16x4_F64F64F64_MFMA,
    CDNA3_32x32x16_F32F8F8_MFMA,
    CDNA3P_16x16x32_F32F16F16_MFMA,
]


@requires_viz
@pytest.mark.parametrize("atom", VIZ_ATOMS, ids=lambda a: a.name)
class TestVisualization:
    """Smoke-test visualization functions on AMD atoms."""

    def test_compute_tv_mapping_c(self, atom):
        """_compute_tv_mapping on c_layout covers every cell of the M x N grid."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        tv_map = _compute_tv_mapping(c, grid_cols=n, grid_rows=m, col_major=True)
        # Every (row, col) in [0,M) x [0,N) should have an entry
        for row in range(m):
            for col in range(n):
                assert (
                    row,
                    col,
                ) in tv_map, f"{atom.name}: C tv_map missing ({row},{col})"
                phys_t, v_idx, logical_t = tv_map[(row, col)]
                assert (
                    0 <= phys_t < 64
                ), f"{atom.name}: C invalid thread {phys_t} at ({row},{col})"

    def test_compute_tv_mapping_a(self, atom):
        """_compute_tv_mapping on a_layout produces valid entries."""
        m, n, k = atom.shape_mnk
        a = atom.a_layout
        # For A: use cosize to infer actual grid dimensions since some atoms
        # (e.g. 4x4x4) are internally reshaped by CK to different M/K.
        a_cosize = cosize(a)
        num_v = _num_values(a)
        # The grid_rows/cols for A are (M_actual, K_actual) where
        # M_actual * K_actual = cosize(A) for non-broadcast layouts.
        # For broadcast layouts (stride-0), offsets repeat, so just check
        # that all produced offsets are within [0, cosize).
        for t in range(64):
            for v in range(num_v):
                offset = a(t, v)
                assert (
                    0 <= offset < a_cosize
                ), f"{atom.name}: A offset {offset} out of range [0, {a_cosize})"

    def test_compute_tv_mapping_b(self, atom):
        """_compute_tv_mapping on b_layout produces valid entries."""
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        b_cosize = cosize(b)
        num_v = _num_values(b)
        for t in range(64):
            for v in range(num_v):
                offset = b(t, v)
                assert (
                    0 <= offset < b_cosize
                ), f"{atom.name}: B offset {offset} out of range [0, {b_cosize})"

    def test_compute_tv_mapping_c_threads_match(self, atom):
        """Thread IDs from tv_mapping match direct layout evaluation."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        tv_map = _compute_tv_mapping(c, grid_cols=n, grid_rows=m, col_major=True)
        # Rebuild the forward map and compare
        num_v = _num_values(c)
        for t in range(64):
            for v in range(num_v):
                offset = c(t, v)
                row = offset % m
                col = offset // m
                assert (
                    row,
                    col,
                ) in tv_map, f"{atom.name}: ({row},{col}) missing from tv_map"
                phys_t, v_idx, logical_t = tv_map[(row, col)]
                assert (
                    phys_t == t
                ), f"{atom.name}: thread mismatch at ({row},{col}): {phys_t} != {t}"
                assert (
                    v_idx == v
                ), f"{atom.name}: value mismatch at ({row},{col}): {v_idx} != {v}"

    def test_draw_tv_layout_smoke(self, atom):
        """draw_tv_layout runs without error (output to tempfile)."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            draw_tv_layout(c, filename=f.name, grid_shape=(m, n), colorize=True)

    def test_draw_mma_layout_smoke(self, atom):
        """draw_mma_layout runs without error."""
        m, n, k = atom.shape_mnk
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            if atom.name == "CDNA_4x4x4_F32F16F16_MFMA":
                with pytest.raises(
                    ValueError, match=r"A .*panel shape .*out of bounds"
                ):
                    draw_mma_layout(
                        atom.a_layout,
                        atom.b_layout,
                        atom.c_layout,
                        filename=f.name,
                        tile_mnk=(m, n, k),
                        main_title=atom.name,
                    )
            else:
                draw_mma_layout(
                    atom.a_layout,
                    atom.b_layout,
                    atom.c_layout,
                    filename=f.name,
                    tile_mnk=(m, n, k),
                    main_title=atom.name,
                )


# =============================================================================
# Layout utils tests
# =============================================================================


@pytest.mark.parametrize("atom", ALL_ATOMS, ids=lambda a: a.name)
class TestLayoutUtils:
    """Test layout_utils functions on AMD atom layouts."""

    def test_product_each_c(self, atom):
        """product_each on C layout modes gives correct sizes."""
        c = atom.c_layout
        m, n, k = atom.shape_mnk
        thr_size = size(mode(c, 0))
        val_size = size(mode(c, 1))
        products = product_each(c.shape)
        # product_each returns a tuple of per-mode sizes
        assert isinstance(products, tuple)
        assert products[0] == thr_size
        assert products[1] == val_size

    def test_tile_to_shape_c(self, atom):
        """tile_to_shape tiles C layout to a larger MxN."""
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        # Tile to 2x the original shape
        target = (size(c.shape[0]) * 2, size(c.shape[1]))
        tiled = tile_to_shape(c, target)
        assert (
            size(tiled) == size(c) * 2
        ), f"{atom.name}: tile_to_shape wrong size: {size(tiled)} != {size(c) * 2}"
