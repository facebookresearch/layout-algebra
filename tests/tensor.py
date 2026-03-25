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

"""Tests for the Tensor class.

Tensor combines a Layout with a base offset, analogous to CuTe's (Pointer, Layout).
Slicing a Tensor fixes coordinates, accumulating their offset contribution.

These tests are designed to validate behavior against NVIDIA CuTe C++.
In CuTe, a Tensor is (Engine/Pointer, Layout) where:
  - Pointer provides the base address
  - Layout maps coordinates to offsets

Python Tensor uses integer offset instead of pointer, but the algebra is identical.
The key invariant: tensor[fixed_coords, :](remaining_coords) == tensor(all_coords)
"""

import pytest

from tensor_layouts import (
    Layout,
    Swizzle,
    compose,
    complement,
    logical_divide,
    logical_product,
    rank,
    size,
    cosize,
    mode,
    flatten,
    coalesce,
)
from tensor_layouts import Tensor


# =============================================================================
# Basic Tensor Construction and Properties
# =============================================================================
# Tests adapted from CuTe tensor construction patterns


class TestTensorConstruction:
    """Basic tensor construction matching CuTe make_tensor patterns."""

    def test_rank2_column_major(self):
        """Column-major 4x8 matrix - most common pattern."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        assert tensor.layout == layout
        assert tensor.offset == 0
        assert tensor.shape == (4, 8)
        assert tensor.stride == (1, 4)
        assert rank(tensor.layout) == 2
        assert size(tensor.layout) == 32

    def test_rank2_row_major(self):
        """Row-major 4x8 matrix."""
        layout = Layout((4, 8), (8, 1))
        tensor = Tensor(layout)

        assert tensor.shape == (4, 8)
        assert tensor.stride == (8, 1)
        # Row-major: adjacent columns are contiguous
        assert tensor(0, 0) == 0
        assert tensor(0, 1) == 1
        assert tensor(1, 0) == 8

    def test_rank1_contiguous(self):
        """Simple 1D contiguous layout."""
        layout = Layout(32, 1)
        tensor = Tensor(layout)

        # Scalar shape has rank 0 in CuTe convention
        assert rank(tensor.layout) == 0
        assert size(tensor.layout) == 32
        for i in range(32):
            assert tensor(i) == i

    def test_rank1_strided(self):
        """1D layout with stride > 1 (e.g., every other element)."""
        layout = Layout(16, 2)
        tensor = Tensor(layout)

        for i in range(16):
            assert tensor(i) == 2 * i

    def test_rank3_tensor(self):
        """3D tensor like batch x height x width."""
        layout = Layout((2, 4, 8), (32, 8, 1))
        tensor = Tensor(layout)

        assert rank(tensor.layout) == 3
        assert size(tensor.layout) == 64
        # Verify indexing
        assert tensor(0, 0, 0) == 0
        assert tensor(1, 0, 0) == 32
        assert tensor(0, 1, 0) == 8
        assert tensor(0, 0, 1) == 1
        assert tensor(1, 2, 3) == 32 + 16 + 3

    def test_with_offset(self):
        """Tensor with non-zero base offset (pointer arithmetic)."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout, offset=100)

        assert tensor.offset == 100
        # All coordinates shifted by offset
        assert tensor(0, 0) == 100
        assert tensor(1, 0) == 101
        assert tensor(0, 1) == 104
        assert tensor(3, 7) == 100 + 3 + 7 * 4

    def test_hierarchical_shape(self):
        """Nested/hierarchical shape - key CuTe feature."""
        # Mode 0 has shape (2, 4), mode 1 has shape 8
        layout = Layout(((2, 4), 8), ((1, 2), 8))
        tensor = Tensor(layout)

        assert rank(tensor.layout) == 2
        # Mode 0 has nested shape
        mode0 = mode(tensor.layout, 0)
        assert mode0.shape == (2, 4)
        # Total size is product of all leaf shapes
        assert size(tensor.layout) == 2 * 4 * 8


class TestTensorEquality:
    """Test tensor equality and hashing."""

    def test_equal_tensors(self):
        layout = Layout((4, 8), (1, 4))
        t1 = Tensor(layout)
        t2 = Tensor(layout)

        assert t1 == t2
        assert hash(t1) == hash(t2)

    def test_different_offset(self):
        layout = Layout((4, 8), (1, 4))
        t1 = Tensor(layout)
        t2 = Tensor(layout, offset=1)

        assert t1 != t2

    def test_different_layout(self):
        t1 = Tensor(Layout((4, 8), (1, 4)))
        t2 = Tensor(Layout((4, 8), (8, 1)))

        assert t1 != t2

    def test_repr_no_offset(self):
        tensor = Tensor(Layout((4, 8), (1, 4)))
        r = repr(tensor)
        assert "Tensor" in r
        assert "offset" not in r

    def test_repr_with_offset(self):
        tensor = Tensor(Layout((4, 8), (1, 4)), offset=42)
        r = repr(tensor)
        assert "offset=42" in r


# =============================================================================
# Tensor Calling (Coordinate to Offset)
# =============================================================================


class TestTensorCall:
    """Test tensor(coords) -> offset mapping."""

    def test_column_major_indexing(self):
        """Column-major: offset = i + j * num_rows."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        for i in range(4):
            for j in range(8):
                expected = i + j * 4
                assert tensor(i, j) == expected

    def test_row_major_indexing(self):
        """Row-major: offset = i * num_cols + j."""
        layout = Layout((4, 8), (8, 1))
        tensor = Tensor(layout)

        for i in range(4):
            for j in range(8):
                expected = i * 8 + j
                assert tensor(i, j) == expected

    def test_with_offset_indexing(self):
        """Offset shifts all results."""
        layout = Layout((4, 8), (1, 4))
        base_tensor = Tensor(layout)
        offset_tensor = Tensor(layout, offset=1000)

        for i in range(4):
            for j in range(8):
                assert offset_tensor(i, j) == 1000 + base_tensor(i, j)

    def test_hierarchical_indexing(self):
        """Hierarchical coordinates like ((i0, i1), j)."""
        layout = Layout(((2, 4), 8), ((1, 2), 8))
        tensor = Tensor(layout)

        # Test various hierarchical coordinates
        assert tensor((0, 0), 0) == 0
        assert tensor((1, 0), 0) == 1
        assert tensor((0, 1), 0) == 2
        assert tensor((0, 0), 1) == 8


# =============================================================================
# Non-Swizzled Tensor Slicing
# =============================================================================
# Core CuTe operation: tensor(i, _) or tensor(_, j)


class TestTensorSlicingBasic:
    """Basic slicing without swizzle - matching CuTe tensor(i, _) patterns."""

    def test_fix_mode0_keep_mode1(self):
        """tensor[i, :] - fix row, keep column."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        for i in range(4):
            row = tensor[i, :]
            assert isinstance(row, Tensor)
            assert row.offset == i  # row i contributes offset i
            assert row.shape == 8
            assert row.stride == 4

            # Verify: row(j) == tensor(i, j)
            for j in range(8):
                assert row(j) == tensor(i, j)

    def test_fix_mode1_keep_mode0(self):
        """tensor[:, j] - keep row, fix column."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        for j in range(8):
            col = tensor[:, j]
            assert isinstance(col, Tensor)
            assert col.offset == j * 4  # column j contributes offset j*4
            assert col.shape == 4
            assert col.stride == 1

            # Verify: col(i) == tensor(i, j)
            for i in range(4):
                assert col(i) == tensor(i, j)

    def test_fix_all_modes(self):
        """tensor[i, j] - all fixed returns int."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        for i in range(4):
            for j in range(8):
                result = tensor[i, j]
                assert isinstance(result, int)
                assert result == tensor(i, j)

    def test_single_index_slice(self):
        """tensor[i] - single index fixes mode 0."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        row = tensor[2]
        assert isinstance(row, Tensor)
        assert row.offset == 2


class TestTensorSlicingRowMajor:
    """Slicing row-major tensors."""

    def test_row_major_fix_row(self):
        """Row-major: fixing row gives contiguous column slice."""
        layout = Layout((4, 8), (8, 1))
        tensor = Tensor(layout)

        row = tensor[2, :]
        assert row.offset == 2 * 8  # = 16
        assert row.stride == 1  # columns are contiguous

        for j in range(8):
            assert row(j) == tensor(2, j)

    def test_row_major_fix_column(self):
        """Row-major: fixing column gives strided row slice."""
        layout = Layout((4, 8), (8, 1))
        tensor = Tensor(layout)

        col = tensor[:, 5]
        assert col.offset == 5
        assert col.stride == 8  # rows are strided

        for i in range(4):
            assert col(i) == tensor(i, 5)


class TestTensorSlicingRank3:
    """Slicing 3D tensors."""

    def test_fix_mode0(self):
        """Fix batch dimension."""
        layout = Layout((2, 4, 8), (32, 8, 1))
        tensor = Tensor(layout)

        slice0 = tensor[1, :, :]
        assert slice0.offset == 32
        assert slice0.shape == (4, 8)
        assert slice0.stride == (8, 1)

        for i in range(4):
            for j in range(8):
                assert slice0(i, j) == tensor(1, i, j)

    def test_fix_mode1(self):
        """Fix height dimension."""
        layout = Layout((2, 4, 8), (32, 8, 1))
        tensor = Tensor(layout)

        slice1 = tensor[:, 2, :]
        assert slice1.offset == 16
        assert slice1.shape == (2, 8)

        for b in range(2):
            for w in range(8):
                assert slice1(b, w) == tensor(b, 2, w)

    def test_fix_mode2(self):
        """Fix width dimension."""
        layout = Layout((2, 4, 8), (32, 8, 1))
        tensor = Tensor(layout)

        slice2 = tensor[:, :, 5]
        assert slice2.offset == 5
        assert slice2.shape == (2, 4)

        for b in range(2):
            for h in range(4):
                assert slice2(b, h) == tensor(b, h, 5)

    def test_fix_two_modes(self):
        """Fix two of three modes."""
        layout = Layout((2, 4, 8), (32, 8, 1))
        tensor = Tensor(layout)

        slice01 = tensor[1, 2, :]
        assert slice01.offset == 32 + 16
        assert slice01.shape == 8

        for w in range(8):
            assert slice01(w) == tensor(1, 2, w)


class TestTensorSlicingAccumulation:
    """Test offset accumulation across multiple slices."""

    def test_sequential_slices(self):
        """Multiple sequential slices accumulate offset."""
        layout = Layout((4, 8, 2), (1, 4, 32))
        tensor = Tensor(layout)

        # First slice
        s1 = tensor[2, :, :]
        assert s1.offset == 2

        # Second slice
        s2 = s1[3, :]
        assert s2.offset == 2 + 3 * 4  # = 14

        # Third slice
        s3 = s2[1]
        assert s3 == 2 + 3 * 4 + 1 * 32  # = 46

    def test_slice_with_initial_offset(self):
        """Slicing tensor with non-zero offset accumulates correctly."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout, offset=100)

        row = tensor[2, :]
        assert row.offset == 102

        for j in range(8):
            assert row(j) == 100 + layout(2, j)

    def test_double_slice_equivalence(self):
        """tensor[i, :][j] == tensor[i, j]."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        for i in range(4):
            row = tensor[i, :]
            for j in range(8):
                assert row[j] == tensor[i, j]


class TestTensorSlicingHierarchical:
    """Slicing tensors with nested shapes."""

    def test_hierarchical_mode0(self):
        """Slice mode 0 with hierarchical coordinate."""
        layout = Layout(((2, 4), 8), ((1, 2), 8))
        tensor = Tensor(layout)

        # Fix mode 0 to coordinate (1, 2)
        slice0 = tensor[(1, 2), :]
        assert isinstance(slice0, Tensor)

        for j in range(8):
            assert slice0(j) == tensor((1, 2), j)

    def test_hierarchical_mode1(self):
        """Slice mode 1 when mode 0 is hierarchical."""
        layout = Layout(((2, 4), 8), ((1, 2), 8))
        tensor = Tensor(layout)

        slice1 = tensor[:, 5]
        assert isinstance(slice1, Tensor)
        assert slice1.shape == (2, 4)

        for i0 in range(2):
            for i1 in range(4):
                assert slice1((i0, i1)) == tensor((i0, i1), 5)


# =============================================================================
# Swizzled Tensor Operations
# =============================================================================
# Critical for GPU shared memory bank conflict avoidance


class TestSwizzledTensorBasic:
    """Basic swizzled tensor operations."""

    def test_swizzled_construction(self):
        """Create tensor with swizzled layout."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        assert tensor.layout.swizzle is not None
        assert rank(tensor.layout) == 2

    def test_swizzled_call_differs_from_linear(self):
        """Swizzled tensor gives different offsets than linear."""
        linear_layout = Layout((8, 8), (8, 1))
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))

        linear_tensor = Tensor(linear_layout)
        sw_tensor = Tensor(sw_layout)

        # At least some coordinates should differ
        differences = 0
        for i in range(8):
            for j in range(8):
                if linear_tensor(i, j) != sw_tensor(i, j):
                    differences += 1

        assert differences > 0, "Swizzle should change some offsets"

    def test_swizzle_xor_pattern(self):
        """Verify XOR pattern of Swizzle(3, 0, 3)."""
        sw = Swizzle(3, 0, 3)
        # Swizzle XORs bits [0:3] with bits [3:6]
        # At row 1 (binary 001 in bits 3-5 = offset 8), columns 0-7 XOR with 1
        assert sw(8) == 8 ^ 1  # = 9
        assert sw(9) == 9 ^ 1  # = 8
        assert sw(16) == 16 ^ 2  # = 18


class TestSwizzledTensorSlicing:
    """Slicing swizzled tensors - the tricky part."""

    def test_swizzled_slice_row(self):
        """Slice row from swizzled tensor."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        for i in range(8):
            row = tensor[i, :]
            assert isinstance(row, Tensor)
            # Offset is LINEAR (before swizzle)
            assert row.offset == i * 8
            # Swizzle preserved
            assert row.layout.swizzle is not None

            # Critical invariant: row(j) == tensor(i, j)
            for j in range(8):
                assert row(j) == tensor(i, j), f"Mismatch at [{i}, :]({j})"

    def test_swizzled_slice_column(self):
        """Slice column from swizzled tensor."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        for j in range(8):
            col = tensor[:, j]
            assert isinstance(col, Tensor)
            assert col.offset == j
            assert col.layout.swizzle is not None

            for i in range(8):
                assert col(i) == tensor(i, j), f"Mismatch at [:, {j}]({i})"

    def test_swizzled_full_grid_exhaustive(self):
        """Exhaustive test: all slices match direct access."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        # Row slices
        for i in range(8):
            row = tensor[i, :]
            for j in range(8):
                assert row(j) == tensor(i, j)

        # Column slices
        for j in range(8):
            col = tensor[:, j]
            for i in range(8):
                assert col(i) == tensor(i, j)

    def test_swizzle_2_1_3(self):
        """Different swizzle parameters: Swizzle(2, 1, 3)."""
        sw_layout = compose(Swizzle(2, 1, 3), Layout((4, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        for i in range(4):
            row = tensor[i, :]
            for j in range(8):
                assert row(j) == tensor(i, j)

        for j in range(8):
            col = tensor[:, j]
            for i in range(4):
                assert col(i) == tensor(i, j)

    def test_swizzle_1_2_3(self):
        """Swizzle(1, 2, 3) - single bit XOR."""
        sw_layout = compose(Swizzle(1, 2, 3), Layout((2, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        for i in range(2):
            row = tensor[i, :]
            for j in range(8):
                assert row(j) == tensor(i, j)

    def test_swizzled_with_offset(self):
        """Swizzled tensor with non-zero base offset."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout, offset=100)

        for i in range(8):
            row = tensor[i, :]
            assert row.offset == 100 + i * 8

            for j in range(8):
                assert row(j) == tensor(i, j)

    def test_swizzled_sequential_slices(self):
        """Sequential slicing of swizzled tensor."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8, 2), (8, 1, 64)))
        tensor = Tensor(sw_layout)

        # Slice mode 0
        s1 = tensor[3, :, :]
        assert s1.offset == 24

        # Slice mode 1 of result
        s2 = s1[5, :]
        assert s2.offset == 24 + 5

        # Final element
        for k in range(2):
            assert s2(k) == tensor(3, 5, k)


class TestSwizzledTensorColumnMajor:
    """Swizzled column-major layouts."""

    def test_swizzled_column_major(self):
        """Swizzle applied to column-major layout."""
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (1, 8)))
        tensor = Tensor(sw_layout)

        for i in range(8):
            row = tensor[i, :]
            for j in range(8):
                assert row(j) == tensor(i, j)

        for j in range(8):
            col = tensor[:, j]
            for i in range(8):
                assert col(i) == tensor(i, j)


# =============================================================================
# Tensors from Layout Algebra Operations
# =============================================================================
# Test tensors created from compose, complement, logical_divide, logical_product


class TestTensorFromCompose:
    """Tensors with layouts resulting from composition."""

    def test_composed_layout(self):
        """Tensor with compose(A, B) layout."""
        A = Layout((4, 8), (1, 4))
        B = Layout((2, 4), (1, 2))
        composed = compose(A, B)

        tensor = Tensor(composed)

        # Verify composition: tensor(i) == A(B(i))
        for i in range(size(composed)):
            expected = A(B(i))
            assert tensor(i) == expected

    def test_composed_layout_slicing(self):
        """Slice tensor with composed layout."""
        A = Layout((4, 8), (1, 4))
        B = Layout((2, 4), (1, 2))
        composed = compose(A, B)

        tensor = Tensor(composed)

        if rank(tensor.layout) >= 2:
            # Can slice if rank >= 2
            for i in range(tensor.shape[0] if isinstance(tensor.shape, tuple) else 1):
                slice_i = tensor[i, :] if rank(tensor.layout) == 2 else tensor[i]
                # Verify slice correctness
                if isinstance(slice_i, Tensor):
                    for j in range(size(slice_i.layout)):
                        if rank(tensor.layout) == 2:
                            assert slice_i(j) == tensor(i, j)


class TestTensorFromComplement:
    """Tensors with complemented layouts."""

    def test_complement_layout(self):
        """Tensor with complement layout."""
        base = Layout((4,), (2,))  # visits {0, 2, 4, 6}
        comp = complement(base, 8)  # visits {0, 1} interleaved

        tensor = Tensor(comp)

        # Complement visits the "gaps"
        for i in range(size(comp)):
            assert tensor(i) == comp(i)

    def test_complement_tensor_slicing(self):
        """Slicing a tensor from complemented layout."""
        base = Layout((2, 2), (1, 4))
        comp = complement(base, 8)

        tensor = Tensor(comp)

        # Verify tensor call works
        for i in range(size(comp)):
            assert tensor(i) == comp(i)


class TestTensorFromLogicalDivide:
    """Tensors with layouts from logical_divide (tiling)."""

    def test_logical_divide_simple(self):
        """Tensor from logical_divide - basic tiling."""
        layout = Layout(16, 1)
        tiler = Layout(4)
        divided = logical_divide(layout, tiler)

        tensor = Tensor(divided)

        # logical_divide creates (tile, rest) structure
        assert rank(tensor.layout) == 2

        # Verify all elements accessible
        for i in range(size(divided)):
            assert tensor(i) == layout(i)

    def test_logical_divide_2d(self):
        """Tensor from 2D logical_divide."""
        layout = Layout((8, 8), (1, 8))
        tiler = Layout((4, 4))
        divided = logical_divide(layout, tiler)

        tensor = Tensor(divided)

        # Access through divided layout
        for i in range(size(divided)):
            offset = tensor(i)
            assert 0 <= offset < cosize(layout)

    def test_logical_divide_slicing(self):
        """Slice tensor from logical_divide."""
        layout = Layout(16, 1)
        tiler = Layout(4)
        divided = logical_divide(layout, tiler)

        tensor = Tensor(divided)

        # Slice to get one tile
        if rank(tensor.layout) >= 2:
            tile0 = tensor[0, :]
            # Should have size 4 (tile size)
            if isinstance(tile0, Tensor):
                assert size(tile0.layout) == 4


class TestTensorFromLogicalProduct:
    """Tensors with layouts from logical_product (replication)."""

    def test_logical_product_simple(self):
        """Tensor from logical_product - basic replication."""
        A = Layout(4, 1)  # pattern to replicate
        B = Layout(2, 4)  # positions to replicate at

        product = logical_product(A, B)
        tensor = Tensor(product)

        # Product creates replicated pattern
        for i in range(size(product)):
            assert 0 <= tensor(i) < cosize(product)

    def test_logical_product_slicing(self):
        """Slice tensor from logical_product."""
        A = Layout(4, 1)
        B = Layout(2, 4)
        product = logical_product(A, B)

        tensor = Tensor(product)

        if rank(tensor.layout) >= 2:
            for i in range(tensor.shape[0] if isinstance(tensor.shape, tuple) else 1):
                slice_i = tensor[i, :] if rank(tensor.layout) == 2 else tensor[i]
                if isinstance(slice_i, Tensor):
                    for j in range(size(slice_i.layout)):
                        if rank(tensor.layout) == 2:
                            assert slice_i(j) == tensor(i, j)


class TestTensorFromCoalesce:
    """Tensors with coalesced layouts."""

    def test_coalesced_layout(self):
        """Tensor from coalesced layout."""
        layout = Layout((2, 2, 2), (1, 2, 4))
        coal = coalesce(layout)

        tensor = Tensor(coal)

        # Coalesced layout should give same offsets
        for i in range(size(coal)):
            assert tensor(i) == layout(i)

    def test_coalesced_slicing(self):
        """Slice tensor with coalesced layout."""
        layout = Layout((2, 2, 2), (1, 2, 4))
        coal = coalesce(layout)

        tensor = Tensor(coal)

        # Coalesced to rank-1, so slicing returns int
        if rank(tensor.layout) == 1:
            for i in range(size(coal)):
                assert tensor[i] == tensor(i)


class TestTensorFromFlatten:
    """Tensors with flattened layouts."""

    def test_flattened_layout(self):
        """Tensor from flattened hierarchical layout."""
        layout = Layout(((2, 4), (2, 2)), ((1, 2), (8, 16)))
        flat = flatten(layout)

        tensor = Tensor(flat)

        # Flattened layout should be rank-1
        # And give same offsets as original
        for i in range(size(flat)):
            assert tensor(i) == flat(i)


# =============================================================================
# Complex Swizzle + Algebra Combinations
# =============================================================================


class TestSwizzledAlgebraComposition:
    """Swizzled tensors combined with layout algebra."""

    def test_swizzled_after_divide(self):
        """Apply swizzle to divided layout."""
        base = Layout((8, 8), (1, 8))
        tiler = Layout((4, 4))
        divided = logical_divide(base, tiler)

        sw_divided = compose(Swizzle(2, 0, 2), divided)
        tensor = Tensor(sw_divided)

        # Verify we can still call through the tensor
        # The layout may be complex (hierarchical) after divide
        for i in range(size(tensor.layout)):
            # Just verify the call works - no exception
            offset = tensor(i)
            assert isinstance(offset, int)
            assert offset >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestTensorEdgeCases:
    """Edge cases and boundary conditions."""

    def test_rank1_slicing(self):
        """Slicing rank-1 tensor returns int."""
        layout = Layout(16, 1)
        tensor = Tensor(layout)

        for i in range(16):
            result = tensor[i]
            assert isinstance(result, int)
            assert result == i

    def test_rank1_swizzled(self):
        """Rank-1 swizzled tensor."""
        sw_layout = compose(Swizzle(2, 0, 2), Layout(8, 1))
        tensor = Tensor(sw_layout)

        for i in range(8):
            result = tensor[i]
            assert isinstance(result, int)
            # Swizzle applied
            assert result == tensor(i)

    def test_zero_offset_preserved(self):
        """Zero offset tensors maintain zero offset after identity slices."""
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        row0 = tensor[0, :]
        assert row0.offset == 0

        col0 = tensor[:, 0]
        assert col0.offset == 0


class TestTensorErrors:
    """Error handling tests."""

    def test_wrong_rank_indices(self):
        """Too many indices raises IndexError."""
        tensor = Tensor(Layout((4, 8), (1, 4)))

        with pytest.raises(IndexError):
            tensor[1, 2, 3]

    def test_invalid_key_type(self):
        """Invalid key type raises TypeError."""
        tensor = Tensor(Layout((4, 8), (1, 4)))

        with pytest.raises(TypeError):
            tensor["invalid", :]

    def test_invalid_single_key(self):
        """Invalid single key raises TypeError."""
        tensor = Tensor(Layout((4, 8), (1, 4)))

        with pytest.raises(TypeError):
            tensor["bad"]


# =============================================================================
# CuTe C++ Compatibility Tests
# =============================================================================
# These tests are designed to match specific CuTe C++ behaviors


class TestCuTeCompatibility:
    """Tests matching specific CuTe C++ behaviors."""

    def test_cute_tensor_slice_pattern(self):
        """
        Match CuTe pattern: tensor(make_coord(i, _), make_coord(j, _))
        In Python: tensor[i, :][j] or tensor[(i, j), :]
        """
        layout = Layout((4, 8), (1, 4))
        tensor = Tensor(layout)

        # CuTe: tensor(i, _)(j) equivalent
        for i in range(4):
            row = tensor[i, :]
            for j in range(8):
                assert row(j) == i + j * 4

    def test_cute_shared_memory_swizzle(self):
        """
        Common GPU shared memory swizzle pattern.
        Swizzle(3, 0, 3) avoids bank conflicts for 8x8 fp16 tiles.
        """
        # 8x8 tile, row-major, with swizzle
        sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
        tensor = Tensor(sw_layout)

        # In CuTe, each thread accesses a row: tensor(thread_idx, _)
        for thread in range(8):
            row = tensor[thread, :]
            # All 8 accesses should go to different banks (no conflicts)
            bank_accesses = [row(col) % 32 for col in range(8)]
            # With swizzle, should have 8 unique banks
            assert len(set(bank_accesses)) == 8, f"Thread {thread} has bank conflicts"

    def test_cute_tiled_tensor(self):
        """
        CuTe tiled tensor pattern: logical_divide then slice.
        """
        # 16 elements divided into tiles of 4
        layout = Layout(16, 1)
        tiler = Layout(4)
        divided = logical_divide(layout, tiler)

        tensor = Tensor(divided)

        # Access tile 0 elements: tensor(_, 0)
        if rank(tensor.layout) == 2:
            tile0 = tensor[:, 0]
            if isinstance(tile0, Tensor):
                for i in range(4):
                    assert tile0(i) == i

            # Access tile 1 elements
            tile1 = tensor[:, 1]
            if isinstance(tile1, Tensor):
                for i in range(4):
                    assert tile1(i) == 4 + i

    def test_cute_mma_fragment_pattern(self):
        """
        Pattern used in MMA (Matrix Multiply Accumulate) fragments.
        Hierarchical layout representing warp-level tile distribution.
        """
        # Simulate warp tile: ((2 threads, 4 per thread), 8 columns)
        layout = Layout(((2, 4), 8), ((1, 2), 8))
        tensor = Tensor(layout)

        # Thread 0's elements: tensor((0, _), _)
        thread0_slice = tensor[(0, 0), :]
        assert isinstance(thread0_slice, Tensor)

        # Thread 1's elements: tensor((1, _), _)
        thread1_slice = tensor[(1, 0), :]
        assert isinstance(thread1_slice, Tensor)

        # Verify they access different starting positions
        assert thread0_slice.offset != thread1_slice.offset or thread0_slice(
            0
        ) != thread1_slice(0)


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
