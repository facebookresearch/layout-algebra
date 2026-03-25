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

import pytest

from tensor_layouts import * # noqa: F401,F403,F405
from tensor_layouts.layout_utils import round_up


## Complement Layouts
# Tests adapted from NVIDIA CuTe: test/unit/cute/core/complement.cpp


def _test_complement_properties(layout, cotarget=None):
    """Test complement function properties as defined by CuTe.

    Verifies:
    1. cosize(completed) >= size(cotarget)
    2. cosize(result) <= round_up(size(cotarget), cosize(layout))
    3. Result is ordered: result(i-1) < result(i) for i >= 1
    4. Result is disjoint from layout: result(i) != layout(j) for i >= 1
    """
    if cotarget is None:
        cotarget = cosize(layout)

    cotarget_size = size(cotarget) if isinstance(cotarget, tuple) else cotarget

    result = complement(layout, cotarget)

    # Create completed layout by bundling layout and result
    # Always bundle them together - Layout(layout, result) works for any ranks
    completed = Layout(layout, result)

    # Property 1: Lower-bound on codomain size of layout ++ complement
    assert (
        cosize(completed) >= cotarget_size
    ), f"cosize(completed)={cosize(completed)} < size(cotarget)={cotarget_size}"

    # Property 2: Upper-bound on codomain size of complement
    # Always use cosize(layout), regardless of rank
    layout_cosize = cosize(layout)
    assert cosize(result) <= round_up(
        cotarget_size, layout_cosize
    ), f"cosize(result)={cosize(result)} > round_up({cotarget_size}, {layout_cosize})={round_up(cotarget_size, layout_cosize)}"

    # Property 3: Result is ordered (CuTe starts at i=1)
    for i in range(1, size(result)):
        assert result(i - 1) < result(
            i
        ), f"result is not ordered: result({i-1})={result(i-1)} >= result({i})={result(i)}"

    # Property 4: Result is disjoint from layout (CuTe starts at i=1)
    for i in range(1, size(result)):
        for j in range(size(layout)):
            assert result(i) != layout(
                j
            ), f"result and layout overlap: result({i})={result(i)} == layout({j})={layout(j)}"

    # Other observations from CuTe
    assert size(result) <= cosize(
        result
    ), f"size(result)={size(result)} > cosize(result)={cosize(result)}"


def test_complement_layout_1_0():
    # Layout<_1,_0>
    _test_complement_properties(Layout(1, 0))
    _test_complement_properties(Layout(1, 0), 2)
    _test_complement_properties(Layout(1, 0), 5)
    _test_complement_properties(Layout(1, 0), (2, 2))


def test_complement_layout_1_1():
    # Layout<_1,_1>
    _test_complement_properties(Layout(1, 1))
    _test_complement_properties(Layout(1, 1), 2)
    _test_complement_properties(Layout(1, 1), 5)
    _test_complement_properties(Layout(1, 1), (2, 2))


def test_complement_layout_1_2():
    # Layout<_1,_2>
    _test_complement_properties(Layout(1, 2), 1)
    _test_complement_properties(Layout(1, 2), 2)
    _test_complement_properties(Layout(1, 2), 8)
    _test_complement_properties(Layout(1, 2), 5)
    _test_complement_properties(Layout(1, 2), (2, 2))


def test_complement_layout_4_0():
    # Layout<_4,_0>
    _test_complement_properties(Layout(4, 0), 1)
    _test_complement_properties(Layout(4, 0), 2)
    _test_complement_properties(Layout(4, 0), 8)


def test_complement_layout_4_1():
    # Layout<_4,_1>
    _test_complement_properties(Layout(4, 1), 1)
    _test_complement_properties(Layout(4, 1), 2)
    _test_complement_properties(Layout(4, 1), 8)


def test_complement_layout_4_2():
    # Layout<_4,_2>
    _test_complement_properties(Layout(4, 2), 1)
    _test_complement_properties(Layout(4, 2))
    _test_complement_properties(Layout(4, 2), 16)
    _test_complement_properties(Layout(4, 2), 19)
    _test_complement_properties(Layout(4, 2), (2, 2))


def test_complement_layout_4_4():
    # Layout<_4,_4>
    _test_complement_properties(Layout(4, 4), 1)
    _test_complement_properties(Layout(4, 4))
    _test_complement_properties(Layout(4, 4), 17)
    _test_complement_properties(Layout(4, 4), (2, 2))


def test_complement_multidimensional():
    # Layout<Shape<_2,_4>> (auto strides)
    _test_complement_properties(Layout((2, 4)))

    # Layout<Shape<_2,_3>>
    _test_complement_properties(Layout((2, 3)))

    # Layout<Shape<_2,_4>, Stride<_1,_4>>
    _test_complement_properties(Layout((2, 4), (1, 4)))

    # Layout<Shape<_2,_4>, Stride<_1,_6>>
    _test_complement_properties(Layout((2, 4), (1, 6)))

    # Layout<Shape<_2,_4,_8>, Stride<_8,_1,_64>>
    _test_complement_properties(Layout((2, 4, 8), (8, 1, 64)))

    # Layout<Shape<_2,_4,_8>, Stride<_8,_1,_0>>
    _test_complement_properties(Layout((2, 4, 8), (8, 1, 0)))
    _test_complement_properties(Layout((2, 4, 8), (8, 1, 0)), 460)


def test_complement_nested():
    # Nested layouts: Shape<Shape<_2,_2>,Shape<_2,_2>>, Stride<Stride<_1,_4>,Stride<_8,_32>>
    _test_complement_properties(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))

    # Nested layouts: Shape<Shape<_2,_2>,Shape<_2,_2>>, Stride<Stride<_1,_32>,Stride<_8,_4>>
    _test_complement_properties(Layout(((2, 2), (2, 2)), ((1, 32), (8, 4))))


def test_complement_non_contiguous():
    # Layout<Shape<_4,_6>, Stride<_1,_6>>
    _test_complement_properties(Layout((4, 6), (1, 6)))

    # Layout<Shape<_4,_2>, Stride<_1,_10>>
    _test_complement_properties(Layout((4, 2), (1, 10)))

    # Layout<Shape<_4,_2>, Stride<_1,_16>>
    _test_complement_properties(Layout((4, 2), (1, 16)))


def test_complement_dynamic():
    # Dynamic shapes/strides
    _test_complement_properties(Layout(12), 1)
    _test_complement_properties(Layout(12))
    _test_complement_properties(Layout(12), 53)
    _test_complement_properties(Layout(12), 128)

    _test_complement_properties(Layout(12, 1), 1)
    _test_complement_properties(Layout(12, 1))
    _test_complement_properties(Layout(12, 1), 53)
    _test_complement_properties(Layout(12, 1), 128)

    _test_complement_properties(Layout(12, 2), 1)
    _test_complement_properties(Layout(12, 2))
    _test_complement_properties(Layout(12, 2), 53)
    _test_complement_properties(Layout(12, 2), 128)

    # Layout with shape (3,6) stride (1, 3)
    _test_complement_properties(Layout((3, 6), (1, 3)))

    # Layout with shape (3,6) stride (1, 9)
    _test_complement_properties(Layout((3, 6), (1, 9)))

    # Layout with shape (3,6) stride (1, 10)
    _test_complement_properties(Layout((3, 6), (1, 10)))

    # Nested with dynamic inner
    _test_complement_properties(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))

    # Layout(64) with shape cotarget
    _test_complement_properties(Layout(64), (32, 4, 4))


## Coalesce Layouts
# Tests adapted from NVIDIA CuTe: test/unit/cute/core/coalesce.cpp


def _test_coalesce_properties(layout):
    """Test coalesce function properties as defined by CuTe.

    Verifies:
    1. depth(coalesce_layout) <= 1
    2. size(coalesce_layout) == size(layout)
    3. coalesce_layout(i) == layout(i) for all i
    """
    coalesce_layout = coalesce(layout)

    # Property 1: Result depth is at most 1 (flattened)
    assert (
        depth(coalesce_layout) <= 1
    ), f"depth(coalesce_layout)={depth(coalesce_layout)} > 1"

    # Property 2: Size is preserved
    assert size(coalesce_layout) == size(
        layout
    ), f"size(coalesce_layout)={size(coalesce_layout)} != size(layout)={size(layout)}"

    # Property 3: All indices map to the same offsets
    for i in range(size(layout)):
        assert coalesce_layout(i) == layout(
            i
        ), f"coalesce_layout({i})={coalesce_layout(i)} != layout({i})={layout(i)}"


def test_coalesce_simple():
    # Layout(1, 0)
    _test_coalesce_properties(Layout(1, 0))

    # Layout(1, 1)
    _test_coalesce_properties(Layout(1, 1))

    # Layout((2, 4)) - auto strides (col-major)
    _test_coalesce_properties(Layout((2, 4)))

    # Layout((2, 4, 6)) - auto strides
    _test_coalesce_properties(Layout((2, 4, 6)))


def test_coalesce_size_one_dimension():
    # Layout with size-1 middle dimension
    # Shape: (2, 1, 6), Stride: (1, 6, 2)
    _test_coalesce_properties(Layout((2, 1, 6), (1, 6, 2)))

    # Layout with size-1 and dynamic stride
    # Shape: (2, 1, 6), Stride: (1, 7, 2)
    _test_coalesce_properties(Layout((2, 1, 6), (1, 7, 2)))

    # Layout with non-contiguous strides
    # Shape: (2, 1, 6), Stride: (4, 7, 8)
    _test_coalesce_properties(Layout((2, 1, 6), (4, 7, 8)))


def test_coalesce_mixed_dynamic():
    # Mixed static/dynamic shapes (all treated as dynamic in Python)
    # Shape: (2, 4, 6) - auto strides
    _test_coalesce_properties(Layout((2, 4, 6)))


def test_coalesce_row_major():
    # Row-major layouts (GenRowMajor equivalent)
    # For (2, 4) row-major: stride = (4, 1)
    _test_coalesce_properties(Layout((2, 4), (4, 1)))

    # (2, 4, 6) row-major: stride = (24, 6, 1)
    _test_coalesce_properties(Layout((2, 4, 6), (24, 6, 1)))

    # (2, 4, 6) row-major with first dim dynamic
    _test_coalesce_properties(Layout((2, 4, 6), (24, 6, 1)))

    # (2, 4, 6) row-major with middle dim dynamic
    _test_coalesce_properties(Layout((2, 4, 6), (24, 6, 1)))

    # (2, 4, 6) row-major with last dim dynamic
    _test_coalesce_properties(Layout((2, 4, 6), (24, 6, 1)))

    # Row-major with size-1 dimension
    # (2, 1, 3) row-major: stride = (3, 3, 1)
    _test_coalesce_properties(Layout((2, 1, 3), (3, 3, 1)))

    # (2, 1, 3) with size-1 as dynamic
    _test_coalesce_properties(Layout((2, 1, 3), (3, 3, 1)))

    # (2, 1, 3) with custom strides
    _test_coalesce_properties(Layout((2, 1, 3), (2, 4, 4)))

    # (2, 1, 3) with stride 0 for size-1 dimension
    _test_coalesce_properties(Layout((2, 1, 3), (2, 0, 4)))


def test_coalesce_nested():
    # Nested/hierarchical layout
    # Shape: ((2,2), (2,2)), Stride: ((1,4), (8,32))
    _test_coalesce_properties(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))


## Compose Layouts
# Tests adapted from NVIDIA CuTe: test/unit/cute/core/composition.cpp


def _test_composition_properties(layout_a, layout_b):
    """Test composition function properties as defined by CuTe.

    Verifies:
    1. compatible(layoutB, layoutR) is True
    2. layoutR(c) == layoutA(layoutB(c)) for all c where B(c) is within A's domain

    Note: The identity R(c) = A(B(c)) only holds when B(c) is within A's
    domain (i.e., B(c) < size(A)). When B extends beyond A's domain,
    composition truncates/zeroes the out-of-domain part, while A(B(c))
    uses last-mode extension. Both behaviors are correct per CuTe semantics.
    """
    layout_r = compose(layout_a, layout_b)

    # Property 1: Layout B is compatible with layout R
    assert compatible(
        layout_b.shape, layout_r.shape
    ), f"layoutB.shape={layout_b.shape} not compatible with layoutR.shape={layout_r.shape}"

    # Property 2: R(c) = A(B(c)) for coordinates within A's domain
    a_size = size(layout_a)
    for c in range(size(layout_b)):
        bc = layout_b(c)
        if bc < a_size:
            assert layout_r(c) == layout_a(
                bc
            ), f"layoutR({c})={layout_r(c)} != layoutA(layoutB({c}))={layout_a(bc)}"


def test_composition_simple():
    # Simple tests

    # Layout(1, 0) o Layout(1, 0)
    _test_composition_properties(Layout(1, 0), Layout(1, 0))

    # Layout(1, 0) o Layout(1, 1)
    _test_composition_properties(Layout(1, 0), Layout(1, 1))

    # Layout(1, 1) o Layout(1, 0)
    _test_composition_properties(Layout(1, 1), Layout(1, 0))

    # Layout(1, 1) o Layout(1, 1)
    _test_composition_properties(Layout(1, 1), Layout(1, 1))

    # Layout(4) o Layout(4)
    _test_composition_properties(Layout(4), Layout(4))

    # Layout(4, 2) o Layout(4)
    _test_composition_properties(Layout(4, 2), Layout(4))

    # Layout(4, 0) o Layout(4)
    _test_composition_properties(Layout(4, 0), Layout(4))

    # Layout(4) o Layout(4, 0)
    _test_composition_properties(Layout(4), Layout(4, 0))

    # Layout(4) o Layout(1, 0)
    _test_composition_properties(Layout(4), Layout(1, 0))


def test_composition_partial():
    # Layout(4) o Layout(2)
    _test_composition_properties(Layout(4), Layout(2))

    # Layout(4, 2) o Layout(2)
    _test_composition_properties(Layout(4, 2), Layout(2))

    # Layout(4) o Layout(2, 2)
    _test_composition_properties(Layout(4), Layout(2, 2))

    # Layout(4, 2) o Layout(2, 2)
    _test_composition_properties(Layout(4, 2), Layout(2, 2))


def test_composition_multidimensional():
    # Layout((4, 3)) o Layout(12)
    _test_composition_properties(Layout((4, 3)), Layout(12))

    # Layout(12) o Layout((4, 3))
    _test_composition_properties(Layout(12), Layout((4, 3)))

    # Layout(12, 2) o Layout((4, 3))
    _test_composition_properties(Layout(12, 2), Layout((4, 3)))

    # Layout(12) o Layout((4, 3), (3, 1))
    _test_composition_properties(Layout(12), Layout((4, 3), (3, 1)))

    # Layout(12, 2) o Layout((4, 3), (3, 1))
    _test_composition_properties(Layout(12, 2), Layout((4, 3), (3, 1)))

    # Layout(12) o Layout((2, 3), (2, 4))
    _test_composition_properties(Layout(12), Layout((2, 3), (2, 4)))

    # Layout((4, 3)) o Layout((4, 3))
    _test_composition_properties(Layout((4, 3)), Layout((4, 3)))

    # Layout((4, 3)) o Layout(6)
    _test_composition_properties(Layout((4, 3)), Layout(6))

    # Layout((4, 3)) o Layout(6, 2)
    _test_composition_properties(Layout((4, 3)), Layout(6, 2))

    # Layout((4, 3)) o Layout((6, 2), (2, 1))
    _test_composition_properties(Layout((4, 3)), Layout((6, 2), (2, 1)))

    # Layout((4, 3)) o Layout((4, 3), (3, 1))
    _test_composition_properties(Layout((4, 3)), Layout((4, 3), (3, 1)))

    # Layout((4, 3), (3, 1)) o Layout((4, 3))
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout((4, 3)))

    # Layout((4, 3), (3, 1)) o Layout(12)
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout(12))

    # Layout((4, 3), (3, 1)) o Layout(6, 2)
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout(6, 2))

    # Layout((4, 3), (3, 1)) o Layout((6, 2), (2, 1))
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout((6, 2), (2, 1)))


def test_composition_nested():
    # Layout((8, 8)) o Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
    _test_composition_properties(
        Layout((8, 8)), Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
    )

    # Layout((8, 8), (8, 1)) o Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
    _test_composition_properties(
        Layout((8, 8), (8, 1)), Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
    )

    # Layout(((4, 2),), ((1, 16),)) o Layout((4, 2), (2, 1))
    _test_composition_properties(Layout(((4, 2),), ((1, 16),)), Layout((4, 2), (2, 1)))

    # Layout((2, 2), (2, 1)) o Layout((2, 2), (2, 1))
    _test_composition_properties(Layout((2, 2), (2, 1)), Layout((2, 2), (2, 1)))

    # Layout((4, 8, 2)) o Layout((2, 2, 2), (2, 8, 1))
    _test_composition_properties(Layout((4, 8, 2)), Layout((2, 2, 2), (2, 8, 1)))

    # Layout((4, 8, 2), (2, 8, 1)) o Layout((2, 2, 2), (1, 8, 2))
    _test_composition_properties(
        Layout((4, 8, 2), (2, 8, 1)), Layout((2, 2, 2), (1, 8, 2))
    )

    # Layout((4, 8, 2), (2, 8, 1)) o Layout((4, 2, 2), (2, 8, 1))
    _test_composition_properties(
        Layout((4, 8, 2), (2, 8, 1)), Layout((4, 2, 2), (2, 8, 1))
    )


def test_composition_dynamic():
    # Dynamic shapes/strides

    # Layout(12, 1) o Layout(4)
    _test_composition_properties(Layout(12, 1), Layout(4))

    # Layout((12, 3), (1, 24)) o Layout(4)
    _test_composition_properties(Layout((12, 3), (1, 24)), Layout(4))

    # Layout(16, 2) o Layout(4, 2)
    _test_composition_properties(Layout(16, 2), Layout(4, 2))

    # Layout((128, 24, 5), (1, 128, 3072)) o Layout(64, 2)
    _test_composition_properties(Layout((128, 24, 5), (1, 128, 3072)), Layout(64, 2))

    # Layout((128, 24, 5), (1, 128, 3072)) o Layout(480, 32)
    _test_composition_properties(Layout((128, 24, 5), (1, 128, 3072)), Layout(480, 32))


def test_composition_cosize_larger():
    # cosize(b) > size(a) and divisibility

    # Layout(1, 0) o Layout(4)
    _test_composition_properties(Layout(1, 0), Layout(4))

    # Layout(1, 1) o Layout(4)
    _test_composition_properties(Layout(1, 1), Layout(4))

    # Layout(4) o Layout(4, 2)
    _test_composition_properties(Layout(4), Layout(4, 2))


def test_composition_extension():
    # Last mode gets extended
    # Layout((4, 3), (3, 1)) o Layout(24)
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout(24))

    # Last mode extension even without last mode divisibility
    # Layout((4, 3), (3, 1)) o Layout(8)
    _test_composition_properties(Layout((4, 3), (3, 1)), Layout(8))

    # Capping a Layout with 1:0 extends in stride-0
    # Layout((4, 3, 1), (3, 1, 0)) o Layout(24)
    _test_composition_properties(Layout((4, 3, 1), (3, 1, 0)), Layout(24))

    # Layout((4, 3, 1), (3, 1, 0)) o Layout(4)
    _test_composition_properties(Layout((4, 3, 1), (3, 1, 0)), Layout(4))


def test_composition_truncation():
    # Pre-coalesced LHS
    # Layout((4, 6, 8), (1, 4, 7)) o Layout(6)
    _test_composition_properties(Layout((4, 6, 8), (1, 4, 7)), Layout(6))

    # Mid-layout truncation
    # Layout((4, 6, 8, 10), (2, 3, 5, 7)) o Layout(6, 12)
    _test_composition_properties(Layout((4, 6, 8, 10), (2, 3, 5, 7)), Layout(6, 12))

    # Layout((8, 8), (8, 1)) o Layout(2, 3) - invalid: stride 3 doesn't satisfy
    # the divisibility requirement with coalesced shape 64. pycute also asserts.
    with pytest.raises(ValueError):
        _test_composition_properties(Layout((8, 8), (8, 1)), Layout(2, 3))

    # Layout((8, 8), (8, 1)) o Layout(3, 3) - same issue
    with pytest.raises(ValueError):
        _test_composition_properties(Layout((8, 8), (8, 1)), Layout(3, 3))

    # Layout(3, 1) o Layout(4)
    _test_composition_properties(Layout(3, 1), Layout(4))

    # Layout((48, 24, 5), (1, 128, 3072)) o Layout(32, 1)
    _test_composition_properties(Layout((48, 24, 5), (1, 128, 3072)), Layout(32, 1))

    # Note: Swizzle composition tests and negative stride tests are skipped
    # as they require additional functionality not yet implemented


## Logical Divide
# Tests adapted from NVIDIA CuTe: test/unit/cute/core/logical_divide.cpp


def _test_logical_divide_properties(layout, tile):
    """Test logical_divide function properties as defined by CuTe.

    Verifies:
    1. Result has the correct structure
    2. Size is preserved (for simple cases) or properly extended (for tile > layout)

    Note: When tile is a Layout, CuTe uses the formula:
        logical_divide(A, B) = compose(A, Layout(B, complement(B, size(A))))
    This produces a different structure than mode-by-mode division with tuple tilers.
    """
    result = logical_divide(layout, tile)

    # Normalize tile to Layout for comparison
    tile_layout = tile if isinstance(tile, Layout) else Layout(tile)

    # For Layout tilers, the result is (Tile, Rest) structure from the CuTe formula
    # For tuple tilers, the result is ((TileM, RestM), (TileN, RestN), ...)

    # Basic property: result should be a valid layout
    assert isinstance(result, Layout), f"Expected Layout, got {type(result)}"

    # For Layout tilers, verify the result rank is 2 (tile, rest)
    if isinstance(tile, Layout):
        # CuTe formula produces rank-2 result: (Tile, Rest)
        assert (
            rank(result) == 2
        ), f"Expected rank 2 for Layout tiler, got {rank(result)}"

        # The tile part (mode 0) should have size equal to size(tiler)
        tile_part = mode(result, 0)
        assert size(tile_part) == size(
            tile_layout
        ), f"Tile part size {size(tile_part)} != tiler size {size(tile_layout)}"


def test_logical_divide_simple():
    # Layout(1, 0) / Layout(1, 0)
    _test_logical_divide_properties(Layout(1, 0), Layout(1, 0))

    # Layout(1, 0) / Layout(1, 1)
    _test_logical_divide_properties(Layout(1, 0), Layout(1, 1))

    # Layout(1, 1) / Layout(1, 0)
    _test_logical_divide_properties(Layout(1, 1), Layout(1, 0))

    # Layout(1, 1) / Layout(1, 1)
    _test_logical_divide_properties(Layout(1, 1), Layout(1, 1))


def test_logical_divide_stride_variations():
    # Layout(6, 1) / Layout(2, 1)
    _test_logical_divide_properties(Layout(6, 1), Layout(2, 1))

    # Layout(6, 1) / Layout(2, 3)
    _test_logical_divide_properties(Layout(6, 1), Layout(2, 3))

    # Layout(6, 1) / Layout((2, 3), (3, 1))
    _test_logical_divide_properties(Layout(6, 1), Layout((2, 3), (3, 1)))

    # Layout(6, 2) / Layout(2, 1)
    _test_logical_divide_properties(Layout(6, 2), Layout(2, 1))

    # Layout(6, 2) / Layout(2, 3)
    _test_logical_divide_properties(Layout(6, 2), Layout(2, 3))

    # Layout(6, 2) / Layout((2, 3), (3, 1))
    _test_logical_divide_properties(Layout(6, 2), Layout((2, 3), (3, 1)))


def test_logical_divide_multidimensional():
    # Layout((6, 6), (1, 12)) / Layout((6, 3), (3, 1))
    _test_logical_divide_properties(Layout((6, 6), (1, 12)), Layout((6, 3), (3, 1)))

    # Layout((6, 6), (12, 1)) / Layout((6, 3), (3, 1))
    _test_logical_divide_properties(Layout((6, 6), (12, 1)), Layout((6, 3), (3, 1)))

    # Layout(32) / Layout(2, 8)
    _test_logical_divide_properties(Layout(32), Layout(2, 8))


def test_logical_divide_size_one():
    # Layout((4, 1), (1, 1)) / Layout(2, 1)
    _test_logical_divide_properties(Layout((4, 1), (1, 1)), Layout(2, 1))

    # Layout((4, 1), (1, 1)) / Layout(2, 2)
    _test_logical_divide_properties(Layout((4, 1), (1, 1)), Layout(2, 2))


def test_logical_divide_large():
    # Layout((8, 8), (1, 8)) / Layout(32, 2)
    _test_logical_divide_properties(Layout((8, 8), (1, 8)), Layout(32, 2))

    # Layout((8, 8), (8, 1)) / Layout(32, 2)
    _test_logical_divide_properties(Layout((8, 8), (8, 1)), Layout(32, 2))


def test_logical_divide_dynamic():
    # Dynamic cases - verify basic structure
    # For Layout tilers, the result is (Tile, Rest) where these are flat values, not nested tuples

    # layout(2) / Layout(32) - tile larger than layout
    result = logical_divide(Layout(2), Layout(32))
    # Result should be rank 2 with (Tile, Rest) structure
    assert rank(result) == 2, f"Expected rank 2, got {rank(result)}"

    # layout(48) / Layout(32)
    result = logical_divide(Layout(48), Layout(32))
    # Tile should be 32, Rest should be 2 (ceil(48/32) = 2)
    assert result.shape[0] == 32, f"Expected tile shape 32, got {result.shape[0]}"
    assert result.shape[1] == 2, f"Expected rest shape 2, got {result.shape[1]}"

    # layout(96) / Layout(32, 2)
    _test_logical_divide_properties(Layout(96), Layout(32, 2))

    # layout(32) / Layout(48) - tile larger than layout
    result = logical_divide(Layout(32), Layout(48))
    # Tile should be 48 (tiler shape), Rest should be ceil(32/48) = 1
    assert rank(result) == 2, f"Expected rank 2, got {rank(result)}"
    assert result.shape[0] == 48, f"Expected tile shape 48, got {result.shape[0]}"

    # layout((32, 4, 4)) / Layout(64)
    result = logical_divide(Layout((32, 4, 4)), Layout(64))
    # For a 1D Layout tiler, the result should have rank 2: (Tile, Rest)
    assert rank(result) == 2, f"Expected rank 2, got {rank(result)}"


def test_logical_divide_dangerous_dynamic():
    # ALLOWED, but dangerous due to the dynamic lhs shapes

    # Layout((128, 4, 3), (1, 512, 0)) / Layout(32)
    _test_logical_divide_properties(Layout((128, 4, 3), (1, 512, 0)), Layout(32))

    # Layout((128, 4, 3), (1, 512, 0)) / Layout(32, 2)
    _test_logical_divide_properties(Layout((128, 4, 3), (1, 512, 0)), Layout(32, 2))

    # Layout((16, 4, 3), (1, 512, 0)) / Layout(32)
    _test_logical_divide_properties(Layout((16, 4, 3), (1, 512, 0)), Layout(32))


## Swizzle Tests


def _test_swizzle_2d(sw_layout):
    """Test swizzle layout with dynamic and static slicing.

    Verifies that:
    1. Dynamic row slicing: tensor[i, :](j) == tensor(i, j)
    2. Dynamic column slicing: tensor[:, j](i) == tensor(i, j)

    This tests that slicing a tensor with swizzled layout preserves correct indexing.
    """
    from tensor_layouts import Tensor

    tensor = Tensor(sw_layout)

    # Get dimensions
    shape = sw_layout.shape
    dim0 = shape[0] if isinstance(shape[0], int) else size(shape[0])
    dim1 = shape[1] if isinstance(shape[1], int) else size(shape[1])

    # Dynamic row slicing: tensor[i, :](j) == tensor(i, j)
    for i in range(dim0):
        sliced = tensor[i, :]
        for j in range(dim1):
            expected = tensor(i, j)
            actual = sliced(j)
            assert actual == expected, (
                f"Row slice mismatch at tensor[{i}, :]({j}): "
                f"expected {expected}, got {actual}"
            )

    # Dynamic column slicing: tensor[:, j](i) == tensor(i, j)
    for j in range(dim1):
        sliced = tensor[:, j]
        for i in range(dim0):
            expected = tensor(i, j)
            actual = sliced(i)
            assert actual == expected, (
                f"Column slice mismatch at tensor[:, {j}]({i}): "
                f"expected {expected}, got {actual}"
            )


def test_swizzle_3_0_3():
    """Test Swizzle(3, 0, 3) with 8x8 row-major layout.

    Swizzle<3,0,3> means:
    - bits=3: 3 bits in each mask
    - base=0: no bits to keep constant at the bottom
    - shift=3: masks are 3 positions apart

    With 8x8 row-major (stride 8 for rows, 1 for cols):
    - Row index contributes bits [3,6)
    - Col index contributes bits [0,3)
    - XOR creates the swizzle pattern
    """
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))  # 8x8 row-major

    _test_swizzle_2d(sw_layout)


def test_swizzle_3_0_neg3():
    """Test Swizzle(3, 0, -3) with 8x8 row-major layout.

    Swizzle<3,0,-3> with negative shift means:
    - The XOR goes in the opposite direction
    - Bits at [0,3) are shifted left and XORed into bits [3,6)
    """
    sw_layout = compose(Swizzle(3, 0, -3), Layout((8, 8), (8, 1)))  # 8x8 row-major

    _test_swizzle_2d(sw_layout)


def test_swizzle_2_1_3():
    """Test Swizzle(2, 1, 3) with nested 2D layout.

    The layout is:
        Shape:  ((2, 2, 2), (2, 2, 2))
        Stride: ((32, 2, 8), (4, 1, 16))

    This creates an 8x8 layout with a complex stride pattern.

    Swizzle<2,1,3> means:
    - bits=2: 2 bits in each mask
    - base=1: keep 1 least-significant bit constant
    - shift=3: masks are 3 positions apart (bits [1,3) and [4,6))
    """
    sw_layout = compose(
        Swizzle(2, 1, 3), Layout(((2, 2, 2), (2, 2, 2)), ((32, 2, 8), (4, 1, 16)))
    )

    _test_swizzle_2d(sw_layout)


def test_swizzle_basic():
    """Test basic Swizzle operations."""
    # Swizzle(3, 0, 3): XOR bits [0,3) with bits [3,6)
    sw = Swizzle(3, 0, 3)

    # Test index 0: bits all 0, XOR gives 0
    assert sw(0) == 0, f"sw(0) = {sw(0)}, expected 0"

    # Test index 8 (binary 001_000): bits [3,6)=001, bits [0,3)=000
    # XOR: 001 ^ 000 = 001, placed at [0,3) -> result = 001_001 = 9
    assert sw(8) == 9, f"sw(8) = {sw(8)}, expected 9"

    # Test index 9 (binary 001_001): bits [3,6)=001, bits [0,3)=001
    # XOR: 001 ^ 001 = 000, placed at [0,3) -> result = 001_000 = 8
    assert sw(9) == 8, f"sw(9) = {sw(9)}, expected 8"

    # Test index 1 (binary 000_001): bits [3,6)=000, bits [0,3)=001
    # XOR: 000 ^ 001 = 001, placed at [0,3) -> result = 000_001 = 1
    assert sw(1) == 1, f"sw(1) = {sw(1)}, expected 1"


def test_swizzle_negative_shift():
    """Test Swizzle with negative shift."""
    # Swizzle(3, 0, -3): XOR bits [0,3) into bits [3,6)
    sw = Swizzle(3, 0, -3)

    # Test index 0: all bits 0
    assert sw(0) == 0, f"sw(0) = {sw(0)}, expected 0"

    # Test index 1 (binary 000_001): bits [0,3)=001
    # Shift left by 3: 001_000 = 8
    # XOR with original: 000_001 ^ 001_000 = 001_001 = 9
    assert sw(1) == 9, f"sw(1) = {sw(1)}, expected 9"

    # Test index 8 (binary 001_000): bits [0,3)=000
    # Shift left by 3: 000_000 = 0
    # XOR with original: 001_000 ^ 000_000 = 001_000 = 8
    assert sw(8) == 8, f"sw(8) = {sw(8)}, expected 8"

    # Test index 9 (binary 001_001): bits [0,3)=001
    # Shift left by 3: 001_000 = 8
    # XOR with original: 001_001 ^ 001_000 = 000_001 = 1
    assert sw(9) == 1, f"sw(9) = {sw(9)}, expected 1"


def test_swizzle_with_base():
    """Test Swizzle with non-zero base."""
    # Swizzle(2, 1, 3): bits [1,3) XOR with bits [4,6), keep bit 0 constant
    sw = Swizzle(2, 1, 3)

    # Index 0: all zero
    assert sw(0) == 0

    # Index 2 (binary 00_010): bits [1,3)=01, bits [4,6)=00
    # mask = 0b110 (bits 1 and 2)
    # (2 >> 3) & 0b110 = 0
    # result = 2 ^ 0 = 2
    assert sw(2) == 2, f"sw(2) = {sw(2)}, expected 2"

    # Index 16 (binary 10000): bits [1,3)=00, bits [4,6)=01
    # (16 >> 3) & 0b110 = 0b10 = 2
    # result = 16 ^ 2 = 18
    assert sw(16) == 18, f"sw(16) = {sw(16)}, expected 18"

    # Index 18 (binary 10010): bits [1,3)=01, bits [4,6)=01
    # (18 >> 3) & 0b110 = 0b10 = 2
    # result = 18 ^ 2 = 16
    assert sw(18) == 16, f"sw(18) = {sw(18)}, expected 16"


def test_composed_layout_repr():
    """Test swizzled Layout string representation."""
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))

    repr_str = repr(sw_layout)
    assert "Swizzle(3, 0, 3)" in repr_str
    assert "(8, 8)" in repr_str


def test_composed_layout_shape():
    """Test that swizzled Layout preserves shape."""
    base_layout = Layout((8, 8), (8, 1))
    sw_layout = compose(Swizzle(3, 0, 3), base_layout)

    assert sw_layout.shape == base_layout.shape
    assert sw_layout.stride == base_layout.stride


###############################################################################
## Tests ported from NVIDIA pycute Python test suite
## Source: https://github.com/NVIDIA/cutlass/tree/main/test/python/pycute/
###############################################################################


## Int Tuple Operations
# Tests adapted from pycute test_int_tuple.py


def test_inner_product():
    """Test inner_product (pycute test_int_tuple.py::test_inner_product)."""
    assert inner_product(2, 3) == 6
    assert inner_product((1, 2), (3, 2)) == 7
    assert inner_product(((2, 3), 4), ((2, 1), 2)) == 15


def test_prefix_product():
    """Test prefix_product (pycute test_int_tuple.py::test_prefix_product)."""
    assert prefix_product(2) == 1
    assert prefix_product((3, 2)) == (1, 3)
    assert prefix_product((3, 2, 4)) == (1, 3, 6)
    assert prefix_product(((2, 3), 4)) == ((1, 2), 6)
    assert prefix_product(((2, 3), (2, 1, 2), (5, 2, 1))) == (
        (1, 2),
        (6, 12, 12),
        (24, 120, 240),
    )


def test_shape_div_pycute():
    """Test shape_div with pycute test cases (test_int_tuple.py::test_shape_div).

    These include nested tuple cases not covered by our other tests.
    """
    assert shape_div((3, 4), 6) == (1, 2)
    assert shape_div((3, 4), 12) == (1, 1)
    assert shape_div((3, 4), 36) == (1, 1)
    # Nested tuple cases
    assert shape_div(((3, 4), 6), 36) == ((1, 1), 2)
    assert shape_div((6, (3, 4)), 36) == (1, (1, 2))


## Complement (pycute Python test_complement.py)
# Uses disjointness property instead of our C++ property-based approach


def _test_complement_disjointness(layout):
    """Test complement using pycute's disjointness property.

    Post-condition: for all a in dom(layout), b in dom(complement),
    layout(a) != complement(b) unless both are 0.
    """
    layoutR = complement(layout)

    for a in range(size(layout)):
        for b in range(size(layoutR)):
            assert (layout(a) != layoutR(b)) or (layout(a) == 0 and layoutR(b) == 0), (
                f"Disjointness violated: layout({a})={layout(a)} == "
                f"complement({b})={layoutR(b)}"
            )


def test_complement_pycute():
    """Test complement with all pycute Python test cases (test_complement.py)."""
    _test_complement_disjointness(Layout(1, 0))
    _test_complement_disjointness(Layout(1, 1))
    _test_complement_disjointness(Layout(4, 0))
    _test_complement_disjointness(Layout((2, 4), (1, 2)))
    _test_complement_disjointness(Layout((2, 3), (1, 2)))
    _test_complement_disjointness(Layout((2, 4), (1, 4)))
    _test_complement_disjointness(Layout((2, 4, 8), (8, 1, 64)))
    _test_complement_disjointness(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))
    _test_complement_disjointness(Layout((2, (3, 4)), (3, (1, 6))))
    _test_complement_disjointness(Layout((4, 6), (1, 6)))
    _test_complement_disjointness(Layout((4, 10), (1, 10)))


## Coalesce (pycute Python test_coalesce.py)


def test_coalesce_pycute():
    """Test coalesce with all pycute Python test cases (test_coalesce.py).

    Uses the pycute helper: verify size and functional equivalence.
    """

    def _check(layout):
        layoutR = coalesce(layout)
        assert size(layoutR) == size(layout)
        for i in range(size(layout)):
            assert layoutR(i) == layout(
                i
            ), f"coalesce({layout})({i}) = {layoutR(i)} != {layout(i)}"

    _check(Layout(1, 0))
    _check(Layout(1, 1))
    _check(Layout((2, 4)))
    _check(Layout((2, 4, 6)))
    _check(Layout((2, 4, 6), (1, 6, 2)))
    _check(Layout((2, 1, 6), (1, 7, 2)))
    _check(Layout((2, 1, 6), (4, 7, 8)))
    _check(Layout((2, (4, 6))))
    _check(Layout((2, 4), (4, 1)))
    _check(Layout((2, 4, 6), (24, 6, 1)))
    _check(Layout((2, 1, 3), (2, 4, 4)))
    _check(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))


## Composition (pycute Python test_composition.py)
# Port cases not already covered by our C++ tests


def test_composition_pycute():
    """Test composition with all pycute Python test cases (test_composition.py).

    Uses the pycute helper: R(i) == A(B(i)) for all i.
    """

    def _check(A, B):
        R = compose(A, B)
        for i in range(size(R)):
            assert R(i) == A(
                B(i)
            ), f"compose({A}, {B})({i}) = {R(i)} != A(B({i})) = {A(B(i))}"

    # All test cases from pycute test_composition.py
    _check(Layout(1, 0), Layout(1, 0))
    _check(Layout(1, 0), Layout(1, 1))
    _check(Layout(1, 1), Layout(1, 0))
    _check(Layout(1, 1), Layout(1, 1))
    _check(Layout(4), Layout(4))
    _check(Layout(4, 2), Layout(4))
    _check(Layout(4), Layout(4, 2))
    _check(Layout(4, 0), Layout(4))
    _check(Layout(4), Layout(4, 0))
    _check(Layout(1, 0), Layout(4))
    _check(Layout(4), Layout(1, 0))
    _check(Layout(4), Layout(2))
    _check(Layout(4, 2), Layout(2))
    _check(Layout(4), Layout(2, 2))
    _check(Layout(4, 2), Layout(2, 2))
    _check(Layout(12), Layout((4, 3)))
    _check(Layout(12, 2), Layout((4, 3)))
    _check(Layout(12), Layout((4, 3), (3, 1)))
    _check(Layout(12, 2), Layout((4, 3), (3, 1)))
    _check(Layout(12), Layout((2, 3), (2, 4)))
    _check(Layout((4, 3)), Layout((4, 3)))
    _check(Layout((4, 3)), Layout(12))
    _check(Layout((4, 3)), Layout(6, 2))
    _check(Layout((4, 3)), Layout((6, 2), (2, 1)))
    _check(Layout((4, 3), (3, 1)), Layout((4, 3)))
    _check(Layout((4, 3), (3, 1)), Layout(12))
    _check(Layout((4, 3), (3, 1)), Layout(6, 2))
    _check(Layout((4, 3), (3, 1)), Layout((6, 2), (2, 1)))
    _check(Layout((8, 8)), Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))))
    _check(
        Layout((8, 8), (8, 1)), Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
    )
    # Layout applied from right with stride (from pycute, not in C++ tests)
    _check(Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))), Layout(8, 4))
    _check(Layout((4, 2), (1, 16)), Layout((4, 2), (2, 1)))
    _check(Layout((2, 2), (2, 1)), Layout((2, 2), (2, 1)))
    _check(Layout((4, 8, 2)), Layout((2, 2, 2), (2, 8, 1)))
    _check(Layout((4, 8, 2), (2, 8, 1)), Layout((2, 2, 2), (1, 8, 2)))
    _check(Layout((4, 8, 2), (2, 8, 1)), Layout((4, 2, 2), (2, 8, 1)))
    # Pre-coalesced LHS
    _check(Layout((4, 6, 8), (1, 4, 7)), Layout(6))
    # Mid-layout truncation
    _check(Layout((4, 6, 8, 10), (2, 3, 5, 7)), Layout(6, 12))


## Right Inverse
# Tests from pycute test_right_inverse.py


def _test_right_inverse(layout):
    """Test right_inverse: L(R(i)) == i for all i in range(size(R))."""
    inv_layout = right_inverse(layout)
    for i in range(size(inv_layout)):
        assert layout(inv_layout(i)) == i, (
            f"right_inverse({layout}): L(R({i})) = " f"{layout(inv_layout(i))} != {i}"
        )


def test_right_inverse_trivial():
    """Right-inverse of trivial layouts (pycute test_right_inverse.py)."""
    _test_right_inverse(Layout(1, 0))
    _test_right_inverse(Layout((1, 1), (0, 0)))
    _test_right_inverse(Layout((3, 7), (0, 0)))
    _test_right_inverse(Layout(1, 1))
    _test_right_inverse(Layout(4, 0))


def test_right_inverse_1d():
    """Right-inverse of 1D layouts (pycute test_right_inverse.py)."""
    _test_right_inverse(Layout(4, 1))
    _test_right_inverse(Layout(4, 2))


def test_right_inverse_2d():
    """Right-inverse of 2D layouts (pycute test_right_inverse.py)."""
    _test_right_inverse(Layout((2, 4), (0, 2)))
    _test_right_inverse(Layout((8, 4), (1, 8)))
    _test_right_inverse(Layout((8, 4), (4, 1)))
    _test_right_inverse(Layout((4, 2), (1, 16)))


def test_right_inverse_3d():
    """Right-inverse of 3D layouts (pycute test_right_inverse.py)."""
    _test_right_inverse(Layout((2, 4, 6), (1, 2, 8)))
    _test_right_inverse(Layout((2, 4, 6), (4, 1, 8)))


## Left Inverse
# Tests from pycute test_left_inverse.py


def _test_left_inverse(layout):
    """Test left_inverse: R(L(i)) == i for all i in range(size(L))."""
    inv_layout = left_inverse(layout)
    for i in range(size(layout)):
        assert inv_layout(layout(i)) == i, (
            f"left_inverse({layout}): R(L({i})) = " f"{inv_layout(layout(i))} != {i}"
        )


def test_left_inverse_trivial():
    """Left-inverse of trivial layouts (pycute test_left_inverse.py)."""
    _test_left_inverse(Layout(1, 0))
    _test_left_inverse(Layout((1, 1), (0, 0)))
    _test_left_inverse(Layout(1, 1))


def test_left_inverse_1d():
    """Left-inverse of 1D layouts (pycute test_left_inverse.py)."""
    _test_left_inverse(Layout(4, 1))
    _test_left_inverse(Layout(4, 2))


def test_left_inverse_2d():
    """Left-inverse of 2D layouts (pycute test_left_inverse.py)."""
    _test_left_inverse(Layout((8, 4), (1, 8)))
    _test_left_inverse(Layout((8, 4), (4, 1)))
    _test_left_inverse(Layout((4, 2), (1, 16)))


def test_left_inverse_3d():
    """Left-inverse of 3D layouts (pycute test_left_inverse.py)."""
    _test_left_inverse(Layout((2, 4, 6), (1, 2, 8)))
    _test_left_inverse(Layout((2, 4, 6), (4, 1, 8)))


###############################################################################
## Tests ported from NVIDIA CuTe C++ test suite
## Source: https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute/core/
##
## These fill gaps not covered by the pycute Python test suite.
###############################################################################


## Logical Product (C++ test_unit_cute/core/logical_product.cpp)
# The C++ test checks: rank(R)==2, A==layout<0>(R), compatible(B, layout<1>(R))


def _test_logical_product_properties(layout_a, layout_b):
    """Test logical_product properties from C++ test suite.

    Verifies:
    1. rank(R) == 2 (result always has 2 modes)
    2. layout_a equals first mode of R
    3. layout_b is compatible with second mode of R
    """
    R = logical_product(layout_a, layout_b)

    # Property 1: Result has rank 2
    assert (
        rank(R) == 2
    ), f"logical_product({layout_a}, {layout_b}): rank={rank(R)}, expected 2"

    # Property 2: First mode of R equals A
    R0 = mode(R, 0)
    assert R0.shape == layout_a.shape and R0.stride == layout_a.stride, (
        f"logical_product({layout_a}, {layout_b}): " f"mode(R,0)={R0} != A={layout_a}"
    )

    # Property 3: B is compatible with second mode of R
    R1 = mode(R, 1)
    assert compatible(layout_b.shape, R1.shape), (
        f"logical_product({layout_a}, {layout_b}): "
        f"B.shape={layout_b.shape} not compatible with mode(R,1).shape={R1.shape}"
    )


def test_logical_product_trivial():
    """Logical product trivial cases (C++ logical_product.cpp lines 56-103)."""
    _test_logical_product_properties(Layout(1, 0), Layout(1, 0))
    _test_logical_product_properties(Layout(1, 1), Layout(1, 0))
    _test_logical_product_properties(Layout(1, 0), Layout(1, 1))
    _test_logical_product_properties(Layout(1, 1), Layout(1, 1))


def test_logical_product_broadcast():
    """Logical product with broadcast (stride-0) (C++ lines 84-103)."""
    _test_logical_product_properties(Layout(3, 1), Layout(4, 0))
    _test_logical_product_properties(Layout(3, 0), Layout(4, 1))
    _test_logical_product_properties(Layout(3, 0), Layout(4, 0))


def test_logical_product_1d():
    """Logical product 1D cases (C++ lines 105-172)."""
    _test_logical_product_properties(Layout(3, 2), Layout(4, 1))
    _test_logical_product_properties(Layout(3, 2), Layout(4))
    _test_logical_product_properties(Layout(3, 32), Layout(32))
    _test_logical_product_properties(Layout(3, 32), Layout(128))


def test_logical_product_multidim_tile():
    """Logical product with multi-dimensional tiles (C++ lines 112-137)."""
    # Layout(3) x Layout((2,4))
    _test_logical_product_properties(Layout(3), Layout((2, 4)))
    # Layout((2,4)) x Layout(3)
    _test_logical_product_properties(Layout((2, 4)), Layout(3))
    # Layout((8,(2,2))) x Layout(4,2)
    _test_logical_product_properties(Layout((8, (2, 2)), (1, (8, 16))), Layout(4, 2))
    # Layout((2,2)) x Layout((3,3),(3,1))
    _test_logical_product_properties(Layout((2, 2), (1, 2)), Layout((3, 3), (3, 1)))


def test_logical_product_large_stride():
    """Logical product with large strides (C++ lines 140-172)."""
    _test_logical_product_properties(Layout(3, 32), Layout((8, 8)))
    _test_logical_product_properties(Layout(3, 32), Layout((8, 8), (8, 1)))


def test_logical_product_nested():
    """Logical product with nested/hierarchical layouts (C++ lines 175-213)."""
    # Layout(((4,2)),((1,16))) x Layout((4,4))
    _test_logical_product_properties(Layout((4, 2), (1, 16)), Layout((4, 4)))
    # Layout(((4,2)),((1,16))) x Layout((4,2),(2,1))
    _test_logical_product_properties(Layout((4, 2), (1, 16)), Layout((4, 2), (2, 1)))
    # Layout(((2,2),(2,2)),((1,4),(8,32))) x Layout((2,2),(1,2))
    _test_logical_product_properties(
        Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
        Layout((2, 2), (1, 2)),
    )
    # Layout(((2,2),(2,2)),((1,4),(8,32))) x Layout((2,2),(2,1))
    _test_logical_product_properties(
        Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
        Layout((2, 2), (2, 1)),
    )
    # Layout(((4,6)),((1,6))) x Layout(3,1)
    _test_logical_product_properties(Layout((4, 6), (1, 6)), Layout(3, 1))


## Left Inverse edge cases (C++ inverse_left.cpp)
# C++ test checks: layout(inv(layout(i))) == layout(i) for all i


def _has_broadcast(layout):
    """Check if layout has any stride-0 mode with shape > 1 (O(rank))."""
    flat = flatten(layout)
    # Handle scalar layouts (shape is int, not tuple)
    if isinstance(flat.shape, int):
        return flat.stride == 0 and flat.shape > 1
    return any(d == 0 and s > 1 for s, d in zip(flat.shape, flat.stride))


def _test_left_inverse_cpp(layout):
    """Test left_inverse using the C++ test's property where applicable.

    C++ checks: layout(inv(layout(i))) == layout(i) for all i.
    This only holds for injective, contiguous layouts. For non-injective ones
    (broadcast, gapped), we just verify size and validity.

    Fast path: layouts with stride-0 modes are trivially non-injective,
    so we skip the expensive O(n) enumeration for them.
    """
    inv_layout = left_inverse(layout)

    # Fast path: stride-0 modes make injectivity impossible
    if _has_broadcast(layout):
        assert size(inv_layout) >= 1, f"left_inverse({layout}): empty result"
        return

    # No broadcast modes — check injectivity and contiguity via enumeration
    vals = set(layout(i) for i in range(size(layout)))
    is_inj = len(vals) == size(layout)
    is_contiguous = vals == set(range(size(layout)))
    if is_inj and is_contiguous:
        for i in range(size(layout)):
            li = layout(i)
            ili = inv_layout(li)
            lili = layout(ili)
            assert lili == li, (
                f"left_inverse({layout}): " f"L(inv(L({i})))={lili} != L({i})={li}"
            )
    else:
        assert size(inv_layout) >= 1, f"left_inverse({layout}): empty result"


def test_left_inverse_cpp_broadcast():
    """Left-inverse of broadcast layouts (C++ inverse_left.cpp)."""
    _test_left_inverse_cpp(Layout((3, 7), (0, 0)))
    _test_left_inverse_cpp(Layout(4, 0))
    _test_left_inverse_cpp(Layout((2, 4), (0, 2)))


def test_left_inverse_cpp_4d():
    """Left-inverse of 4D layout with stride-0 (C++ inverse_left.cpp)."""
    _test_left_inverse_cpp(Layout((2, 4, 4, 6), (4, 1, 0, 8)))


def test_left_inverse_cpp_coprime_gap():
    """Left-inverse with coprime stride gap (C++ inverse_left.cpp)."""
    _test_left_inverse_cpp(Layout((4, 2), (1, 5)))


def test_left_inverse_cpp_large():
    """Left-inverse of large layouts (C++ inverse_left.cpp)."""
    _test_left_inverse_cpp(Layout((128, 128), (65536, 1)))
    _test_left_inverse_cpp(Layout((128, 160), (65536, 1)))
    _test_left_inverse_cpp(Layout((128, 3, 160), (65536, 512, 1)))
    _test_left_inverse_cpp(Layout((128, 64), (131072, 2)))
    _test_left_inverse_cpp(Layout((32, 4, 4, 4), (262144, 4, 8388608, 1)))


def test_left_inverse_cpp_broadcast_middle():
    """Left-inverse with broadcast in middle position (C++ inverse_left.cpp)."""
    _test_left_inverse_cpp(Layout((2, 2, 2), (4, 0, 1)))


def test_left_inverse_cpp_deep_nested():
    """Left-inverse of deeply nested layout (C++ inverse_left.cpp line 210).

    Shape:  (((( 32, 4), 1), ( 32,  2)),        4), 1, (2,  2),  2)
    Stride: ((((262144, 4), 0), (  0,  1)), 8388608), 0, (2, 16), 32)
    """
    _test_left_inverse_cpp(
        Layout(
            ((((32, 4), 1), (32, 2)), 4, 1, (2, 2), 2),
            ((((262144, 4), 0), (0, 1)), 8388608, 0, (2, 16), 32),
        )
    )


## Right Inverse edge cases (C++ inverse_right.cpp)
# C++ test checks: layout(inv(i)) == i for all i in range(size(inv))


def _test_right_inverse_cpp(layout):
    """Test right_inverse using the C++ property.

    C++ checks: layout(inv(i)) == i for all i in range(size(inv))
    """
    inv_layout = right_inverse(layout)
    for i in range(size(inv_layout)):
        li = layout(inv_layout(i))
        assert li == i, f"right_inverse({layout}): " f"L(R({i}))={li} != {i}"


def test_right_inverse_cpp_4d():
    """Right-inverse of 4D layout with stride-0 (C++ inverse_right.cpp)."""
    _test_right_inverse_cpp(Layout((2, 4, 4, 6), (4, 1, 0, 8)))


def test_right_inverse_cpp_coprime_gap():
    """Right-inverse with coprime stride gap (C++ inverse_right.cpp)."""
    _test_right_inverse_cpp(Layout((4, 2), (1, 5)))


def test_right_inverse_cpp_large():
    """Right-inverse of large layouts (C++ inverse_right.cpp)."""
    _test_right_inverse_cpp(Layout((128, 128), (65536, 1)))
    _test_right_inverse_cpp(Layout((128, 160), (65536, 1)))
    _test_right_inverse_cpp(Layout((128, 3, 160), (65536, 512, 1)))
    _test_right_inverse_cpp(Layout((128, 64), (131072, 2)))
    _test_right_inverse_cpp(Layout((32, 4, 4, 4), (262144, 4, 8388608, 1)))


def test_right_inverse_cpp_broadcast_middle():
    """Right-inverse with broadcast in middle position (C++ inverse_right.cpp)."""
    _test_right_inverse_cpp(Layout((2, 2, 2), (4, 0, 1)))


def test_right_inverse_cpp_deep_nested():
    """Right-inverse of deeply nested layout (C++ inverse_right.cpp line 210)."""
    _test_right_inverse_cpp(
        Layout(
            ((((32, 4), 1), (32, 2)), 4, 1, (2, 2), 2),
            ((((262144, 4), 0), (0, 1)), 8388608, 0, (2, 16), 32),
        )
    )


## Composition edge case (C++ composition.cpp line 241-246)


def test_composition_transposed_strides():
    """Composition with same shape but transposed strides (C++ composition.cpp).

    Layout((4,3)) o Layout((4,3),(3,1)) -- col-major transposed.
    """
    _test_composition_properties(Layout((4, 3)), Layout((4, 3), (3, 1)))


## Complement edge case (pycute Python test_complement.py)


def test_complement_pycute_extra():
    """Complement case from pycute Python tests not in C++ tests."""
    _test_complement_disjointness(Layout((4, 10), (1, 10)))


###############################################################################
## Tests for newly added functions
## These cover tuple_max, elem_scale, crd2crd, has_none, slice_modes,
## slice_and_offset, hier_unzip, zipped_product,
## tiled_product
###############################################################################


## tuple_max


def test_tuple_max():
    """Test tuple_max function."""
    assert tuple_max(5) == 5
    assert tuple_max((3, 7, 2)) == 7
    assert tuple_max(((1, 9), (4, 2))) == 9
    assert tuple_max((1,)) == 1
    assert tuple_max(0) == 0


## elem_scale


def test_elem_scale():
    """Test elem_scale function."""
    # Scalar x scalar
    assert elem_scale(3, 4) == 12
    assert elem_scale(0, 5) == 0
    # Scalar x tuple -> scalar * product(tuple)
    assert elem_scale(2, (3, 4)) == 24
    assert elem_scale(1, (2, 3, 4)) == 24
    # Tuple x tuple -> pairwise
    assert elem_scale((2, 3), (4, 5)) == (8, 15)
    assert elem_scale((1, 1), (7, 8)) == (7, 8)
    # Tuple x scalar -> error
    import pytest

    with pytest.raises(TypeError):
        elem_scale((2, 3), 4)


## crd2crd


def test_crd2crd():
    """Test crd2crd function."""
    # int -> tuple: expand index into shape
    result = crd2crd(3, (2, 4))
    assert result == (1, 1)  # 3 = 1 + 1*2
    # tuple -> int: flatten coordinate using src_shape
    result = crd2crd((1, 0), 8, (2, 4))
    assert result == 1  # 1*1 + 0*2 = 1
    # tuple -> tuple: pairwise identity
    result = crd2crd((1, 2), (3, 4))
    assert result == (1, 2)
    # int -> int: identity
    assert crd2crd(5, 8) == 5


## has_none


def test_has_none():
    """Test has_none function."""
    assert has_none(None) is True
    assert has_none(3) is False
    assert has_none(0) is False
    assert has_none((1, None, 3)) is True
    assert has_none((1, 2, 3)) is False
    assert has_none((1, (2, None))) is True
    assert has_none((1, (2, 3))) is False
    assert has_none((1, 2)) is False


## slice_modes


def test_slice_modes():
    """Test slice_modes function."""
    # None keeps the element
    assert slice_modes(None, 4) == (4,)
    # Int drops the element
    assert slice_modes(0, 4) == ()
    # Tuple cases
    assert slice_modes((None, 0), (3, 4)) == (3,)
    assert slice_modes((0, None), (3, 4)) == (4,)
    assert slice_modes((None, None), (3, 4)) == (3, 4)
    assert slice_modes((0, 0), (3, 4)) == ()
    # Nested
    assert slice_modes(((None, 0), None), ((3, 4), 5)) == (3, 5)


## Layout.__call__ with None (slicing)


def test_layout_slice():
    """Test Layout.__call__ with None coordinates for slicing."""
    layout = Layout((4, 8), (1, 4))

    # Slice: keep dim 0, fix dim 1
    # Result preserves tuple structure (matching pycute behavior)
    sub = layout(None, 0)
    assert sub.shape == (4,)
    assert sub.stride == (1,)

    # Slice: fix dim 0, keep dim 1
    sub = layout(0, None)
    assert sub.shape == (8,)
    assert sub.stride == (4,)

    # Keep both (identity slice)
    sub = layout(None, None)
    assert size(sub) == size(layout)

    # Verify sublayout indexing: use slice_and_offset for offset-aware check
    layout_3d = Layout((2, 3, 4), (1, 2, 6))
    sub, offset = slice_and_offset((None, 1, None), layout_3d)
    # sub should be a layout over dims 0 and 2
    for i in range(2):
        for k in range(4):
            assert sub(i, k) + offset == layout_3d(i, 1, k)


## slice_and_offset


def test_slice_and_offset():
    """Test slice_and_offset function."""
    layout = Layout((4, 8), (1, 4))

    # Slice dim 0, fix dim 1 to 3
    sub, offset = slice_and_offset((None, 3), layout)
    assert size(sub) == 4
    assert offset == 12  # 3 * 4

    # Fix dim 0 to 2, slice dim 1
    sub, offset = slice_and_offset((2, None), layout)
    assert size(sub) == 8
    assert offset == 2  # 2 * 1

    # Verify: sublayout(i) + offset == original(fixed_coord, i)
    for i in range(size(sub)):
        assert sub(i) + offset == layout(
            2, i
        ), f"i={i}: {sub(i) + offset} != {layout(2, i)}"


## zipped_product


def test_zipped_product():
    """Test zipped_product function."""
    # Basic case
    result = zipped_product(Layout(4, 1), Layout(3, 1))
    assert rank(result) == 2
    # First mode should be layout_a
    r0 = mode(result, 0)
    assert r0.shape == 4 and r0.stride == 1

    # Multi-mode
    A = Layout((4, 8), (1, 4))
    B = (2, 2)
    result = zipped_product(A, B)
    assert rank(result) == 2
    # Verify size is correct: size(A) * size(B)
    assert size(result) == size(A) * 4  # B has size 2*2=4


## tiled_product


def test_tiled_product():
    """Test tiled_product function."""
    result = tiled_product(Layout(4, 1), Layout(3, 1))
    # Should have rank >= 2: ((A-modes), rest0, rest1, ...)
    assert rank(result) >= 2
    # size should equal size(A) * size(B)
    assert size(result) == 12

    # Multi-mode
    A = Layout((4, 8), (1, 4))
    B = (2, 2)
    result = tiled_product(A, B)
    assert size(result) == size(A) * 4


## hier_unzip


def test_hier_unzip():
    """Test hier_unzip with logical_divide."""
    # Basic: same as zipped_divide
    A = Layout((4, 8), (1, 4))
    B = (2, 4)
    result_zip = zipped_divide(A, B)
    result_hier = hier_unzip(logical_divide, A, B)
    # They should produce the same result
    assert size(result_zip) == size(result_hier)
    for i in range(size(result_zip)):
        assert result_zip(i) == result_hier(i), (
            f"zipped_divide vs hier_unzip mismatch at {i}: "
            f"{result_zip(i)} != {result_hier(i)}"
        )

    # None case
    result = hier_unzip(logical_divide, Layout(4, 1), None)
    assert rank(result) == 2


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
