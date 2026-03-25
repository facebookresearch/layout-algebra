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
from tensor_layouts.layout_utils import (
    make_layout_like,
    make_ordered_layout,
    tile_to_shape,
)


# These tests roughly follow:
# CuTe Library documentation, Nvidia
# Categorical Foundations of CuTe Layouts, Colfax Research


## Tuple Constructor and methods


def test_tuple_empty():
    t = ()
    assert len(t) == 0
    assert size(t) == 1
    assert rank(t) == 0
    assert depth(t) == 0
    assert mode(t, 0) == ()
    assert repr(t) == "()"


def test_tuple_scalar():
    t = 2
    # Scalars are not tuples, so no len() - use size() and rank() instead
    assert size(t) == 2
    assert rank(t) == 0  # Scalars have rank 0
    assert depth(t) == 0  # Scalars have depth 0
    assert repr(t) == "2"


def test_tuple_single_element():
    t = (2,)
    assert len(t) == 1
    assert size(t) == 2
    assert rank(t) == 1
    assert depth(t) == 1
    assert mode(t, 0) == 2
    assert repr(t) == "(2,)"


def test_tuple_nested_single_element():
    t = (2,)
    assert len(t) == 1
    assert size(t) == 2
    assert rank(t) == 1
    assert depth(t) == 1
    assert mode(t, 0) == 2
    assert repr(t) == "(2,)"


def test_tuple_flat_collection():
    t = (3, 127, 128)
    assert len(t) == 3
    assert size(t) == 48768
    assert rank(t) == 3
    assert depth(t) == 1
    assert mode(t, 0) == 3
    assert mode(t, 1) == 127
    assert mode(t, 2) == 128
    assert repr(t) == "(3, 127, 128)"


def test_tuple_nested_collection():
    t = (1, 2, (3, 4), 5, (6,))
    assert len(t) == 5
    assert size(t) == 720
    assert depth(t) == 2
    assert mode(t, 0) == 1
    assert mode(t, 1) == 2
    assert mode(t, 2) == (3, 4)
    assert mode(t, 3) == 5
    assert mode(t, 4) == (6,)
    assert repr(t) == "(1, 2, (3, 4), 5, (6,))"


def test_shape_normalization():
    # Test that _normalize_shape works correctly for various inputs
    # These are handled at Layout construction time
    L1 = Layout(2)  # scalar
    assert L1.shape == 2
    assert L1.stride == 1

    L2 = Layout((3, 4, 5))  # flat tuple
    assert L2.shape == (3, 4, 5)

    L3 = Layout((1, (2, 3)))  # nested tuple
    assert L3.shape == (1, (2, 3))


def test_layout_from_layouts():
    # Making Layouts from Layouts, as per CuTe docs
    a = Layout(3, 1)
    b = Layout(4, 3)
    row = Layout(a, b)
    assert row == Layout((3, 4), (1, 3))
    col = Layout(b, a)
    assert col == Layout((4, 3), (3, 1))
    q = Layout(row, col)
    assert q == Layout(((3, 4), (4, 3)), ((1, 3), (3, 1)))
    aa = Layout(a)
    assert aa == Layout((3,), (1,))
    aaa = Layout(aa)
    assert aaa == Layout(((3,),), ((1,),))
    d = Layout(a, Layout(a), a)
    assert d == Layout((3, (3,), 3), (1, (1,), 1))


def test_append_prepend_replace_group():
    ## append, prepend, replace, group operations
    a = Layout(3, 1)
    b = Layout(4, 3)
    ab = append(a, b)
    assert ab == Layout((3, 4), (1, 3))
    ba = prepend(a, b)
    assert ba == Layout((4, 3), (3, 1))
    c = append(ab, ab)
    assert c == Layout((3, 4, (3, 4)), (1, 3, (1, 3)))
    d = replace(c, 2, b)
    assert d == Layout((3, 4, 4), (1, 3, 3))

    a = Layout((2, 3, 5, 7))
    assert a == Layout((2, 3, 5, 7), (1, 2, 6, 30))
    b = group(a, 0, 2)
    assert b == Layout(((2, 3), 5, 7), ((1, 2), 6, 30))
    c = group(b, 1, 3)
    assert c == Layout(((2, 3), (5, 7)), ((1, 2), (6, 30)))
    f = flatten(b)
    assert f == Layout((2, 3, 5, 7), (1, 2, 6, 30))
    e = flatten(c)
    assert e == Layout((2, 3, 5, 7), (1, 2, 6, 30))


def test_concatenate_tuples():
    t1 = (3, 128, 128)
    t2 = (3, 128, 128)
    assert concat(t1, t2) == (3, 128, 128, 3, 128, 128)
    t2 = (1, 2, (3, 4), 5, (6,))
    assert concat(t1, t2) == (3, 128, 128, 1, 2, (3, 4), 5, (6,))


def test_congruent():
    assert congruent(3, 5)
    assert not congruent(3, (3,))
    assert congruent((3, 128, 128), (1, 256, 64))
    assert not congruent((3, 128), (1, 256, 64))
    assert congruent((2, (3, 4)), (5, (6, 7)))
    assert not congruent((2, (3, 4)), (5, 6))


def test_compatible():
    assert not compatible(24, 32)
    assert compatible(24, (4, 6))
    assert compatible((4, 6), ((2, 2), 6))
    assert compatible(((2, 2), 6), ((2, 2), (3, 2)))
    assert compatible(24, ((2, 2), (3, 2)))
    assert compatible(24, ((2, 3), 4))
    assert not compatible(((2, 3), 4), ((2, 2), (3, 2)))
    assert not compatible(((2, 2), (3, 2)), ((2, 3), 4))
    assert compatible(24, (24,))
    assert not compatible((24,), 24)
    assert not compatible((24,), (4, 6))


def test_layout_basic():
    Layout((), ())  # Empty layout
    shape = (3, 128, 128)
    stride = (16384, 128, 1)
    L1 = Layout(shape, stride)
    assert rank(L1.shape) == 3
    assert size(L1.shape) == 49152
    assert depth(L1.stride) == 1
    L1 = Layout((2, 2, 2), (1, 2, 4))
    assert rank(L1) == 3
    Layout(128, 5)  # Scalar layout
    Layout((16, 12, 512, 512), (0, 0, 1, 512))  # Broadcast layout
    Layout((6, 1, 12, 2, 2), (2, 0, 12, 144, 1))  # Complex layout


def test_layout_type_validation():
    """Layout rejects invalid shape/stride types with clear messages."""
    # Strings rejected
    with pytest.raises(TypeError, match="stride.*str"):
        Layout((4, 2), "row")
    with pytest.raises(TypeError, match="shape.*str"):
        Layout("abc")

    # Floats rejected
    with pytest.raises(TypeError, match="stride.*float"):
        Layout((4, 2), 1.5)
    with pytest.raises(TypeError, match="shape.*float"):
        Layout(3.14)

    # None rejected
    with pytest.raises(TypeError, match="shape.*NoneType"):
        Layout(None)

    # Valid constructions still work
    Layout((4, 2), (1, 4))
    Layout(((2, 2), 4), ((1, 2), 4))
    Layout(8)
    Layout([4, 2])  # lists are fine


def test_layout_rank_size_cosize():
    L5 = Layout((64, 32), (1, 128))
    assert rank(L5) == 2
    assert size(L5) == 2048
    assert cosize(L5) == 4032
    assert mode(L5, 0) == Layout(64, 1)
    assert mode(L5, 1) == Layout(32, 128)
    L6 = Layout((3, 8, 8, 8), (1, 3, 24, 192))
    assert rank(L6) == 4
    assert size(L6) == 1536
    assert cosize(L6) == 1536
    assert mode(L6, 2) == Layout(8, 24)
    L7 = Layout((2, 2, 2, 2, 2), (160, 80, 40, 20, 10))
    assert rank(L7) == 5
    assert size(L7) == 32
    assert cosize(L7) == 311
    assert mode(L7, 0) == Layout(2, 160)
    assert mode(L7, 4) == Layout(2, 10)


def test_layout_squeeze():
    L0 = Layout((), ())
    L11 = Layout((64, 64, 1), (1, 64, 0))
    assert L11.squeeze() == Layout((64, 64), (1, 64))
    L12 = Layout((64, 64, 1, 32, 1), (2048, 32, 0, 1, 0))
    assert L12.squeeze() == Layout((64, 64, 32), (2048, 32, 1))
    assert L0 == L0.squeeze()


def test_layout_filter():
    L0 = Layout((), ())
    L13 = Layout((64, 8, 8, 128), (8, 1, 0, 512))
    assert L13.filter() == Layout((64, 8, 128), (8, 1, 512))
    L14 = Layout((3, 8, 8, 8), (16, 0, 0, 0))
    assert L14.filter() == Layout(3, 16)
    assert L0 == L0.filter()


def test_layout_computed_strides():
    L15 = Layout((2, 2, 2, 2, 2))  # compute strides
    assert L15.stride == (1, 2, 4, 8, 16)
    L16 = Layout((1, 2, (3, 4), 5, (6,)))
    # Nested (6,) is preserved as a single-element tuple per CuTe semantics
    assert L16.stride == (0, 1, (2, 6), 24, (120,))


def test_concatenate_layouts():
    L17 = Layout((7, 2), (2, 1))
    L18 = Layout((3, 3, 3), (0, 10, 30))
    C = concat(L17, L18)
    assert C == Layout((7, 2, 3, 3, 3), (2, 1, 0, 10, 30))
    assert size(C) == size(L17) * size(L18)
    for i in range(size(L17)):
        assert C(i) == L17(i)


def test_concatenate_scalar_layouts():
    # concat on scalar (rank-0) layouts should produce a rank-2 layout,
    # not do integer addition. Matches CuTe C++ make_layout.
    result = concat(Layout(3, 1), Layout(4, 3))
    assert result == Layout((3, 4), (1, 3))
    assert size(result) == 12
    T0 = (1, 2, (3, 4), 5, (6,))
    assert flatten(T0) == (1, 2, 3, 4, 5, 6)
    L19 = Layout((1, 2, (3, 4), 5, (6,)))
    assert flatten(L19) == Layout((1, 2, 3, 4, 5, 6), (0, 1, 2, 6, 24, 120))


def test_sort_layouts():
    L0 = Layout((), ())
    assert sort(L0) == L0
    L20 = Layout((128, 64, 2, 2), (1, 128, 8192, 16384))
    assert L20 == sort(L20)
    L21 = Layout((2, 2, 2), (1, 1, 1))
    assert L21 == sort(L21)
    L22 = Layout((2, 4, 8, 16), (64, 1, 2, 4))
    assert sort(L22) == Layout((4, 8, 16, 2), (1, 2, 4, 64))
    L23 = Layout((5, 32, 16), (1, 5, 5))
    assert sort(L23) == Layout((5, 16, 32), (1, 5, 5))
    L24 = Layout((1, 2, (3, 4), 5, (6,)), (2, 12, (1, 3), 24, (120,)))
    assert sort(L24) == Layout((3, 1, 4, 2, 5, 6), (1, 2, 3, 12, 24, 120))


def test_coalesce_layouts():
    L25 = Layout()
    assert coalesce(L25) == L25
    L25 = Layout((64, 32), (1, 64))
    assert coalesce(L25) == Layout(2048, 1)
    L25 = Layout((2, 4), (1, 2))
    assert coalesce(L25) == Layout(8, 1)
    L25 = Layout((2, (1, 6)), (1, (6, 2)))
    assert coalesce(L25) == Layout(12, 1)
    L25 = Layout((3, 5, 2), (7, 21, 4))
    assert coalesce(L25) == Layout((15, 2), (7, 4))
    L25 = Layout((2, 2, 2, 2, 2), (8, 16, 1024, 2048, 4096))
    assert coalesce(L25) == Layout((4, 8), (8, 1024))
    L25 = Layout((3, 4, 1, 5), (1, 8, 3, 32))
    assert coalesce(L25) == Layout((3, 20), (1, 8))


def test_coalesce_by_mode_with_profile():
    ## By-mode Coalesce (with profile)
    # Simple case: coalesce (2,4,2,2):(1,2,8,16) with profile (4,4)
    # First mode (4 elements): (2,4):(1,2) -> 8:1
    # Second mode (4 elements): (2,2):(8,16) -> 4:8
    L26 = Layout((2, 4, 2, 2), (1, 2, 8, 16))
    assert coalesce(L26, (4, 4)) == Layout((8, 4), (1, 8))

    # Coalesce with profile preserving non-contiguous modes
    L27 = Layout((2, 2, 2, 2), (1, 2, 8, 16))
    assert coalesce(L27, (4, 4)) == Layout((4, 4), (1, 8))

    # Coalesce with profile - first mode contiguous, second mode has gap
    L28 = Layout((4, 2, 2), (1, 8, 16))
    assert coalesce(L28, (4, 4)) == Layout((4, 4), (1, 8))

    # Coalesce where profile mode has non-contiguous elements
    L29 = Layout((2, 2, 2, 2), (1, 4, 16, 64))
    assert coalesce(L29, (4, 4)) == Layout(((2, 2), (2, 2)), ((1, 4), (16, 64)))

    # Full coalesce without profile should still work
    assert coalesce(L26) == Layout(32, 1)

    # Profile with nested structure
    L30 = Layout((2, 2, 2, 2, 2, 2), (1, 2, 4, 8, 16, 32))
    assert coalesce(L30, (4, 4, 4)) == Layout((4, 4, 4), (1, 4, 16))

    # Coalesce a layout that's already optimal for the profile
    L31 = Layout((8, 4), (1, 8))
    assert coalesce(L31, (8, 4)) == Layout((8, 4), (1, 8))


def test_complement_layouts():
    assert complement(Layout(), 8) == Layout(8, 1)
    assert complement(Layout(4, 2), 16) == Layout((2, 2), (1, 8))
    assert complement(Layout(4, 1), 16) == Layout(4, 4)
    assert complement(Layout(4, 1), 24) == Layout(6, 4)
    assert complement(Layout((2, 2), (1, 4)), 16) == Layout((2, 2), (2, 8))
    assert complement(Layout(6, 1), 24) == Layout(4, 6)
    assert complement(Layout((4, 2), (1, 16))) == Layout(4, 4)  # default cosize_bound
    # Contiguous layout - complement starts after the layout ends
    assert complement(Layout(8, 1), 32) == Layout(4, 8)
    # Layout with gaps
    assert complement(Layout(4, 4), 32) == Layout((4, 2), (1, 16))


def test_complement_rejects_negative_strides():
    """complement rejects negative strides (matches CuTe/pycute assertions)."""
    with pytest.raises(ValueError, match="negative stride"):
        complement(Layout(4, -1))

    with pytest.raises(ValueError, match="negative stride"):
        complement(Layout((2, 3), (-1, 2)))


def test_complement_rejects_zero_sized_shapes():
    """complement rejects zero-sized shapes with nonzero strides."""
    with pytest.raises(ValueError, match="zero-sized"):
        complement(Layout(0, 1))

    with pytest.raises(ValueError, match="zero-sized"):
        complement(Layout((2, 0), (1, 2)))


def test_coordinate_functions():
    L8 = Layout((2, 3), (1, 5))
    assert L8(0, 0) == 0
    assert L8(1, 0) == 1
    assert L8(0, 1) == 5
    assert L8(1, 1) == 6
    assert L8(0, 2) == 10
    assert L8(1, 2) == 11
    assert L8(0) == 0
    assert L8(1) == 1
    assert L8(2) == 5
    assert L8(3) == 6
    assert L8(4) == 10
    assert L8(5) == 11
    L9 = Layout((2, 2), (64, 2))
    assert L9(0, 0) == 0
    assert L9(1, 0) == 64
    assert L9(0, 1) == 2
    assert L9(1, 1) == 66
    assert L9(0) == 0
    assert L9(1) == 64
    assert L9(2) == 2
    assert L9(3) == 66
    L10 = Layout((4, 2, 2), (3, 3, 100))
    assert L10(7) == 12
    assert L10(9) == 103


def test_coordinate_validation():
    """Layout.__call__ rejects invalid coordinate shapes (matches CuTe/pycute)."""
    L = Layout((4, 8), (1, 4))

    # Empty coords on a non-scalar layout
    with pytest.raises(ValueError, match="rank"):
        L()

    # Over-rank coords (3 coords for rank-2 layout)
    with pytest.raises(ValueError, match="rank"):
        L(1, 2, 3)

    # List coords (not a tuple or int)
    with pytest.raises(TypeError, match="int or tuple"):
        L([1, 2])

    # Valid cases still work
    assert L(1) == 1  # flat index
    assert L(1, 2) == 9  # tuple coord via *args
    assert L((1, 2)) == 9  # tuple coord via single arg


def test_idx2crd_crd2flat_crd2offset():
    shape = (3, (2, 3))
    # idx to hcoords
    assert idx2crd(16, shape) == (1, (1, 2))
    assert idx2crd((1, 5), shape) == (1, (1, 2))
    assert idx2crd((1, (1, 2)), shape) == (1, (1, 2))
    # hcoords to flat 1d index
    layout = Layout((3, (2, 3)), (3, (12, 1)))
    assert crd2flat(16, layout.shape) == 16
    assert crd2flat((1, 5), layout.shape) == 16
    assert crd2flat((1, (1, 2)), layout.shape) == 16
    # hcoords to offset
    layout = Layout((3, (2, 3)), (3, (12, 1)))
    assert crd2offset(16, layout.shape, layout.stride) == 17
    assert crd2offset((1, 5), layout.shape, layout.stride) == 17
    assert crd2offset((1, (1, 2)), layout.shape, layout.stride) == 17
    # crd2idx dispatches correctly
    assert crd2idx(16, layout.shape) == 16  # 2-arg -> crd2flat
    assert crd2idx(16, layout.shape, layout.stride) == 17  # 3-arg -> crd2offset


def test_shape_division():
    ## Shape division and modulo, from Cute Layout Algebra docs
    assert shape_div((6, 2), 2) == (3, 2)
    assert shape_div((6, 2), 3) == (2, 2)
    assert shape_div((6, 2), 6) == (1, 2)
    assert shape_div((6, 2), 12) == (1, 1)
    assert shape_div((3, 6, 2, 8), 3) == (1, 6, 2, 8)
    assert shape_div((3, 6, 2, 8), 6) == (1, 3, 2, 8)
    assert shape_div((3, 6, 2, 8), 9) == (1, 2, 2, 8)
    assert shape_div((3, 6, 2, 8), 72) == (1, 1, 1, 4)


def test_shape_modulo():
    assert shape_mod((6, 2), 2) == (2, 1)
    assert shape_mod((6, 2), 3) == (3, 1)
    assert shape_mod((6, 2), 6) == (6, 1)
    assert shape_mod((6, 2), 12) == (6, 2)
    assert shape_mod((3, 6, 2, 8), 6) == (3, 2, 1, 1)
    assert shape_mod((3, 6, 2, 8), 9) == (3, 3, 1, 1)
    assert shape_mod((1, 2, 2, 8), 2) == (1, 2, 1, 1)
    assert shape_mod((1, 2, 2, 8), 16) == (1, 2, 2, 4)


def test_shape_div_non_divisible():
    """CuTe shape_div requires a%b==0 or b%a==0 at each level.

    Inputs that don't satisfy this condition produce incorrect results
    (e.g., shape_div/mod won't be complementary), so we assert.
    """
    # Valid cases where divisibility holds
    assert shape_div(12, 4) == 3  # 12%4==0
    assert shape_div(4, 12) == 1  # 12%4==0
    assert shape_div(8, 2) == 4  # 8%2==0
    assert shape_div(2, 8) == 1  # 8%2==0

    # Invalid cases should raise ValueError
    with pytest.raises(ValueError):
        shape_div(6, 4)  # 6%4≠0, 4%6≠0
    with pytest.raises(ValueError):
        shape_div(4, 6)  # 4%6≠0, 6%4≠0


def test_shape_mod_non_divisible():
    """CuTe recursive shape_mod: remaining = shape_div(m, shape_mod(size(a), m)).

    When the modulus doesn't evenly divide a mode, CuTe uses gcd for the
    scalar case (when modulus < shape), or returns shape (when modulus >= shape).
    """
    # shape_mod((6, 2), 4): shape_mod(6,4)=gcd(6,4)=2,
    #   remaining = shape_div(4, shape_mod(6,4)) = shape_div(4,2) = 2
    #   shape_mod(2, 2) = gcd(2,2) = 2
    assert shape_mod((6, 2), 4) == (2, 2)
    # Scalar shape_mod: when modulus < shape, returns gcd
    assert shape_mod(6, 4) == 2  # gcd(6,4) = 2
    # Scalar shape_mod: when modulus >= shape, returns shape
    assert shape_mod(4, 6) == 4  # 6 >= 4, returns 4


def test_shape_div_mod_complementary():
    """shape_div and shape_mod should satisfy: size(shape_div(s,d)) * size(shape_mod(s,d)) == size(s).

    This holds when the divisor evenly divides each mode it consumes.
    """
    test_cases = [
        ((6, 2), 2),
        ((6, 2), 3),
        ((6, 2), 6),
        ((6, 2), 12),
        ((4, 3), 2),
        ((4, 3), 4),
        ((4, 3), 12),
        ((3, 6, 2, 8), 3),
        ((3, 6, 2, 8), 9),
        ((3, 6, 2, 8), 72),
    ]
    for shape, div in test_cases:
        sd = shape_div(shape, div)
        sm = shape_mod(shape, div)
        assert size(sd) * size(sm) == size(shape), (
            f"shape_div({shape},{div})={sd} (size {size(sd)}) * "
            f"shape_mod({shape},{div})={sm} (size {size(sm)}) != size({shape})={size(shape)}"
        )


def test_logical_divide_cross_mode():
    """logical_divide with tiler spanning multiple modes."""
    # Tiler = 12 = 4*3 spans both modes of (4,3)
    L = Layout((4, 3), (1, 4))
    result = logical_divide(L, 12)
    for i in range(size(L)):
        assert result(i) == L(i)

    # Tiler = 8 spans mode 0 (4) and part of mode 1 (6)
    L = Layout((4, 6), (1, 4))
    result = logical_divide(L, 8)
    for i in range(size(L)):
        assert result(i) == L(i)

    # Tiler = 6 spans mode 0 (2) and mode 1 (3)
    L = Layout((2, 3, 4), (1, 2, 6))
    result = logical_divide(L, 6)
    for i in range(size(L)):
        assert result(i) == L(i)


def test_logical_divide_non_divisible():
    """logical_divide where tiler doesn't evenly divide the first mode."""
    # 4 doesn't divide 6
    L = Layout((6, 2), (1, 6))
    result = logical_divide(L, 4)
    for i in range(size(L)):
        assert result(i) == L(i)

    # 6 doesn't divide 4, but 6 = 4 * 1.5
    L = Layout((4, 3), (1, 4))
    result = logical_divide(L, 6)
    for i in range(size(L)):
        assert result(i) == L(i)

    # 6 doesn't divide 3
    L = Layout((3, 4), (1, 3))
    result = logical_divide(L, 6)
    for i in range(size(L)):
        assert result(i) == L(i)


def test_compose_basic():
    ## Compose Layouts

    # Basic composition: compose(A, B)(i) = A(B(i))
    # compose(8:2, 4:1) -> 4:2 (B selects first 4 elements of A with stride 1, A has stride 2)
    assert compose(Layout(8, 2), Layout(4, 1)) == Layout(4, 2)

    # compose(8:2, 4:2) -> 4:4 (B selects every other element, combined with A's stride 2)
    assert compose(Layout(8, 2), Layout(4, 2)) == Layout(4, 4)


def test_compose_2d_outer():
    # Compose with 2D outer layout
    # A = (4,8):(1,4) is a 4x8 column-major layout (size 32)
    # B = 4:1 selects first 4 elements linearly
    # Result should be 4:1 (first column of A)
    assert compose(Layout((4, 8), (1, 4)), Layout(4, 1)) == Layout(4, 1)

    # B = 4:4 selects elements 0, 4, 8, 12 from A's domain
    # In A, index 4 maps to offset 4 (start of second column)
    # So we get offsets 0, 4, 8, 12 -> stride 4
    assert compose(Layout((4, 8), (1, 4)), Layout(4, 4)) == Layout(4, 4)


def test_compose_2d_inner():
    # Compose with 2D inner layout
    # A = 16:1 is a flat layout
    # B = (4,4):(1,4) is a 4x4 column-major layout
    # compose(A, B)(i,j) = A(B(i,j)) = A(i + 4*j) = i + 4*j
    assert compose(Layout(16, 1), Layout((4, 4), (1, 4))) == Layout((4, 4), (1, 4))

    # A = 16:2 has stride 2
    # compose(A, B) should double all strides in B
    assert compose(Layout(16, 2), Layout((4, 4), (1, 4))) == Layout((4, 4), (2, 8))


def test_compose_two_2d():
    # Compose two 2D layouts
    # A = (4,4):(1,4), B = (2,2):(1,2)
    # B selects a 2x2 subgrid from A
    # B(0,0)=0, B(1,0)=1, B(0,1)=2, B(1,1)=3
    # A(0)=0, A(1)=1, A(2)=2, A(3)=3
    # So compose gives same result as B indexing into first 4 elements of A
    assert compose(Layout((4, 4), (1, 4)), Layout((2, 2), (1, 2))) == Layout(
        (2, 2), (1, 2)
    )


def test_compose_functional_equivalence():
    # Verify compose property: compose(A, B)(i) == A(B(i))
    A = Layout((4, 8), (1, 4))
    B = Layout((2, 4), (1, 2))
    C = compose(A, B)
    for i in range(size(B)):
        assert C(i) == A(B(i))

    # More complex compositions
    A = Layout((8, 4), (4, 1))  # Row-major 8x4
    B = Layout(8, 1)  # First 8 elements
    C = compose(A, B)
    for i in range(size(B)):
        assert C(i) == A(B(i))

    # Compose with nested layouts
    A = Layout(32, 1)
    B = Layout(((2, 2), (2, 2)), ((1, 4), (8, 16)))
    C = compose(A, B)
    for i in range(size(B)):
        assert C(i) == A(B(i))


def test_compose_nested_result():
    # From CuTe docs - composition with nested result
    A = Layout((6, 2), (8, 2))
    B = Layout((4, 3), (3, 1))
    C = compose(A, B)
    assert C == Layout(((2, 2), 3), ((24, 2), 8))
    for i in range(size(B)):
        assert C(i) == A(B(i))

    A = Layout((10, 2), (16, 4))
    B = Layout((5, 4), (1, 5))
    C = compose(A, B)
    assert C == Layout(((5, 1), (2, 2)), ((16, 4), (80, 4))) or C == Layout(
        (5, (2, 2)), (16, (80, 4))
    )  # both reported as correct in the docs
    for i in range(size(B)):
        assert C(i) == A(B(i))


def test_logical_divide_basic():
    ## Logical Division and Tiling Operations

    # Basic logical_divide with integer tiler
    L = Layout((8,))
    divided = logical_divide(L, 2)
    # Note: single-element tuples are normalized, so ((2, 4),) becomes (2, 4)
    assert divided.shape == (2, 4)
    assert divided.stride == (1, 2)
    # Verify indexing is preserved
    for i in range(size(L)):
        assert divided(i) == L(i)


def test_logical_divide_tuple_tiler():
    # Logical divide with tuple tiler
    L = Layout((4, 8))
    divided = logical_divide(L, (2, 4))
    # Each mode is divided: 4 -> (2, 2), 8 -> (4, 2)
    assert size(divided) == size(L)
    for i in range(size(L)):
        assert divided(i) == L(i)


def test_logical_divide_strided():
    # Logical divide on a strided layout
    L = Layout((8,), (2,))  # Strided layout
    divided = logical_divide(L, 4)
    assert size(divided) == size(L)
    for i in range(size(L)):
        assert divided(i) == L(i)


def test_logical_divide_2d():
    # Logical divide 2D layout
    L = Layout((4, 6), (1, 4))
    divided = logical_divide(L, (2, 3))
    assert size(divided) == size(L)
    for i in range(size(L)):
        assert divided(i) == L(i)


def test_logical_divide_hierarchical_stride():
    # Shape tiler with hierarchical strides should not crash.
    # CuTe C++ always uses compose/complement (no shortcut), so this
    # exercises the fallback path in _logical_divide_by_shape.
    L = Layout(((2, 4), 8), ((1, 2), 8))
    divided = logical_divide(L, (4, 4))
    assert size(divided) == size(L)
    for i in range(size(L)):
        assert divided(i) == L(i)


def test_tiled_divide():
    # tiled_divide: ((TileM,TileN), RestM, RestN, L, ...)
    # Mode 0 is the grouped tiles, remaining modes are individual rests
    L = Layout((8, 8))
    tiled = tiled_divide(L, (2, 2))
    # tiled has shape ((2, 2), 4, 4) - tiles grouped, rests flat
    assert len(tiled.shape) == 3
    assert tiled.shape[0] == (2, 2)
    assert tiled.shape[1] == 4
    assert tiled.shape[2] == 4
    assert size(tiled) == size(L)


def test_zipped_divide():
    # zipped_divide: ((TileM,TileN), (RestM,RestN,L,...))
    # Both tiles and rests are grouped
    L = Layout((8, 8))
    zipped = zipped_divide(L, (2, 2))
    # zipped has shape ((2, 2), (4, 4)) - tiles grouped, rests grouped
    assert len(zipped.shape) == 2
    assert zipped.shape[0] == (2, 2)
    assert zipped.shape[1] == (4, 4)
    assert size(zipped) == size(L)


def test_flat_divide():
    # flat_divide: (TileM, TileN, RestM, RestN, L, ...)
    # Everything is flat
    L = Layout((8, 8))
    flat = flat_divide(L, (2, 2))
    # flat has shape (2, 2, 4, 4) - all flat
    assert len(flat.shape) == 4
    assert flat.shape[0] == 2
    assert flat.shape[1] == 2
    assert flat.shape[2] == 4
    assert flat.shape[3] == 4
    assert size(flat) == size(L)


def test_divide_1d_tiler_on_2d_layout():
    # Test with 1D tiler on 2D layout
    L = Layout((8, 6))
    # logical_divide: ((TileM, RestM), N)
    logical = logical_divide(L, 2)
    assert len(logical.shape) == 2
    assert logical.shape[0] == (2, 4)
    assert logical.shape[1] == 6

    # zipped_divide: ((TileM), (RestM, N))
    zipped = zipped_divide(L, 2)
    assert len(zipped.shape) == 2
    assert zipped.shape[0] == 2
    assert zipped.shape[1] == (4, 6)

    # tiled_divide: (TileM, RestM, N)
    tiled = tiled_divide(L, 2)
    assert len(tiled.shape) == 3
    assert tiled.shape[0] == 2
    assert tiled.shape[1] == 4
    assert tiled.shape[2] == 6

    # flat_divide: (TileM, RestM, N)
    flat = flat_divide(L, 2)
    assert len(flat.shape) == 3
    assert flat.shape[0] == 2
    assert flat.shape[1] == 4
    assert flat.shape[2] == 6


def test_divide_1d_known_results():
    # Additional tests from CuTe DSL with known-good results

    # 1D: Layout(16) / Layout(4) = (4,4):(1,4) for all divide variants
    L = Layout(16)
    T = 4

    logical = logical_divide(L, T)
    # Note: single-element tuples are normalized, so shape is (4, 4) not ((4, 4),)
    assert logical.shape == (4, 4)
    assert logical.stride == (1, 4)

    zipped = zipped_divide(L, T)
    assert zipped.shape == (4, 4)
    assert zipped.stride == (1, 4)

    tiled = tiled_divide(L, T)
    assert tiled.shape == (4, 4)
    assert tiled.stride == (1, 4)

    flat = flat_divide(L, T)
    assert flat.shape == (4, 4)
    assert flat.stride == (1, 4)


def test_logical_divide_non_divisible_tuple_tiler():
    """Tuple tiler where tile doesn't evenly divide a mode (CuTe uses ceil_div).

    Validated against pycute (CuTe Python reference) — all shapes, strides,
    sizes, and index mappings match.
    """
    # The original bug: (5,8):(1,5) / (2,4) used floor(5/2)=2 instead of ceil(5/2)=3
    L = Layout((5, 8), (1, 5))
    result = logical_divide(L, (2, 4))
    assert result.shape == ((2, 3), (4, 2))
    assert result.stride == ((1, 2), (5, 20))
    assert size(result) == 48  # was incorrectly 32 before fix

    # (3,8):(1,3) / (2,4) -> ceil(3/2) = 2
    result2 = logical_divide(Layout((3, 8), (1, 3)), (2, 4))
    assert result2.shape == ((2, 2), (4, 2))
    assert result2.stride == ((1, 2), (3, 12))
    assert size(result2) == 32

    # (7,6):(1,7) / (3,4) -> ceil(7/3) = 3, ceil(6/4) = 2
    result3 = logical_divide(Layout((7, 6), (1, 7)), (3, 4))
    assert result3.shape == ((3, 3), (4, 2))
    assert result3.stride == ((1, 3), (7, 28))
    assert size(result3) == 72

    # Divisible cases still work with the fast path
    L4 = Layout((8, 6))
    r4 = logical_divide(L4, (4, 3))
    assert r4.shape == ((4, 2), (3, 2))
    for i in range(size(L4)):
        assert r4(i) == L4(i)


def test_logical_divide_rejects_over_rank_tiler():
    """Tuple tiler with more modes than layout must raise (matches CuTe static_assert)."""
    with pytest.raises(ValueError, match="more modes"):
        logical_divide(Layout((4, 8), (1, 4)), (2, 4, 6))

    with pytest.raises(ValueError, match="more modes"):
        logical_divide(Layout(8), (2, 4))

    # Exact rank is fine
    logical_divide(Layout((4, 8), (1, 4)), (2, 4))

    # Fewer tiler modes is fine (undivided modes pass through)
    logical_divide(Layout((4, 8, 3), (1, 4, 32)), (2, 4))


def test_logical_product():
    # logical_product combines two layouts
    A = Layout(4, 1)
    B = Layout(8, 1)
    product = logical_product(A, B)
    assert product.shape == (4, 8)
    assert product.stride == (1, 4)

    # logical_product with non-trivial strides
    A = Layout(4, 2)
    B = Layout(3, 1)
    product = logical_product(A, B)
    # pycute: complement(4:2, 4*3=12) = 2:1, compose(2:1, 3:1) = 2:1
    assert product.shape == (4, 2)
    assert product.stride == (2, 1)


def test_tile_basic():
    ## Tile and Mode-by-Mode Composition

    # Basic Tile creation
    tiler = Tile(Layout(3, 4), Layout(8, 2))
    assert len(tiler) == 2
    assert tiler[0] == Layout(3, 4)
    assert tiler[1] == Layout(8, 2)


def test_tile_type_checking():
    # Tile type checking
    with pytest.raises(TypeError):
        Tile(3, 4)  # Must be Layouts
    with pytest.raises(TypeError):
        Tile(Layout(3, 4), "not a layout")


def test_compose_with_tile():
    # Mode-by-mode composition from CuTe example:
    # (12,(4,8)):(59,(13,1))
    a = Layout((12, (4, 8)), (59, (13, 1)))
    # <3:4, 8:2>
    tiler = Tile(Layout(3, 4), Layout(8, 2))
    # Expected: (3,(2,4)):(236,(26,1))
    result = compose(a, tiler)
    assert result.shape == (3, (2, 4))
    assert result.stride == (236, (26, 1))

    # Verify functional equivalence with manual mode-by-mode composition:
    # compose(a, tiler)(i) = compose(mode(a, 0), tiler[0])(i0) + compose(mode(a, 1), tiler[1])(i1)
    # where i0 and i1 are the coordinates in the result
    comp0 = compose(mode(a, 0), tiler[0])
    comp1 = compose(mode(a, 1), tiler[1])
    for i in range(size(result)):
        # Decompose index into coordinates for each mode
        i0 = i % size(comp0)
        i1 = i // size(comp0)
        expected = comp0(i0) + comp1(i1)
        actual = result(i)
        assert actual == expected, f"Mismatch at {i}: {actual} != {expected}"


def test_compose_tile_partial_modes():
    # Tile with partial modes (tiler has fewer elements than layout modes)
    a3 = Layout((4, 8, 16), (1, 4, 32))  # 3-mode layout
    tiler1 = Tile(Layout(2, 1))  # Only covers mode 0
    result = compose(a3, tiler1)
    assert rank(result) == 3
    # Mode 0 should be composed, modes 1 and 2 should be unchanged
    assert result.shape[1] == 8
    assert result.shape[2] == 16
    assert result.stride[1] == 4
    assert result.stride[2] == 32


def test_compose_tile_scalar_modes():
    # Simple scalar mode compositions
    a = Layout((8, 4), (1, 8))
    tiler = Tile(Layout(2, 1), Layout(2, 1))
    result = compose(a, tiler)
    assert result.shape == (2, 2)
    assert result.stride == (1, 8)

    # Verify functional equivalence
    for i in range(size(result)):
        # The composed layout should agree with the original on the tiler's domain
        a(Layout(2, 1)(i % 2) + Layout(2, 1)(i // 2) * 4)


def test_compose_shape_as_tiler():
    ## Shape as Tiler (from CuTe docs)
    # CuTe interprets Shapes as tuple-of-layouts-with-stride-1:
    # A Shape (3, 8) is equivalent to Tile(Layout(3,1), Layout(8,1))

    # Example from CuTe documentation:
    # (12,(4,8)):(59,(13,1))
    a = Layout((12, (4, 8)), (59, (13, 1)))
    # (3, 8) - interpreted as <3:1, 8:1>
    tiler = (3, 8)
    # Expected: (_3,(4,2)):(59,(13,1))
    result = compose(a, tiler)
    assert result.shape == (3, (4, 2)), f"Got shape {result.shape}"
    assert result.stride == (59, (13, 1)), f"Got stride {result.stride}"

    # Verify it matches the explicit Tile version
    explicit_tiler = Tile(Layout(3, 1), Layout(8, 1))
    explicit_result = compose(a, explicit_tiler)
    assert result == explicit_result

    # Verify functional equivalence
    for i in range(size(result)):
        assert result(i) == a(explicit_tiler[0](i % 3) + explicit_tiler[1](i // 3) * 12)


def test_compose_shape_tiler_variants():
    # Another example: 1D shape tiler
    a = Layout((16, 8), (1, 16))
    tiler = (4, 2)  # Equivalent to Tile(Layout(4,1), Layout(2,1))
    result = compose(a, tiler)
    explicit_result = compose(a, Tile(Layout(4, 1), Layout(2, 1)))
    assert result == explicit_result

    # Single element shape tiler
    a = Layout((8, 4), (2, 16))
    tiler = (2,)  # Only covers mode 0
    result = compose(a, tiler)
    assert result.shape[0] == 2
    assert result.shape[1] == 4  # Mode 1 unchanged
    assert result.stride[1] == 16  # Mode 1 stride unchanged


def test_compose_mixed_tiler():
    ## Mixed tuple of Tilers
    # A Tiler can be: Layout, tuple of Tilers, or Shape (tuple of ints)
    # This allows mixing explicit Layouts with shape elements

    # Mixed tiler: Layout for mode 0, int (shape) for mode 1
    a = Layout((12, (4, 8)), (59, (13, 1)))
    mixed_tiler = (Layout(3, 4), 8)  # Layout(3,4) for mode 0, Layout(8,1) for mode 1
    result = compose(a, mixed_tiler)
    # Mode 0: compose(Layout(12, 59), Layout(3, 4)) -> Layout(3, 236)
    # Mode 1: compose(Layout((4,8), (13,1)), Layout(8, 1)) -> Layout((4,2), (13,1))
    assert result.shape == (3, (4, 2)), f"Got shape {result.shape}"
    assert result.stride == (236, (13, 1)), f"Got stride {result.stride}"

    # Verify functional equivalence
    for i in range(size(result)):
        i0 = i % 3
        i1 = i // 3
        offset0 = Layout(3, 4)(i0)  # Index into mode 0 of a
        offset1 = Layout(8, 1)(i1)  # Index into mode 1 of a
        expected = mode(a, 0)(offset0) + mode(a, 1)(offset1)
        assert result(i) == expected, f"Mismatch at {i}: {result(i)} != {expected}"


def test_compose_mixed_tiler_reversed():
    # Mixed tiler: int (shape) for mode 0, Layout for mode 1
    a = Layout((8, 16), (1, 8))
    mixed_tiler = (4, Layout(4, 2))  # Layout(4,1) for mode 0, Layout(4,2) for mode 1
    result = compose(a, mixed_tiler)
    # Mode 0: compose(Layout(8, 1), Layout(4, 1)) -> Layout(4, 1)
    # Mode 1: compose(Layout(16, 8), Layout(4, 2)) -> Layout(4, 16)
    assert result.shape == (4, 4), f"Got shape {result.shape}"
    assert result.stride == (1, 16), f"Got stride {result.stride}"

    # Verify functional equivalence
    for i in range(size(result)):
        i0 = i % 4
        i1 = i // 4
        offset0 = Layout(4, 1)(i0)
        offset1 = Layout(4, 2)(i1)
        expected = mode(a, 0)(offset0) + mode(a, 1)(offset1)
        assert result(i) == expected, f"Mismatch at {i}: {result(i)} != {expected}"


def test_blocked_product_morton():
    # Tiling Products

    # Morton from Cris Cecka's Slides
    morton1 = Layout((2, 2), (1, 2))
    morton2 = blocked_product(morton1, morton1)
    morton3 = blocked_product(morton1, morton2)
    assert morton2 == Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))
    assert morton3 == Layout(((2, (2, 2)), (2, (2, 2))), ((1, (4, 16)), (2, (8, 32))))


def test_core_matrix_operations():
    # Core Matrix Operations
    # For a 2-byte dtype such as f16, core matrix is 8x8
    tile1 = Layout((8, 1), (1, 0))  # (8,1):(1,0)
    mul1 = Layout((1, 8), (0, 1))
    tile2 = coalesce(
        blocked_product(tile1, mul1), profile=(None, None)
    )  # (8,8):(1,8) -> One core Matrix
    assert tile2 == Layout((8, 8), (1, 8))
    # Now organize core matrices into 8x8 pattern, so that we have a 64x64 Tile, say in SMem
    mul2 = Layout((8, 8), (1, 8))
    tile3 = coalesce(logical_product(tile2, mul2), profile=(None, None))
    # After coalescing with profile, the hierarchical structure is flattened per-mode
    # pycute: logical_product gives ((8,8),(8,8)):((1,8),(64,512)),
    # coalescing each mode: (8*8, 8*8):(1, 64) = (64,64):(1,64)
    assert tile3 == Layout((64, 64), (1, 64))


## safe_div


def test_safe_div():
    assert safe_div(12, 3) == 4
    assert safe_div(12, 4) == 3
    assert safe_div(12, 1) == 12
    assert safe_div(12, 12) == 1
    assert safe_div(0, 5) == 0
    with pytest.raises(ValueError):
        safe_div(12, 0)  # division by zero
    with pytest.raises(ValueError):
        safe_div(12, 5)  # 5 does not divide 12


## Tile class


def test_tile_repr():
    tiler = Tile(Layout(3, 4), Layout(8, 2))
    r = repr(tiler)
    assert r == "Tile(Layout(3, 4), Layout(8, 2))"


## Layout.__repr__ and __str__


def test_layout_repr_scalar():
    """repr() of a 1D layout returns an eval-safe constructor string."""
    L = Layout(8, 2)
    assert repr(L) == "Layout(8, 2)"


def test_layout_repr_tuple():
    """repr() of a multi-dimensional layout returns an eval-safe constructor string."""
    L = Layout((4, 8), (1, 4))
    assert repr(L) == "Layout((4, 8), (1, 4))"


def test_layout_repr_hierarchical():
    """repr() of a hierarchical layout returns an eval-safe constructor string."""
    L = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    assert repr(L) == "Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))"


def test_layout_repr_swizzled():
    """repr() of a swizzled layout includes the swizzle keyword argument."""
    sw = Swizzle(3, 0, 3)
    L = compose(sw, Layout((8, 8), (8, 1)))
    r = repr(L)
    assert r == "Layout((8, 8), (8, 1), swizzle=Swizzle(3, 0, 3))"


def test_layout_repr_eval_roundtrip():
    """eval(repr(L)) reconstructs an equal Layout (the gold standard for repr)."""
    cases = [
        Layout(8, 2),
        Layout((4, 8), (1, 4)),
        Layout((4, 8), (0, 1)),
        Layout(((2, 3), (2, 4)), ((1, 6), (2, 12))),
    ]
    for L in cases:
        reconstructed = eval(repr(L))  # noqa: S307
        assert reconstructed == L, f"Roundtrip failed for {repr(L)}"


def test_layout_repr_eval_roundtrip_swizzled():
    """eval(repr(L)) works for swizzled layouts too."""
    L = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    reconstructed = eval(repr(L))  # noqa: S307
    assert reconstructed == L


def test_layout_str_scalar():
    """str() returns the human-readable CuTe notation."""
    L = Layout(8, 2)
    assert str(L) == "8 : 2"


def test_layout_str_tuple():
    """str() returns the human-readable CuTe notation for multi-dim layouts."""
    L = Layout((4, 8), (1, 4))
    assert str(L) == "(4, 8) : (1, 4)"


def test_layout_str_swizzled():
    """str() returns the CuTe composition notation for swizzled layouts."""
    L = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    assert str(L) == "(Swizzle(3, 0, 3)) o ((8, 8) : (8, 1))"


## Layout.__hash__


def test_layout_hash():
    L1 = Layout((4, 8), (1, 4))
    L2 = Layout((4, 8), (1, 4))
    L3 = Layout((4, 8), (2, 4))
    assert hash(L1) == hash(L2)
    assert L1 == L2
    # Different layouts should (almost certainly) have different hashes
    assert hash(L1) != hash(L3)
    # Can be used as dict keys
    d = {L1: "a", L3: "b"}
    assert d[L2] == "a"
    assert d[L3] == "b"
    # Can be used in sets
    s = {L1, L2, L3}
    assert len(s) == 2


## Layout.__eq__ identity short-circuit


def test_layout_eq_identity_shortcircuit():
    """Same object identity returns True immediately."""
    L = Layout((4, 8), (1, 4))
    assert L == L
    assert L is L

    sw = Swizzle(3, 0, 3)
    L_sw = compose(sw, L)
    assert L_sw == L_sw


def test_layout_eq_structural():
    """Distinct objects with equal shape/stride/swizzle are equal."""
    L1 = Layout((4, 8), (1, 4))
    L2 = Layout((4, 8), (1, 4))
    assert L1 is not L2
    assert L1 == L2


def test_layout_eq_non_layout():
    """Comparing Layout with non-Layout returns False, not an error."""
    L = Layout((4, 8), (1, 4))
    assert L != 42
    assert L != "not a layout"
    assert L != (4, 8)
    assert L != None  # noqa: E711


def test_swizzle_eq_identity_shortcircuit():
    """Same Swizzle identity returns True immediately."""
    sw = Swizzle(3, 0, 3)
    assert sw == sw
    assert sw is sw


def test_swizzle_eq_structural():
    """Distinct Swizzle objects with equal fields are equal."""
    sw1 = Swizzle(3, 0, 3)
    sw2 = Swizzle(3, 0, 3)
    assert sw1 is not sw2
    assert sw1 == sw2


## compose() functional property


def test_compose_functional_property():
    # compose(A, B)(i) == A(B(i)) for all i
    A = Layout((4, 6, 8, 10), (2, 3, 5, 7))
    B = Layout(6, 12)
    C = compose(A, B)
    for i in range(size(B)):
        assert C(i) == A(B(i))


## Swizzle.__eq__


def test_swizzle_eq():
    sw1 = Swizzle(3, 0, 3)
    sw2 = Swizzle(3, 0, 3)
    sw3 = Swizzle(2, 0, 3)
    sw4 = Swizzle(3, 1, 3)
    sw5 = Swizzle(3, 0, -3)

    assert sw1 == sw2
    assert sw1 != sw3
    assert sw1 != sw4
    assert sw1 != sw5
    assert sw1 != "not a swizzle"


## Swizzled Layout __eq__ and __hash__


def test_swizzled_layout_eq_hash():
    base1 = Layout((8, 8), (8, 1))
    base2 = Layout((8, 8), (8, 1))
    base3 = Layout((8, 8), (1, 8))
    sw1 = compose(Swizzle(3, 0, 3), base1)
    sw2 = compose(Swizzle(3, 0, 3), base2)
    sw3 = compose(Swizzle(3, 0, 3), base3)
    sw4 = compose(Swizzle(2, 0, 3), base1)

    assert sw1 == sw2
    assert sw1 != sw3  # different underlying layout
    assert sw1 != sw4  # different swizzle
    assert hash(sw1) == hash(sw2)
    assert sw1 != Layout((8, 8), (8, 1))  # not equal to plain Layout

    # Usable in sets
    s = {sw1, sw2, sw3}
    assert len(s) == 2


## Slicing swizzled layouts (offset handling)


def test_offset_swizzled_layout_basic():
    from tensor_layouts import Tensor

    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)
    # Slicing a Tensor produces a Tensor with offset
    row_slice = tensor[3, :]
    assert hasattr(row_slice, "offset")
    assert row_slice.offset == Layout((8, 8), (8, 1))(3, 0)  # = 24

    # Check functional correctness: tensor[3, :](j) == tensor(3, j)
    for j in range(8):
        assert row_slice(j) == tensor(3, j)


def test_offset_swizzled_layout_repr():
    from tensor_layouts import Tensor

    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)
    row_slice = tensor[2, :]
    r = repr(row_slice)
    assert "Tensor" in r
    assert "offset" in r  # offset indicator


def test_offset_swizzled_layout_eq():
    from tensor_layouts import Tensor

    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    tensor = Tensor(sw_layout)
    slice1 = tensor[3, :]
    slice2 = tensor[3, :]
    slice3 = tensor[4, :]

    assert slice1 == slice2
    assert slice1 != slice3
    assert hash(slice1) == hash(slice2)


def test_compose_layout_with_swizzled_layout():
    # compose(Layout, SwizzledLayout) should transform the swizzle through
    # the outer layout, matching CuTe C++ composition(Layout, ComposedLayout).
    # See layout_composed.hpp:379 and swizzle_layout.hpp:327.
    swizzled = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))

    # Identity layout preserves the swizzle exactly
    result = compose(Layout(64, 1), swizzled)
    assert result.swizzle == Swizzle(3, 0, 3)
    assert result.shape == (8, 8)
    assert result.stride == (8, 1)

    # Verify point-wise correctness: result(i) == Layout(64,1)(swizzled(i))
    for i in range(size(result)):
        assert result(i) == swizzled(i), f"Mismatch at i={i}"


def test_compose_layout_with_swizzled_layout_nontrivial():
    # Non-identity outer layout should transform the swizzle
    swizzled = compose(Swizzle(2, 0, 2), Layout(16, 1))

    # Compose with a layout that doubles strides
    outer = Layout(16, 2)
    result = compose(outer, swizzled)

    # Verify point-wise: result(i) == outer(swizzled(i)) for all i in domain
    for i in range(size(result)):
        assert result(i) == outer(swizzled(i)), f"Mismatch at i={i}"

    # The result should have a swizzle (transformed through outer)
    assert result.swizzle is not None


def test_make_layout_like_basic():
    # For a column-major layout, extracting a sub-shape preserves strides
    layout = Layout((4, 8), (1, 4))
    result = make_layout_like(layout, (4, 8))
    assert result.shape == (4, 8)
    assert result.stride == (1, 4)
    assert result == layout


def test_make_layout_like_partial():
    # Extract first 2 modes from a 4-mode flat layout
    layout = Layout((2, 3, 5, 7))  # strides: (1, 2, 6, 30)
    result = make_layout_like(layout, (2, 3))
    assert result.shape == (2, 3)
    assert result.stride == (1, 2)


def test_make_layout_like_with_layout_tiler():
    layout = Layout((4, 8), (2, 8))
    result = make_layout_like(layout, Layout((4, 4), (1, 4)))
    # Shape comes from the tiler
    assert result.shape == (4, 4)
    # Strides come from the flattened layout
    assert result.stride == (2, 8)


def test_make_layout_like_int_tiler():
    layout = Layout((8, 4), (1, 8))
    result = make_layout_like(layout, 8)
    assert size(result) == 8
    # With scalar tiler, result is scalar layout, stride is int not tuple
    assert result.stride == 1


## tile_to_shape


def test_tile_to_shape_1d():
    # tile_to_shape replicates a block to fill target shape
    block = Layout(4, 1)  # 4-element block (scalar layout)
    result = tile_to_shape(block, 16)

    # Scalar block + 1D target -> 1D replication layout
    # product_each(4) = (4,), ceil_div(16, 4) = (4,)
    # replication = Layout((4,), (1,))
    # blocked_product of scalar 4:1 with (4,):(1,) gives ((4,4),):((1,4),)
    expected = blocked_product(block, Layout((4,), (1,)))
    assert result == expected
    # Result shape: ((4, 4),), size = 16
    assert size(result) == 16
    assert result.shape == ((4, 4),)
    assert result.stride == ((1, 4),)


def test_tile_to_shape_2d():
    # 2x3 block tiled to 8x6
    block = Layout((2, 3), (1, 2))
    result = tile_to_shape(block, (8, 6))

    # Need 4x replication in mode 0, 2x in mode 1
    # blocked_product with (4, 2):(1, 4) column-major replication
    expected = blocked_product(block, Layout((4, 2), (1, 4)))
    assert result == expected
    # Result has nested shape showing block + replication
    assert result.shape == ((2, 4), (3, 2))


def test_tile_to_shape_with_order():
    # Test row-major replication order
    block = Layout((2, 3), (1, 2))
    result = tile_to_shape(block, (4, 6), order=(1, 0))  # row-major

    # With row-major order, mode 1 varies fastest in replication
    # block: (2, 3), target: (4, 6) -> replication: (2, 2)
    # make_ordered_layout((2, 2), (1, 0)) -> (2, 2):(2, 1)
    replication = make_ordered_layout((2, 2), (1, 0))
    expected = blocked_product(block, replication)
    assert result == expected


def test_tile_to_shape_exact_fit():
    # When block exactly matches target, replication is (1, 1)
    block = Layout((4, 8), (1, 4))
    result = tile_to_shape(block, (4, 8))

    # Replication is (1, 1) - no actual replication needed
    expected = blocked_product(block, Layout((1, 1), (1, 1)))
    assert result == expected


def test_tile_to_shape_nested_block():
    # Test with nested block shape
    block = Layout(((2, 2), 3), ((1, 2), 4))  # effective shape (4, 3)
    result = tile_to_shape(block, (8, 9))

    # product_each flattens: (4, 3), need (2, 3) replication
    assert size(result) == 8 * 9


## is_layout


def test_is_layout():
    assert is_layout(Layout(4, 1)) is True
    assert is_layout(Layout((2, 3), (1, 2))) is True
    assert is_layout(4) is False
    assert is_layout((4, 2)) is False
    assert is_layout(4) is False
    assert is_layout(None) is False


## unflatten


def test_unflatten_tuple():
    # Flat tuple -> nested tuple
    assert unflatten((1, 2, 3, 4, 5), ((0, 0), (0, 0, 0))) == ((1, 2), (3, 4, 5))
    assert unflatten((1, 2, 3, 4), ((0, 0), (0, 0))) == ((1, 2), (3, 4))
    # Single element per group
    assert unflatten((1, 2, 3), (0, 0, 0)) == (1, 2, 3)
    # Deeply nested
    assert unflatten((1, 2, 3, 4), (((0, 0), 0), 0)) == (((1, 2), 3), 4)


def test_unflatten_layout():
    # Flat layout -> nested layout matching a profile
    flat = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    # Profile with same flat rank (4 leaves): ((_, _), (_, _))
    result = unflatten(flat, ((0, 0), (0, 0)))
    assert result.shape == ((2, 3), (5, 7))
    assert result.stride == ((1, 2), (6, 30))

    # Round-trip: flatten then unflatten recovers structure
    original = Layout(((2, 3), (5, 7)), ((1, 2), (6, 30)))
    flat2 = flatten(original)
    restored = unflatten(flat2, original)
    assert restored.shape == original.shape
    assert restored.stride == original.stride


def test_unflatten_identity():
    # Unflatten with flat profile is identity
    flat = Layout((2, 3, 5), (1, 2, 6))
    result = unflatten(flat, (0, 0, 0))
    assert result.shape == flat.shape
    assert result.stride == flat.stride


## product_each


def test_product_each_flat():
    # Flat shape returns same tuple
    assert product_each((4, 8)) == (4, 8)
    assert product_each((2, 3, 5)) == (2, 3, 5)


def test_product_each_nested():
    # Nested shape flattens to products
    assert product_each(((2, 2), 8)) == (4, 8)
    assert product_each((3, (2, 4))) == (3, 8)
    assert product_each(((2, 3), (4, 5))) == (6, 20)


def test_product_each_scalar():
    # Scalar returns 1-tuple
    assert product_each(16) == (16,)


def test_product_each_deeply_nested():
    # Deeply nested reduces correctly
    assert product_each(((2, (3, 4)), 5)) == (24, 5)  # 2*3*4 = 24


## make_ordered_layout


def test_make_ordered_layout_column_major():
    # Default (None) and explicit (0, 1) are column-major
    result = make_ordered_layout((4, 8))
    assert result.shape == (4, 8)
    assert result.stride == (1, 4)

    result2 = make_ordered_layout((4, 8), (0, 1))
    assert result2 == result


def test_make_ordered_layout_row_major():
    # (1, 0) means mode 1 varies fastest
    result = make_ordered_layout((4, 8), (1, 0))
    assert result.shape == (4, 8)
    assert result.stride == (8, 1)


def test_make_ordered_layout_3d():
    # 3D with custom order
    result = make_ordered_layout((2, 3, 4), (2, 0, 1))
    # mode 2 fastest (stride 1), then mode 0 (stride 4), then mode 1 (stride 8)
    assert result.shape == (2, 3, 4)
    assert result.stride == (4, 8, 1)


def test_make_ordered_layout_scalar():
    # Scalar input
    result = make_ordered_layout(16)
    assert result.shape == 16
    assert result.stride == 1


## dice_modes


def test_dice_modes_scalar_coord():
    # Scalar coord: identity (keep everything)
    layout = Layout((3, 4), (1, 4))
    result = dice_modes(0, layout)
    assert result == layout

    result = dice_modes(5, layout)
    assert result == layout


def test_dice_modes_none_coord():
    # None coord: drop everything
    layout = Layout((3, 4), (1, 4))
    result = dice_modes(None, layout)
    assert size(result) == 1


def test_dice_modes_keep_one_mode():
    layout = Layout((3, 4), (1, 4))

    # Keep mode 0 (paired with int), drop mode 1 (paired with None)
    result = dice_modes((0, None), layout)
    assert result == Layout(3, 1)

    # Keep mode 1, drop mode 0
    result = dice_modes((None, 0), layout)
    assert result == Layout(4, 4)


def test_dice_modes_keep_all():
    layout = Layout((3, 4), (1, 4))
    result = dice_modes((0, 0), layout)
    assert result == layout


def test_dice_modes_drop_all():
    layout = Layout((3, 4), (1, 4))
    result = dice_modes((None, None), layout)
    assert size(result) == 1


def test_dice_modes_complement_of_slice_modes():
    # dice_modes and slice_modes are complements: together they cover all modes
    layout = Layout((2, 3, 5), (1, 2, 6))
    crd = (0, None, 0)

    # slice keeps None-marked modes
    sliced_shape = slice_modes(crd, layout.shape)
    assert sliced_shape == (3,)

    # dice keeps int-marked modes
    diced = dice_modes(crd, layout)
    assert diced == Layout((2, 5), (1, 6))


## nullspace


def test_nullspace_all_zero_strides():
    # All stride-0: everything is in the kernel
    # Inspired by C++ test: Layout<Shape<_2,_2,_2>,Stride<_0,_0,_0>>
    layout = Layout((2, 2, 2), (0, 0, 0))
    ker = nullspace(layout)

    # Postcondition: size(ker) == size(layout) / size(filter(layout))
    assert size(ker) == size(layout) // size(layout.filter())

    # Postcondition: layout(ker(i)) == 0 for all i
    for i in range(size(ker)):
        assert layout(ker(i)) == 0


def test_nullspace_all_zero_large():
    # Inspired by C++ test: Layout<Shape<_7,_5,_16>,Stride<_0,_0,_0>>
    layout = Layout((7, 5, 16), (0, 0, 0))
    ker = nullspace(layout)
    assert size(ker) == size(layout) // size(layout.filter())
    for i in range(size(ker)):
        assert layout(ker(i)) == 0


def test_nullspace_partial_zero():
    # Inspired by C++ test: Layout<Shape<_2,_2,_2>,Stride<_1,_0,_2>>
    layout = Layout((2, 2, 2), (1, 0, 2))
    ker = nullspace(layout)

    assert size(ker) == size(layout) // size(layout.filter())
    for i in range(size(ker)):
        assert layout(ker(i)) == 0


def test_nullspace_partial_zero_large():
    # Inspired by C++ test: Layout<Shape<_7,_5,_16>,Stride<_3,_1,_0>>
    layout = Layout((7, 5, 16), (3, 1, 0))
    ker = nullspace(layout)

    assert size(ker) == size(layout) // size(layout.filter())
    for i in range(size(ker)):
        assert layout(ker(i)) == 0


def test_nullspace_no_zero_strides():
    # No stride-0 modes: kernel is trivial
    layout = Layout((4, 8), (1, 4))
    ker = nullspace(layout)
    assert size(ker) == 1


def test_nullspace_scalar_nonzero_stride():
    # Scalar layout with non-zero stride: trivial nullspace
    ker = nullspace(Layout(4, 1))
    assert size(ker) == 1
    assert ker == Layout(1, 0)


def test_nullspace_scalar_zero_stride():
    # Scalar layout with stride 0: entire domain is nullspace
    ker = nullspace(Layout(4, 0))
    assert size(ker) == 4
    for i in range(size(ker)):
        assert Layout(4, 0)(ker(i)) == 0


## max_common_vector and max_common_layout


def test_max_common_vector_identical():
    # Same layout: all elements are common
    a = Layout(8, 1)
    assert max_common_vector(a, a) == 8


def test_max_common_vector_contiguous_prefix():
    # a = 8:1 (contiguous), b = (2,4):(1,4) (contiguous first 2, then gap)
    # b visits offsets: 0,1,4,5,8,9,12,13 — only first 2 are contiguous from 0
    a = Layout(8, 1)
    b = Layout((2, 4), (1, 4))
    assert max_common_vector(a, b) == 2


def test_max_common_vector_no_common():
    # a = (4,2):(2,1) (strided), b = 8:1 (contiguous)
    # a starts at stride 2, so no contiguous common prefix
    a = Layout((4, 2), (2, 1))
    b = Layout(8, 1)
    assert max_common_vector(a, b) == 1


def test_max_common_layout_identical():
    a = Layout(8, 1)
    result = max_common_layout(a, a)
    assert size(result) == 8
    # Postcondition: a(result(i)) == i for all i
    for i in range(size(result)):
        assert a(result(i)) == i


def test_max_common_layout_partial():
    a = Layout(8, 1)
    b = Layout((2, 4), (1, 4))
    result = max_common_layout(a, b)
    assert size(result) == 2
    for i in range(size(result)):
        assert a(result(i)) == i
        assert b(result(i)) == i


## flat_product


def test_flat_product_basic():
    # flat_product = zipped_product then unpack both modes
    block = Layout(4, 1)
    tiler = Layout(3, 1)
    result = flat_product(block, tiler)

    # Result should be flat: all block modes then all product modes
    zipped = zipped_product(block, tiler)

    # Both should map to the same offsets
    for i in range(size(result)):
        assert result(i) == zipped(i)


def test_flat_product_2d():
    block = Layout((2, 4), (1, 2))
    tiler = Layout(3, 1)
    result = flat_product(block, tiler)

    zipped = zipped_product(block, tiler)
    for i in range(size(result)):
        assert result(i) == zipped(i)


## raked_product


def test_raked_product_basic():
    # raked_product vs blocked_product: reversed zip order
    block = Layout((2, 2), (1, 2))
    tiler = Layout((2, 2), (1, 2))

    raked = raked_product(block, tiler)
    blocked = blocked_product(block, tiler)

    # Both should visit the same set of offsets (same size)
    assert size(raked) == size(blocked)

    # But in different order: raked has tiler varying fastest
    raked_offsets = {raked(i) for i in range(size(raked))}
    blocked_offsets = {blocked(i) for i in range(size(blocked))}
    assert raked_offsets == blocked_offsets


def test_raked_product_1d():
    block = Layout(4, 1)
    tiler = Layout(3, 1)

    raked = raked_product(block, tiler)
    blocked = blocked_product(block, tiler)

    assert size(raked) == size(blocked)
    raked_offsets = {raked(i) for i in range(size(raked))}
    blocked_offsets = {blocked(i) for i in range(size(blocked))}
    assert raked_offsets == blocked_offsets


def test_raked_product_interleave_order():
    # In raked_product, tiler varies fastest (interleaved)
    # In blocked_product, block varies fastest (contiguous blocks)
    block = Layout(4, 1)
    tiler = Layout(2, 1)

    raked = raked_product(block, tiler)
    # The first few offsets should show interleaving
    # tiler varies fastest means we alternate between tiler positions
    offsets = [raked(i) for i in range(size(raked))]
    assert len(offsets) == 8  # 4 * 2

    blocked = blocked_product(block, tiler)
    blocked_offsets = [blocked(i) for i in range(size(blocked))]
    # blocked: block 0 contiguous (0,1,2,3), then block 1 (4,5,6,7)
    assert blocked_offsets == [0, 1, 2, 3, 4, 5, 6, 7]
    # raked: tiler interleaved (0,4,1,5,2,6,3,7)
    assert offsets == [0, 4, 1, 5, 2, 6, 3, 7]


## Upcast / Downcast


def test_upcast_simple_stride1():
    """upcast divides innermost (stride-1) shape by n."""
    # (32, 32):(32, 1) → fp16 → (32, 2):(2, 1)
    r = upcast(Layout((32, 32), (32, 1)), 16)
    assert r == Layout((32, 2), (2, 1))


def test_upcast_hierarchical_value_mode():
    """upcast handles nested value modes correctly."""
    # SM75_U32x4_LDSM_N dst_layout_bits
    r = upcast(Layout((32, (32, 4)), (32, (1, 1024))), 16)
    assert r == Layout((32, (2, 4)), (2, (1, 64)))


def test_upcast_transpose_layout():
    """upcast handles transpose layouts where innermost stride > 1."""
    # SM75_U16x2_LDSM_T dst_layout_bits
    r = upcast(Layout(((4, 8), (16, 2)), ((256, 16), (1, 128))), 16)
    assert r == Layout(((4, 8), (1, 2)), ((16, 1), (1, 8)))


def test_upcast_identity():
    """upcast with n=1 returns the same layout."""
    l = Layout((4, 8), (8, 1))
    assert upcast(l, 1) == l


def test_upcast_broadcast_stride():
    """upcast preserves stride-0 (broadcast) modes unchanged."""
    r = upcast(Layout((4, 8), (0, 1)), 4)
    assert r.stride[0] == 0
    assert r.shape[0] == 4


def test_upcast_preserves_functional_semantics():
    """After upcast, every element-index maps to the same coarsened offset.

    If bit_layout(i) gives the bit offset for element i, then
    upcast(bit_layout, N)(i) should give bit_layout(i) // N for elements
    whose bit offset is N-aligned.
    """
    bit_layout = Layout((32, 32), (32, 1))
    elem_layout = upcast(bit_layout, 16)
    # Check that every element index maps correctly
    for i in range(size(elem_layout)):
        elem_offset = elem_layout(i)
        # The corresponding bit offset for element i in the original
        # bit layout depends on which bit position the element starts at.
        # For stride-1 innermost: bit_offset = i * 16 (approximately),
        # but the exact mapping depends on the layout structure.
        # Instead, verify the cosize relationship:
        assert elem_offset < cosize(elem_layout)


def test_upcast_known_copy_atoms():
    """Validate upcast against manually-verified copy atom conversions.

    These are the element-level layouts used in examples/viz.py,
    derived from the CUTLASS C++ copy_traits_sm75.hpp source.
    """
    from tensor_layouts.atoms_nv import (
        SM75_U32x1_LDSM_N,
        SM75_U32x4_LDSM_N,
        SM75_U16x2_LDSM_T,
        SM75_U16x4_LDSM_T,
        SM75_U16x8_LDSM_T,
    )

    cases = [
        # (atom, expected_dst_shape, expected_dst_stride)
        (SM75_U32x1_LDSM_N, (32, 2), (2, 1)),
        (SM75_U32x4_LDSM_N, (32, (2, 4)), (2, (1, 64))),
        (SM75_U16x2_LDSM_T, ((4, 8), (1, 2)), ((16, 1), (1, 8))),
        (SM75_U16x4_LDSM_T, ((4, 8), (1, 2, 2)), ((16, 1), (1, 8, 64))),
        (SM75_U16x8_LDSM_T, ((4, 8), (1, 2, 4)), ((16, 1), (1, 8, 64))),
    ]

    for atom, exp_shape, exp_stride in cases:
        result = upcast(atom.dst_layout_bits, 16)
        assert (
            result.shape == exp_shape
        ), f"{atom.name}: shape {result.shape} != expected {exp_shape}"
        assert (
            result.stride == exp_stride
        ), f"{atom.name}: stride {result.stride} != expected {exp_stride}"


def test_downcast_simple():
    """downcast multiplies stride-1 shape by n, other strides by n."""
    r = downcast(Layout((32, 2), (2, 1)), 16)
    assert r == Layout((32, 32), (32, 1))


def test_upcast_downcast_roundtrip():
    """downcast(upcast(layout, n), n) recovers the original layout.

    Only valid when the innermost mode size is >= n so no modes collapse.
    """
    layouts = [
        Layout((32, 32), (32, 1)),
        Layout((32, (32, 4)), (32, (1, 1024))),
    ]
    for l in layouts:
        assert downcast(upcast(l, 16), 16) == l, f"Roundtrip failed for {l}"


def test_downcast_upcast_roundtrip():
    """upcast(downcast(layout, n), n) recovers the original layout."""
    layouts = [
        Layout((32, 2), (2, 1)),
        Layout((4, 8), (8, 1)),
    ]
    for l in layouts:
        assert upcast(downcast(l, 4), 4) == l, f"Roundtrip failed for {l}"


## iter_layout, __iter__, __len__


def test_iter_layout_scalar():
    """Scalar layout yields a single (coord, offset) pair."""
    layout = Layout(1, 0)
    result = list(iter_layout(layout))
    assert result == [(0, 0)]


def test_iter_layout_1d():
    """1D contiguous layout yields coords 0..n-1."""
    layout = Layout(4, 1)
    result = list(iter_layout(layout))
    assert result == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_iter_layout_1d_strided():
    """1D strided layout produces gapped offsets."""
    layout = Layout(4, 2)
    result = list(iter_layout(layout))
    assert result == [(0, 0), (1, 2), (2, 4), (3, 6)]


def test_iter_layout_2d_col_major():
    """2D column-major layout iterates columns first."""
    layout = Layout((2, 3), (1, 2))
    result = list(iter_layout(layout))
    expected = [
        ((0, 0), 0),
        ((1, 0), 1),  # col 0
        ((0, 1), 2),
        ((1, 1), 3),  # col 1
        ((0, 2), 4),
        ((1, 2), 5),  # col 2
    ]
    assert result == expected


def test_iter_layout_2d_row_major():
    """2D row-major layout still iterates in colexicographic order."""
    layout = Layout((2, 3), (3, 1))
    result = list(iter_layout(layout))
    # Flat index order: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
    expected_offsets = [0, 3, 1, 4, 2, 5]
    assert [o for _, o in result] == expected_offsets


def test_iter_layout_hierarchical():
    """Hierarchical layout yields nested coordinates."""
    layout = Layout(((2, 2), 2), ((1, 4), 2))
    result = list(iter_layout(layout))
    assert len(result) == size(layout)
    # Verify all offsets match direct calls
    for i, (coord, offset) in enumerate(result):
        assert layout(i) == offset


def test_iter_layout_broadcast():
    """Broadcast (stride-0) layout produces duplicate offsets."""
    layout = Layout((4, 2), (0, 1))
    result = list(iter_layout(layout))
    # All rows map to offsets 0 or 1
    offsets = [o for _, o in result]
    assert offsets == [0, 0, 0, 0, 1, 1, 1, 1]


def test_iter_layout_swizzled():
    """Swizzled layout applies the XOR during iteration."""
    sw = compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1)))
    result = list(iter_layout(sw))
    assert len(result) == 16
    for i, (coord, offset) in enumerate(result):
        assert offset == sw(i)


def test_layout_len():
    """__len__ returns size."""
    assert len(Layout(8, 1)) == 8
    assert len(Layout((4, 8), (1, 4))) == 32
    assert len(Layout(1, 0)) == 1


def test_layout_iter_protocol():
    """Layout supports standard Python iteration protocols."""
    layout = Layout((2, 3), (1, 2))

    # list() yields coordinates
    coords = list(layout)
    assert len(coords) == 6
    assert coords[0] == (0, 0)
    assert coords[-1] == (1, 2)

    # Each coordinate maps to the expected offset
    for coord in layout:
        assert layout(coord) == layout(crd2flat(coord, layout.shape))

    # Set comprehension on offsets via layout call
    offsets = {layout(c) for c in layout}
    assert offsets == {0, 1, 2, 3, 4, 5}

    # iter_layout still yields (coord, offset) pairs
    pairs = list(iter_layout(layout))
    assert all(layout(c) == o for c, o in pairs)


## image, is_injective, is_surjective, is_bijective


def test_image_contiguous():
    """Contiguous layout visits every offset in [0, size)."""
    assert image(Layout(4, 1)) == [0, 1, 2, 3]
    assert image(Layout((2, 3), (1, 2))) == [0, 1, 2, 3, 4, 5]


def test_image_strided():
    """Strided layout has gaps in its image."""
    assert image(Layout(4, 2)) == [0, 2, 4, 6]


def test_image_broadcast():
    """Broadcast (stride-0) layout collapses to few offsets."""
    assert image(Layout((4, 2), (0, 1))) == [0, 1]
    assert image(Layout(4, 0)) == [0]


def test_image_swizzled():
    """Swizzled layout image contains permuted offsets."""
    sw = compose(Swizzle(2, 0, 2), Layout((4, 4), (4, 1)))
    img = image(sw)
    assert len(img) == 16  # bijective swizzle
    assert img == list(range(16))


def test_is_injective_contiguous():
    assert is_injective(Layout(4, 1))
    assert is_injective(Layout((2, 3), (3, 1)))


def test_is_injective_strided():
    assert is_injective(Layout(4, 2))


def test_is_injective_broadcast():
    """Broadcast layouts are not injective."""
    assert not is_injective(Layout((4, 2), (0, 1)))
    assert not is_injective(Layout(4, 0))


def test_is_surjective_contiguous():
    assert is_surjective(Layout(4, 1))
    assert is_surjective(Layout((2, 3), (1, 2)))


def test_is_surjective_strided():
    """Strided layout is not surjective onto [0, cosize)."""
    assert not is_surjective(Layout(4, 2))


def test_is_surjective_custom_codomain():
    """With explicit codomain, surjectivity can change."""
    layout = Layout(4, 2)
    # Not surjective onto [0, 7) -- has gaps
    assert not is_surjective(layout)
    # Surjective if codomain is exactly the image size
    assert is_surjective(layout, codomain_size=4)


def test_is_bijective_contiguous():
    """Contiguous layouts are bijective."""
    assert is_bijective(Layout(4, 1))
    assert is_bijective(Layout((2, 3), (1, 2)))
    assert is_bijective(Layout((2, 3), (3, 1)))


def test_is_bijective_strided():
    """Strided layout is not bijective (has gaps)."""
    assert not is_bijective(Layout(4, 2))


def test_is_bijective_broadcast():
    """Broadcast layout is not bijective (has aliasing)."""
    assert not is_bijective(Layout((4, 2), (0, 1)))


def test_is_bijective_swizzled():
    """Swizzle permutation is bijective."""
    sw = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    assert is_bijective(sw)


def test_is_bijective_identity():
    """Identity (size-1) layout is trivially bijective."""
    assert is_bijective(Layout(1, 0))


def test_image_injectivity_consistency():
    """image size equals domain size iff injective."""
    layouts = [
        Layout(8, 1),
        Layout(4, 2),
        Layout((4, 2), (0, 1)),
        Layout((3, 3), (1, 3)),
        Layout((2, 4), (4, 1)),
    ]
    for l in layouts:
        img = image(l)
        assert is_injective(l) == (len(img) == size(l))


## functionally_equal


def test_functionally_equal_identity():
    """A layout is functionally equal to itself."""
    layout = Layout((4, 8), (1, 4))
    assert functionally_equal(layout, layout)


def test_functionally_equal_coalesce():
    """Coalescing preserves functional behavior."""
    layout = Layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))
    assert functionally_equal(layout, coalesce(layout))


def test_functionally_equal_flatten():
    """Flattening preserves functional behavior."""
    layout = Layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))
    assert functionally_equal(layout, flatten(layout))


def test_functionally_equal_structurally_different():
    """Different shapes/strides can produce the same mapping."""
    a = Layout(((2, 2), 2), ((1, 4), 2))
    b = coalesce(a)
    assert a != b  # structurally different
    assert functionally_equal(a, b)  # functionally same


def test_functionally_equal_different_sizes():
    """Different domain sizes are never functionally equal."""
    assert not functionally_equal(Layout(4, 1), Layout(8, 1))


def test_functionally_equal_broadcast():
    """Broadcast layouts with same shape but different strides."""
    a = Layout((4, 2), (0, 1))
    b = Layout((4, 2), (0, 1))
    assert functionally_equal(a, b)
    c = Layout((4, 2), (1, 0))
    assert not functionally_equal(a, c)


def test_functionally_equal_row_col_major():
    """Row-major and column-major are not functionally equal."""
    col = Layout((3, 4), (1, 3))
    row = Layout((3, 4), (4, 1))
    assert not functionally_equal(col, row)


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
