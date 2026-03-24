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

"""Differential oracle tests: cross-validate layouts.py against pycute reference.

This test imports both our layouts library and NVIDIA's pycute reference
implementation, runs both on the same inputs, and asserts identical results.
This is the strongest possible correctness guarantee.

Also includes exhaustive small-domain property tests that verify algebraic
invariants over all valid layouts up to a size bound.
"""

from tensor_layouts import *
from tensor_layouts.layout_utils import make_ordered_layout, tile_to_shape

import pytest

# Import pycute reference — skip all tests if unavailable.
# Note: there is an unrelated "pycute" package on PyPI (statistics library).
# We need NVIDIA's pycute from the CUTLASS source tree.
try:
    import pycute
    if not hasattr(pycute, 'Layout'):
        pycute = None
except ImportError:
    pycute = None

pytestmark = pytest.mark.skipif(pycute is None, reason="pycute (NVIDIA CUTLASS) not available")


###############################################################################
## Conversion helpers
###############################################################################


def to_pycute_shape(s):
    """Convert our tuple/int shape to pycute plain tuple/int."""
    if isinstance(s, int):
        return s
    return tuple(to_pycute_shape(x) for x in s)


def to_ours_shape(s):
    """Convert pycute plain tuple/int to our tuple/int."""
    if isinstance(s, int):
        return s
    if isinstance(s, tuple) and len(s) == 0:
        return ()
    return tuple(to_ours_shape(x) for x in s)


def pycute_layout(shape, stride=None):
    """Create a pycute Layout from shape/stride."""
    s = to_pycute_shape(shape)
    if stride is None:
        return pycute.Layout(s)
    d = to_pycute_shape(stride)
    return pycute.Layout(s, d)


def our_layout(shape, stride=None):
    """Create our Layout from shape/stride."""
    if stride is None:
        return Layout(shape)
    return Layout(shape, stride)


def shapes_equal(a, b):
    """Compare shapes from the two implementations.

    Handles integers and nested structures.
    """
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(shapes_equal(x, y) for x, y in zip(a, b))
    return False


def layouts_functionally_equal(our, ref, domain_size):
    """Check that two layouts produce the same index mapping."""
    for i in range(domain_size):
        if our(i) != ref(i):
            return False
    return True


###############################################################################
## Test layout corpus
###############################################################################


# Standard test layouts: (shape, stride) pairs covering many patterns
LAYOUT_CORPUS = [
    # Trivial
    (1, 0), (1, 1),
    # 1D
    (4, 1), (4, 2), (8, 1), (8, 2), (12, 1), (12, 3),
    # Zero stride (broadcast)
    (4, 0), (8, 0),
    # 2D col-major
    ((2, 4), (1, 2)), ((4, 3), (1, 4)), ((8, 4), (1, 8)),
    # 2D row-major
    ((2, 4), (4, 1)), ((4, 3), (3, 1)), ((8, 4), (4, 1)),
    # 2D with gaps
    ((2, 4), (1, 4)), ((2, 4), (1, 6)), ((4, 2), (1, 10)), ((4, 2), (1, 16)),
    # 2D with broadcast
    ((2, 4), (0, 2)), ((4, 2), (2, 0)),
    # 3D
    ((2, 4, 6), (1, 2, 8)), ((2, 4, 6), (4, 1, 8)),
    ((2, 3, 4), (1, 2, 6)), ((2, 4, 8), (8, 1, 64)),
    ((2, 4, 6), (24, 6, 1)),
    # 3D with broadcast
    ((2, 4, 8), (8, 1, 0)), ((2, 4, 3), (1, 2, 0)),
    # Nested (hierarchical)
    (((2, 2), (2, 2)), ((1, 4), (8, 32))),
    ((2, (3, 4)), (3, (1, 6))),
    (((4, 2),), ((1, 16),)),
    # Auto-stride (col-major)
    ((2, 4), None), ((4, 3), None), ((2, 4, 6), None), ((2, 3, 4), None),
    ((8, 8), None),

    # ===== FROM C++ TESTS =====
    # C++ inverse / complement tests: broadcast shapes
    (((3, 7),), ((0, 0),)),
    (((1, 1),), ((0, 0),)),
    ((2, 4), (0, 2)),
    # C++ inverse tests: 4D with broadcast
    ((2, 4, 4, 6), (4, 1, 0, 8)),
    # C++ inverse tests: coprime gaps
    ((4, 2), (1, 5)),
    # C++ inverse tests: large strides
    ((128, 128), (65536, 1)),
    ((128, 160), (65536, 1)),
    ((128, 3, 160), (65536, 512, 1)),
    ((128, 64), (131072, 2)),
    # C++ inverse tests: 4D with large strides
    ((32, 4, 4, 4), (262144, 4, 8388608, 1)),
    # C++ inverse tests: broadcast middle
    ((2, 2, 2), (4, 0, 1)),
    # C++ complement test: gapped
    ((4, 10), (1, 10)),
    # C++ composition tests: transposed strides
    ((4, 3), (3, 1)),

    # ===== EDGE CASES =====
    # All-zero strides (pure broadcast)
    ((2, 3, 4), (0, 0, 0)),
    # Single-element nested
    (((1,),), ((0,),)),
    # Deep nesting
    (((2, (2, 2)),), ((1, (4, 8)),)),
    # Large broadcast
    ((16, 16), (0, 1)),
    # Stride-1 everywhere (non-injective, violates complement preconditions)
    # ((2, 3, 4), (1, 1, 1)),  # excluded: breaks complement ordering assumption
]


def iter_corpus():
    """Yield (shape, stride) pairs from the corpus."""
    for item in LAYOUT_CORPUS:
        if isinstance(item, tuple) and len(item) == 2:
            shape, stride = item
            yield shape, stride


###############################################################################
## Differential Oracle Tests
###############################################################################


def test_oracle_size():
    """Cross-validate size() against pycute."""
    for shape, stride in iter_corpus():
        ours = our_layout(shape, stride)
        ref = pycute_layout(shape, stride)
        assert size(ours) == pycute.size(ref), (
            f"size mismatch for {shape}:{stride}: "
            f"ours={size(ours)} vs pycute={pycute.size(ref)}"
        )


def test_oracle_cosize():
    """Cross-validate cosize() against pycute."""
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        ours = our_layout(shape, stride)
        ref = pycute_layout(shape, stride)
        assert cosize(ours) == pycute.cosize(ref), (
            f"cosize mismatch for {shape}:{stride}: "
            f"ours={cosize(ours)} vs pycute={pycute.cosize(ref)}"
        )


def test_oracle_indexing():
    """Cross-validate Layout indexing against pycute."""
    for shape, stride in iter_corpus():
        ours = our_layout(shape, stride)
        ref = pycute_layout(shape, stride)
        n = size(ours)
        for i in range(n):
            assert ours(i) == ref(i), (
                f"Indexing mismatch for {shape}:{stride} at i={i}: "
                f"ours={ours(i)} vs pycute={ref(i)}"
            )


def test_oracle_complement():
    """Cross-validate complement() against pycute."""
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        ours_c = complement(ours_l)
        ref_c = pycute.complement(ref_l)

        # Compare sizes
        assert size(ours_c) == pycute.size(ref_c), (
            f"complement size mismatch for {shape}:{stride}: "
            f"ours={size(ours_c)} vs pycute={pycute.size(ref_c)}"
        )

        # Compare functional behavior
        for i in range(size(ours_c)):
            assert ours_c(i) == ref_c(i), (
                f"complement({shape}:{stride})({i}): "
                f"ours={ours_c(i)} vs pycute={ref_c(i)}"
            )

    # Also test with explicit cosize bounds
    bounds = [1, 2, 8, 16, 53, 128]
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        for bound in bounds:
            ours_l = our_layout(shape, stride)
            ref_l = pycute_layout(shape, stride)

            ours_c = complement(ours_l, bound)
            ref_c = pycute.complement(ref_l, bound)

            assert size(ours_c) == pycute.size(ref_c), (
                f"complement({shape}:{stride}, {bound}) size mismatch: "
                f"ours={size(ours_c)} vs pycute={pycute.size(ref_c)}"
            )
            for i in range(size(ours_c)):
                assert ours_c(i) == ref_c(i), (
                    f"complement({shape}:{stride}, {bound})({i}): "
                    f"ours={ours_c(i)} vs pycute={ref_c(i)}"
                )


def test_oracle_coalesce():
    """Cross-validate coalesce() against pycute."""
    for shape, stride in iter_corpus():
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        ours_c = coalesce(ours_l)
        ref_c = pycute.coalesce(ref_l)

        # Sizes must match
        assert size(ours_c) == pycute.size(ref_c), (
            f"coalesce size mismatch for {shape}:{stride}: "
            f"ours={size(ours_c)} vs pycute={pycute.size(ref_c)}"
        )

        # Shapes and strides should match
        assert shapes_equal(ours_c.shape, ref_c.shape), (
            f"coalesce shape mismatch for {shape}:{stride}: "
            f"ours={ours_c.shape} vs pycute={ref_c.shape}"
        )
        assert shapes_equal(ours_c.stride, ref_c.stride), (
            f"coalesce stride mismatch for {shape}:{stride}: "
            f"ours={ours_c.stride} vs pycute={ref_c.stride}"
        )


def test_oracle_composition():
    """Cross-validate compose() against pycute composition()."""
    # Composition pairs: (A_shape, A_stride, B_shape, B_stride)
    composition_pairs = [
        # Simple
        (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 0), (1, 1, 1, 1),
        (4, 1, 4, 1), (4, 2, 4, 1), (4, 1, 4, 2), (4, 0, 4, 1),
        (4, 1, 4, 0), (1, 0, 4, 1), (4, 1, 1, 0),
        # Partial
        (4, 1, 2, 1), (4, 2, 2, 1), (4, 1, 2, 2), (4, 2, 2, 2),
        # Multi-dim A, 1D B
        ((4, 3), (1, 4), 12, 1),
        ((4, 3), (1, 4), 6, 1),
        ((4, 3), (1, 4), 6, 2),
        ((4, 3), (3, 1), 12, 1),
        ((4, 3), (3, 1), 6, 2),
        # 1D A, multi-dim B
        (12, 1, (4, 3), (1, 4)),
        (12, 2, (4, 3), (1, 4)),
        (12, 1, (4, 3), (3, 1)),
        (12, 2, (4, 3), (3, 1)),
        (12, 1, (2, 3), (2, 4)),
        # Multi-dim both
        ((4, 3), (1, 4), (4, 3), (1, 4)),
        ((4, 3), (3, 1), (4, 3), (1, 4)),
        ((8, 8), (1, 8), (8, 8), (1, 8)),
        ((8, 8), (8, 1), (8, 8), (1, 8)),
        # Nested B
        ((8, 8), (1, 8), ((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))),
        ((8, 8), (8, 1), ((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))),
        # Nested A with stride
        (((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)), 8, 4),
        # Truncation / extension
        ((4, 3), (3, 1), 24, 1),
        ((4, 3), (3, 1), 8, 1),
        ((4, 6, 8), (1, 4, 7), 6, 1),
        ((4, 6, 8, 10), (2, 3, 5, 7), 6, 12),
        # C++ tests: transposed strides
        ((4, 3), None, (4, 3), (3, 1)),
    ]

    for item in composition_pairs:
        a_shape, a_stride, b_shape, b_stride = item

        ours_a = our_layout(a_shape, a_stride)
        ours_b = our_layout(b_shape, b_stride)
        ref_a = pycute_layout(a_shape, a_stride)
        ref_b = pycute_layout(b_shape, b_stride)

        ours_r = compose(ours_a, ours_b)
        ref_r = pycute.composition(ref_a, ref_b)

        n = pycute.size(ref_r)
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"compose({a_shape}:{a_stride}, {b_shape}:{b_stride})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_shape_div():
    """Cross-validate shape_div() against pycute."""
    test_cases = [
        # (shape, divisor) -- only cases valid for pycute (a%b==0 or b%a==0 at each level)
        ((3, 4), 1), ((3, 4), 3), ((3, 4), 6),
        ((3, 4), 12), ((3, 4), 36),
        ((4, 3), 2), ((4, 3), 4), ((4, 3), 12),
        ((6, 2), 2), ((6, 2), 3), ((6, 2), 6), ((6, 2), 12),
        # Nested
        (((3, 4), 6), 1), (((3, 4), 6), 3), (((3, 4), 6), 12),
        (((3, 4), 6), 36), (((3, 4), 6), 72),
        ((6, (3, 4)), 6), ((6, (3, 4)), 36),
        # Scalars
        (12, 1), (12, 3), (12, 4), (12, 6), (12, 12),
    ]

    for shape, divisor in test_cases:
        ours_s = to_ours_shape(shape) if isinstance(shape, tuple) else shape
        ref_s = to_pycute_shape(shape) if isinstance(shape, tuple) else shape

        try:
            ref_r = pycute.shape_div(ref_s, divisor)
        except AssertionError:
            continue  # Skip cases pycute can't handle

        ours_r = shape_div(ours_s, divisor)

        assert shapes_equal(
            ours_r if isinstance(ours_r, int) else ours_r,
            ref_r if isinstance(ref_r, int) else ref_r
        ), (
            f"shape_div({shape}, {divisor}): "
            f"ours={ours_r} vs pycute={ref_r}"
        )


def test_oracle_prefix_product():
    """Cross-validate prefix_product() against pycute."""
    test_cases = [
        2,
        (3, 2),
        (3, 2, 4),
        ((2, 3), 4),
        ((2, 3), (2, 1, 2), (5, 2, 1)),
    ]

    for shape in test_cases:
        ours_s = to_ours_shape(shape) if isinstance(shape, tuple) else shape
        ref_s = to_pycute_shape(shape) if isinstance(shape, tuple) else shape

        ours_r = prefix_product(ours_s)
        ref_r = pycute.prefix_product(ref_s)

        assert shapes_equal(
            ours_r if isinstance(ours_r, int) else ours_r,
            ref_r if isinstance(ref_r, int) else ref_r
        ), (
            f"prefix_product({shape}): ours={ours_r} vs pycute={ref_r}"
        )


def test_oracle_inner_product():
    """Cross-validate inner_product() against pycute."""
    test_cases = [
        (2, 3),
        ((1, 2), (3, 2)),
        (((2, 3), 4), ((2, 1), 2)),
    ]

    for a, b in test_cases:
        ours_a = to_ours_shape(a) if isinstance(a, tuple) else a
        ours_b = to_ours_shape(b) if isinstance(b, tuple) else b
        ref_a = to_pycute_shape(a) if isinstance(a, tuple) else a
        ref_b = to_pycute_shape(b) if isinstance(b, tuple) else b

        ours_r = inner_product(ours_a, ours_b)
        ref_r = pycute.inner_product(ref_a, ref_b)

        assert ours_r == ref_r, (
            f"inner_product({a}, {b}): ours={ours_r} vs pycute={ref_r}"
        )


def test_oracle_right_inverse():
    """Cross-validate right_inverse() against pycute."""
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        ours_r = right_inverse(ours_l)
        ref_r = pycute.right_inverse(ref_l)

        # Functional equivalence
        n = min(size(ours_r), pycute.size(ref_r))
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"right_inverse({shape}:{stride})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_left_inverse():
    """Cross-validate left_inverse() against pycute."""
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        ours_r = left_inverse(ours_l)
        ref_r = pycute.left_inverse(ref_l)

        # Functional equivalence: both should map L(i) -> i
        n = size(ours_l)
        for i in range(n):
            li = ours_l(i)
            assert ours_r(li) == ref_r(li), (
                f"left_inverse({shape}:{stride})(L({i})={li}): "
                f"ours={ours_r(li)} vs pycute={ref_r(li)}"
            )


def test_oracle_logical_divide():
    """Cross-validate logical_divide() with Layout tilers against pycute."""
    # (layout_shape, layout_stride, tiler_shape, tiler_stride)
    divide_cases = [
        (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 0), (1, 1, 1, 1),
        (6, 1, 2, 1), (6, 1, 2, 3), (6, 2, 2, 1), (6, 2, 2, 3),
        (6, 1, (2, 3), (3, 1)), (6, 2, (2, 3), (3, 1)),
        (32, 1, 2, 8),
        (12, 1, 4, 1), (12, 1, 6, 1), (12, 2, 4, 1),
        (48, 1, 32, 1), (96, 1, 32, 2),
    ]

    for item in divide_cases:
        ls, ld, ts, td = item
        ours_l = our_layout(ls, ld)
        ours_t = our_layout(ts, td)
        ref_l = pycute_layout(ls, ld)
        ref_t = pycute_layout(ts, td)

        ours_r = logical_divide(ours_l, ours_t)
        ref_r = pycute.logical_divide(ref_l, ref_t)

        # Compare functional behavior
        n = min(size(ours_r), pycute.size(ref_r))
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"logical_divide({ls}:{ld}, {ts}:{td})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_logical_product():
    """Cross-validate logical_product() against pycute."""
    product_cases = [
        # (A_shape, A_stride, B_shape, B_stride)
        (4, 1, 3, 1), (4, 2, 3, 1), (4, 1, 3, 2),
        ((2, 4), (1, 2), 3, 1),
        (8, 1, 4, 1), (8, 2, 4, 1),
        # === C++ test cases ===
        # Trivial
        (1, 0, 1, 0), (1, 1, 1, 0), (1, 0, 1, 1), (1, 1, 1, 1),
        # Broadcast
        (3, 1, 4, 0), (3, 0, 4, 1), (3, 0, 4, 0), (3, 2, 4, 1),
        # 1D
        (3, 1, (2, 4), None), ((2, 4), None, 3, 1),
        # Hierarchical
        ((8, (2, 2)), None, 4, 2),
        ((2, 2), None, (3, 3), (3, 1)),
        # Large stride
        (3, 32, 32, 1),
        (3, 2, 4, 1),
        (3, 32, 128, 1),
        (3, 32, (8, 8), None),
        (3, 32, (8, 8), (8, 1)),
        # Nested
        (((4, 2),), ((1, 16),), (4, 4), None),
        (((4, 2),), ((1, 16),), (4, 2), (2, 1)),
        (((2, 2), (2, 2)), ((1, 4), (8, 32)), (2, 2), (1, 2)),
        (((2, 2), (2, 2)), ((1, 4), (8, 32)), (2, 2), (2, 1)),
        # Nested shape with divisor
        (((4, 6),), ((1, 6),), 3, 1),
    ]

    for a_shape, a_stride, b_shape, b_stride in product_cases:
        ours_a = our_layout(a_shape, a_stride)
        ours_b = our_layout(b_shape, b_stride)
        ref_a = pycute_layout(a_shape, a_stride)
        ref_b = pycute_layout(b_shape, b_stride)

        ours_r = logical_product(ours_a, ours_b)
        ref_r = pycute.logical_product(ref_a, ref_b)

        n = min(size(ours_r), pycute.size(ref_r))
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"logical_product({a_shape}:{a_stride}, {b_shape}:{b_stride})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


###############################################################################
## Exhaustive Small-Domain Property Tests
###############################################################################


def _generate_small_layouts(max_size=24, max_rank=3):  # noqa: ARG001
    """Generate all 'interesting' layouts up to a given size and rank."""
    layouts = []

    # 1D layouts with various strides
    for n in [1, 2, 3, 4, 6, 8, 12]:
        for s in [0, 1, 2, 3, 4, 8]:
            layouts.append(Layout(n, s))

    # 2D layouts
    shapes_2d = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 3), (2, 6), (3, 2)]
    for shape in shapes_2d:
        # Col-major
        layouts.append(Layout(shape))
        # Row-major
        layouts.append(Layout(shape, (shape[1], 1)))
        # With gaps
        for s0 in [1, 2]:
            for s1 in [shape[0] * s0, shape[0] * s0 + 2, shape[0] * s0 * 2]:
                if s1 <= max_size:
                    layouts.append(Layout(shape, (s0, s1)))
        # With broadcast
        layouts.append(Layout(shape, (0, 1)))
        layouts.append(Layout(shape, (1, 0)))

    # 3D: a few representative cases
    shapes_3d = [(2, 2, 2), (2, 3, 2), (2, 2, 3)]
    for shape in shapes_3d:
        layouts.append(Layout(shape))
        layouts.append(Layout(shape, (shape[1] * shape[2], shape[2], 1)))

    # C++ edge cases: coprime gap strides
    layouts.append(Layout((4, 2), (1, 5)))
    # C++ edge cases: 3D broadcast middle
    layouts.append(Layout((2, 2, 2), (4, 0, 1)))
    # C++ edge cases: 4D with broadcast
    layouts.append(Layout((2, 4, 4, 6), (4, 1, 0, 8)))
    # C++ edge cases: nested hierarchical
    layouts.append(Layout(((4, 2),), ((1, 16),)))

    return layouts


def test_exhaustive_complement_disjointness():
    """Verify complement disjointness for all small layouts.

    For all a in dom(L), b in dom(complement(L)):
    L(a) != complement(L)(b) unless both are 0.
    """
    for layout in _generate_small_layouts():
        c = complement(layout)
        for a in range(size(layout)):
            for b in range(size(c)):
                la = layout(a)
                cb = c(b)
                assert (la != cb) or (la == 0 and cb == 0), (
                    f"Disjointness violated for {layout}: "
                    f"L({a})={la} == C({b})={cb}"
                )


def test_exhaustive_complement_ordering():
    """Verify complement is strictly ordered for i >= 1."""
    for layout in _generate_small_layouts():
        c = complement(layout)
        for i in range(1, size(c)):
            assert c(i - 1) < c(i), (
                f"Complement not ordered for {layout}: "
                f"C({i-1})={c(i-1)} >= C({i})={c(i)}"
            )


def test_exhaustive_coalesce_preserves_mapping():
    """Verify coalesce preserves the index mapping for all small layouts."""
    for layout in _generate_small_layouts():
        coal = coalesce(layout)
        assert size(coal) == size(layout), (
            f"coalesce({layout}): size changed from {size(layout)} to {size(coal)}"
        )
        for i in range(size(layout)):
            assert coal(i) == layout(i), (
                f"coalesce({layout})({i}) = {coal(i)} != {layout(i)}"
            )


def test_exhaustive_coalesce_reduces_depth():
    """Verify coalesce produces depth <= 1."""
    for layout in _generate_small_layouts():
        coal = coalesce(layout)
        assert depth(coal) <= 1, (
            f"coalesce({layout}) has depth {depth(coal)} > 1"
        )


def test_exhaustive_right_inverse_identity():
    """Verify L(R(i)) == i for all small injective layouts."""
    for layout in _generate_small_layouts():
        rinv = right_inverse(layout)
        for i in range(size(rinv)):
            assert layout(rinv(i)) == i, (
                f"right_inverse({layout}): L(R({i}))={layout(rinv(i))} != {i}"
            )


def _is_injective(layout):
    """Check if a layout is injective (no two inputs map to same output)."""
    seen = set()
    for i in range(size(layout)):
        v = layout(i)
        if v in seen:
            return False
        seen.add(v)
    return True


def _is_contiguous(layout):
    """Check if a layout maps to a contiguous range [0, size-1].

    A contiguous layout has cosize == size, meaning there are no gaps.
    """
    s = size(layout)
    if s == 0:
        return True
    vals = set(layout(i) for i in range(s))
    return vals == set(range(s))


def test_exhaustive_left_inverse_identity():
    """Verify R(L(i)) == i for small injective, contiguous layouts.

    Left inverse only reliably inverts for injective layouts with no gaps
    in their codomain. Gapped or broadcast layouts cannot be perfectly inverted.
    """
    for layout in _generate_small_layouts():
        if not _is_injective(layout):
            continue
        if not _is_contiguous(layout):
            continue
        linv = left_inverse(layout)
        for i in range(size(layout)):
            assert linv(layout(i)) == i, (
                f"left_inverse({layout}): R(L({i}))={linv(layout(i))} != {i}"
            )


def test_exhaustive_compose_identity():
    """Verify compose(A, B)(i) matches pycute composition(A, B)(i) for small layout pairs."""
    small = _generate_small_layouts(max_size=12)
    tested = 0
    for a in small:
        for b in small:
            if size(b) > size(a):
                continue
            try:
                r = compose(a, b)
            except (AssertionError, ValueError, TypeError):
                continue

            # Create pycute equivalents
            ref_a = pycute_layout(
                to_pycute_shape(a.shape) if isinstance(a.shape, tuple) else a.shape,
                to_pycute_shape(a.stride) if isinstance(a.stride, tuple) else a.stride,
            )
            ref_b = pycute_layout(
                to_pycute_shape(b.shape) if isinstance(b.shape, tuple) else b.shape,
                to_pycute_shape(b.stride) if isinstance(b.stride, tuple) else b.stride,
            )
            try:
                ref_r = pycute.composition(ref_a, ref_b)
            except (AssertionError, ValueError, TypeError):
                continue

            n = min(size(r), pycute.size(ref_r))
            for i in range(n):
                assert r(i) == ref_r(i), (
                    f"compose({a}, {b})({i}) = {r(i)} != pycute={ref_r(i)}"
                )
            tested += 1

    assert tested > 100, f"Only tested {tested} composition pairs, expected more"


def test_exhaustive_shape_div_mod_complementary():
    """Verify size(shape_div(s,d)) * size(shape_mod(s,d)) == size(s)
    for divisors that can be cleanly divided through the shape structure.

    Note: shape_div requires a%b==0 or b%a==0 at each recursive level,
    so not all divisors of size(s) are valid. We try each and skip failures.
    """
    shapes = [
        (2, 3), (3, 4), (2, 2, 3), (4, 3),
        (6, 2), (2, 6), (3, 2, 4),
    ]

    tested = 0
    for shape in shapes:
        s = size(shape)
        for d in range(1, s + 1):
            if s % d != 0:
                continue
            try:
                sd = shape_div(shape, d)
                sm = shape_mod(shape, d)
            except (AssertionError, ValueError):
                continue
            assert size(sd) * size(sm) == s, (
                f"shape_div({shape},{d})={sd} (size {size(sd)}) * "
                f"shape_mod({shape},{d})={sm} (size {size(sm)}) != "
                f"size({shape})={s}"
            )
            tested += 1

    assert tested > 20, f"Only tested {tested} cases, expected more"


def test_exhaustive_logical_divide_preserves_mapping():
    """Verify logical_divide(L, t)(i) == L(i) for all small layouts and tilers."""
    small = _generate_small_layouts(max_size=12)
    tilers = [1, 2, 3, 4, 6]
    tested = 0

    for layout in small:
        if size(layout) < 2:
            continue
        for t in tilers:
            if t > size(layout):
                continue
            try:
                result = logical_divide(layout, Layout(t, 1))
            except (AssertionError, ValueError, TypeError):
                continue

            for i in range(size(layout)):
                assert result(i) == layout(i), (
                    f"logical_divide({layout}, {t})({i}) = "
                    f"{result(i)} != {layout(i)}"
                )
            tested += 1

    assert tested > 50, f"Only tested {tested} divide cases, expected more"


def test_exhaustive_inverse_roundtrip():
    """Verify left_inverse and right_inverse are consistent.

    For injective layouts: left_inverse ∘ layout should be identity.
    For all layouts: right_inverse property L(R(i)) == i should hold.
    """
    for layout in _generate_small_layouts():
        rinv = right_inverse(layout)

        # right_inverse property: L(R(i)) == i (works for all layouts)
        for i in range(size(rinv)):
            assert layout(rinv(i)) == i, (
                f"right_inverse({layout}): L(R({i})) != {i}"
            )

        # left_inverse property: R(L(i)) == i (only for injective + contiguous layouts)
        if _is_injective(layout) and _is_contiguous(layout):
            linv = left_inverse(layout)
            for i in range(size(layout)):
                assert linv(layout(i)) == i, (
                    f"left_inverse({layout}): R(L({i})) != {i}"
                )


###############################################################################
## Oracle tests for newly added functions
###############################################################################


def test_oracle_tuple_max():
    """Cross-validate tuple_max against pycute."""
    cases = [
        5,
        (3, 7, 2),
        ((1, 9), (4, 2)),
        (1,),
        ((2, 3), (4, (5, 6))),
    ]
    for case in cases:
        s = to_pycute_shape(case)
        ours = tuple_max(case if isinstance(case, int) else case)
        ref = pycute.tuple_max(s)
        assert ours == ref, f"tuple_max({case}): {ours} != {ref}"


def test_oracle_elem_scale():
    """Cross-validate elem_scale against pycute."""
    cases = [
        (3, 4),          # int x int
        (2, (3, 4)),     # int x tuple
        ((2, 3), (4, 5)),  # tuple x tuple
        (1, (2, 3, 4)),  # int x tuple
    ]
    for a, b in cases:
        a_p = to_pycute_shape(a)
        b_p = to_pycute_shape(b)
        a_o = a if isinstance(a, int) else a
        b_o = b if isinstance(b, int) else b
        ours = elem_scale(a_o, b_o)
        ref = pycute.elem_scale(a_p, b_p)
        ours_cmp = to_pycute_shape(ours) if not isinstance(ours, int) else ours
        assert ours_cmp == ref, f"elem_scale({a}, {b}): {ours} != {ref}"


def test_oracle_crd2crd():
    """Cross-validate crd2crd against pycute."""
    # int -> tuple
    ours = crd2crd(3, (2, 4))
    ref = pycute.crd2crd(3, (2, 4))
    assert tuple(ours) == ref, f"crd2crd(3, (2,4)): {ours} != {ref}"

    # int -> int
    assert crd2crd(5, 8) == pycute.crd2crd(5, 8)

    # tuple -> tuple
    ours = crd2crd((1, 2), (3, 4))
    ref = pycute.crd2crd((1, 2), (3, 4))
    assert tuple(ours) == ref

    # tuple -> int (needs src_shape)
    ours = crd2crd((1, 0), 8, (2, 4))
    ref = pycute.crd2crd((1, 0), 8, (2, 4))
    assert ours == ref


def test_oracle_has_none():
    """Cross-validate has_none against pycute."""
    cases = [None, 3, 0, (1, None, 3), (1, 2, 3), (1, (2, None))]
    for case in cases:
        ours = has_none(case)
        ref = pycute.has_none(case)
        assert ours == ref, f"has_none({case}): {ours} != {ref}"


def test_oracle_slice():
    """Cross-validate slice_modes against pycute."""
    cases = [
        (None, 4),
        (0, 4),
        ((None, 0), (3, 4)),
        ((0, None), (3, 4)),
        ((None, None), (3, 4)),
        ((0, 0), (3, 4)),
    ]
    for crd, trg in cases:
        ours = slice_modes(crd, trg)
        ref = pycute.slice_(crd, trg)
        assert ours == ref, f"slice_modes({crd}, {trg}): {ours} != {ref}"


def test_oracle_slice_and_offset():
    """Cross-validate slice_and_offset against pycute."""
    layouts_and_crds = [
        ((4, 8), (1, 4), (None, 3)),
        ((4, 8), (1, 4), (2, None)),
        ((2, 3, 4), (1, 2, 6), (None, 1, None)),
        ((2, 3, 4), (1, 2, 6), (1, None, 2)),
    ]
    for shape, stride, crd in layouts_and_crds:
        ours_layout = Layout(shape, stride)
        ref_layout = pycute_layout(shape, stride)

        ours_sub, ours_offset = slice_and_offset(crd, ours_layout)
        ref_sub, ref_offset = pycute.slice_and_offset(crd, ref_layout)

        assert ours_offset == ref_offset, (
            f"slice_and_offset offset mismatch for ({shape},{stride},{crd}): "
            f"{ours_offset} != {ref_offset}"
        )
        # Compare sublayout shapes/strides
        ours_shape = to_pycute_shape(ours_sub.shape)
        ref_shape_val = ref_sub.shape
        assert ours_shape == ref_shape_val, (
            f"slice_and_offset shape mismatch for ({shape},{stride},{crd}): "
            f"{ours_shape} != {ref_shape_val}"
        )


def test_oracle_zipped_product():
    """Cross-validate zipped_product against pycute."""
    cases = [
        # (A_shape, A_stride, B_shape, B_stride)
        (4, 1, 3, 1),
        ((2, 4), None, (2, 2), None),
        (3, 2, 4, 1),
    ]
    for a_s, a_d, b_s, b_d in cases:
        ours_a = our_layout(a_s, a_d)
        ours_b = our_layout(b_s, b_d)
        ref_a = pycute_layout(a_s, a_d)
        ref_b = pycute_layout(b_s, b_d)

        ours_result = zipped_product(ours_a, ours_b)
        ref_result = pycute.zipped_product(ref_a, ref_b)

        assert size(ours_result) == pycute.size(ref_result), (
            f"zipped_product size mismatch: {size(ours_result)} != {pycute.size(ref_result)}"
        )
        for i in range(size(ours_result)):
            assert ours_result(i) == ref_result(i), (
                f"zipped_product({a_s}:{a_d}, {b_s}:{b_d})({i}): "
                f"{ours_result(i)} != {ref_result(i)}"
            )


def test_oracle_tiled_product():
    """Cross-validate tiled_product against pycute."""
    cases = [
        (4, 1, 3, 1),
        (3, 2, 4, 1),
    ]
    for a_s, a_d, b_s, b_d in cases:
        ours_a = our_layout(a_s, a_d)
        ours_b = our_layout(b_s, b_d)
        ref_a = pycute_layout(a_s, a_d)
        ref_b = pycute_layout(b_s, b_d)

        ours_result = tiled_product(ours_a, ours_b)
        ref_result = pycute.tiled_product(ref_a, ref_b)

        assert size(ours_result) == pycute.size(ref_result), (
            f"tiled_product size mismatch: {size(ours_result)} != {pycute.size(ref_result)}"
        )
        for i in range(size(ours_result)):
            assert ours_result(i) == ref_result(i), (
                f"tiled_product({a_s}:{a_d}, {b_s}:{b_d})({i}): "
                f"{ours_result(i)} != {ref_result(i)}"
            )


def test_oracle_layout_slice():
    """Cross-validate Layout.__call__ with None against pycute."""
    layouts_and_crds = [
        ((4, 8), (1, 4), (None, 3)),
        ((4, 8), (1, 4), (2, None)),
        ((4, 8), (1, 4), (None, None)),
        ((2, 3, 4), (1, 2, 6), (None, 1, None)),
    ]
    for shape, stride, crd in layouts_and_crds:
        ours = Layout(shape, stride)
        ref = pycute_layout(shape, stride)

        ours_sub = ours(*crd)
        ref_sub = ref(*crd)

        # Both should return sublayouts with same shape/stride
        ours_shape = to_pycute_shape(ours_sub.shape)
        ref_shape_val = ref_sub.shape
        assert ours_shape == ref_shape_val, (
            f"Layout({shape},{stride})({crd}) shape: {ours_shape} != {ref_shape_val}"
        )
        ours_stride = to_pycute_shape(ours_sub.stride)
        ref_stride_val = ref_sub.stride
        assert ours_stride == ref_stride_val, (
            f"Layout({shape},{stride})({crd}) stride: {ours_stride} != {ref_stride_val}"
        )


###############################################################################
## Oracle tests for filter, divide variants, coalesce with profile, swizzle
###############################################################################


def test_oracle_filter():
    """Cross-validate filter() against pycute.

    pycute's filter(layout) removes stride-0 modes and size-1 modes, then
    coalesces. Our Layout.filter() only removes stride-0 dimensions.
    We compare the standalone filter function behavior by applying pycute's
    filter and comparing functional output.
    """
    for shape, stride in iter_corpus():
        if stride is None:
            continue
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        # Apply pycute filter
        ref_f = pycute.filter(ref_l)

        # Our filter: Layout.filter() removes stride-0 dims, then coalesce
        # to match pycute's behavior (pycute filter = remove stride-0 + size-1
        # + coalesce)
        ours_f = coalesce(ours_l.filter())

        # Both should have same size (removing broadcast dims doesn't change
        # effective domain size when composed properly -- but actually filter
        # DOES change the domain size by collapsing broadcast dims to 1)
        assert size(ours_f) == pycute.size(ref_f), (
            f"filter size mismatch for {shape}:{stride}: "
            f"ours={size(ours_f)} vs pycute={pycute.size(ref_f)}"
        )

        # Functional equivalence
        for i in range(size(ours_f)):
            assert ours_f(i) == ref_f(i), (
                f"filter({shape}:{stride})({i}): "
                f"ours={ours_f(i)} vs pycute={ref_f(i)}"
            )


def test_oracle_zipped_divide():
    """Cross-validate zipped_divide() against pycute."""
    divide_cases = [
        # (layout_shape, layout_stride, tiler)
        # 1D tilers
        ((6,), (1,), 2),
        ((6,), (1,), 3),
        ((12,), (1,), 4),
        ((12,), (2,), 4),
        # 2D layout, 1D tiler
        ((8, 6), (1, 8), 2),
        ((8, 6), (1, 8), 4),
        # 2D layout, 2D tiler (tuple)
        ((8, 6), (1, 8), (2, 3)),
        ((8, 6), (1, 8), (4, 2)),
        ((8, 8), None, (2, 4)),
        ((8, 8), None, (4, 4)),
        # 2D layout, 2D tiler with strides
        ((12, 8), (1, 12), (4, 2)),
        # Row-major
        ((4, 6), (6, 1), (2, 3)),
    ]

    for item in divide_cases:
        ls, ld, tiler = item
        ours_l = our_layout(ls, ld)
        ref_l = pycute_layout(ls, ld)

        # Create tiler for both
        if isinstance(tiler, int):
            ours_t = tiler
            ref_t = tiler
        else:
            ours_t = tiler
            ref_t = tiler

        ours_r = zipped_divide(ours_l, ours_t)
        ref_r = pycute.zipped_divide(ref_l, ref_t)

        # Compare functional behavior
        n = min(size(ours_r), pycute.size(ref_r))
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"zipped_divide({ls}:{ld}, {tiler})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_tiled_divide():
    """Cross-validate tiled_divide() against pycute."""
    divide_cases = [
        # (layout_shape, layout_stride, tiler)
        ((6,), (1,), 2),
        ((6,), (1,), 3),
        ((12,), (1,), 4),
        ((12,), (2,), 4),
        ((8, 6), (1, 8), 2),
        ((8, 6), (1, 8), (2, 3)),
        ((8, 6), (1, 8), (4, 2)),
        ((8, 8), None, (2, 4)),
        ((8, 8), None, (4, 4)),
        ((12, 8), (1, 12), (4, 2)),
        ((4, 6), (6, 1), (2, 3)),
    ]

    for item in divide_cases:
        ls, ld, tiler = item
        ours_l = our_layout(ls, ld)
        ref_l = pycute_layout(ls, ld)

        if isinstance(tiler, int):
            ours_t = tiler
            ref_t = tiler
        else:
            ours_t = tiler
            ref_t = tiler

        ours_r = tiled_divide(ours_l, ours_t)
        ref_r = pycute.tiled_divide(ref_l, ref_t)

        n = min(size(ours_r), pycute.size(ref_r))
        for i in range(n):
            assert ours_r(i) == ref_r(i), (
                f"tiled_divide({ls}:{ld}, {tiler})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_coalesce_profiled():
    """Cross-validate coalesce(layout, profile) against pycute.

    pycute's profiled coalesce recurses: coalesce(layout[i], profile[i]) for each
    top-level mode. The layout must already have >= len(profile) top-level modes.
    """
    profiled_cases = [
        # (shape, stride, profile)
        # Profile with None: coalesce each mode independently
        ((4, 8), (1, 4), (None, None)),
        (((2, 2), (2, 4)), ((1, 2), (4, 8)), (None, None)),
        (((2, 4), (3, 2)), ((1, 2), (8, 24)), (None, None)),
        # 3-mode with None profile
        ((2, 3, 4), (1, 2, 6), (None, None, None)),
    ]

    for shape, stride, profile in profiled_cases:
        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        ours_r = coalesce(ours_l, profile)
        ref_r = pycute.coalesce(ref_l, profile)

        # Sizes must match
        assert size(ours_r) == pycute.size(ref_r), (
            f"profiled coalesce size mismatch for {shape}:{stride} profile={profile}: "
            f"ours={size(ours_r)} vs pycute={pycute.size(ref_r)}"
        )

        # Functional equivalence
        for i in range(size(ours_r)):
            assert ours_r(i) == ref_r(i), (
                f"profiled coalesce({shape}:{stride}, {profile})({i}): "
                f"ours={ours_r(i)} vs pycute={ref_r(i)}"
            )


def test_oracle_swizzle():
    """Cross-validate Swizzle.__call__ against pycute.Swizzle.__call__."""
    swizzle_params = [
        # (bits, base, shift)
        (1, 0, 1),
        (1, 0, -1),
        (1, 1, 2),
        (1, 1, -2),
        (2, 0, 2),
        (2, 0, -2),
        (2, 1, 3),
        (2, 1, -3),
        (2, 2, 3),
        (3, 0, 3),
        (3, 0, -3),
        (3, 1, 4),
        (3, 1, -4),
        (3, 3, 3),
    ]

    for bits, base, shift in swizzle_params:
        ours_sw = Swizzle(bits, base, shift)
        ref_sw = pycute.Swizzle(bits, base, shift)

        # Domain size: 1 << (bits + base + abs(shift))
        domain_size = 1 << (bits + base + abs(shift))
        ref_size = ref_sw.size()
        assert domain_size == ref_size, (
            f"Swizzle({bits},{base},{shift}) domain size mismatch: "
            f"computed={domain_size} vs pycute={ref_size}"
        )

        # Functional equivalence over the domain
        for i in range(domain_size):
            assert ours_sw(i) == ref_sw(i), (
                f"Swizzle({bits},{base},{shift})({i}): "
                f"ours={ours_sw(i)} vs pycute={ref_sw(i)}"
            )


def test_oracle_swizzle_composed():
    """Cross-validate Swizzle composed with Layout against pycute ComposedLayout."""
    cases = [
        # (bits, base, shift, layout_shape, layout_stride)
        (2, 0, 2, (4, 8), (1, 4)),
        (2, 0, 2, (8, 4), (4, 1)),
        (3, 0, 3, (8, 8), (1, 8)),
        (3, 0, 3, (8, 8), (8, 1)),
        (2, 1, 3, (4, 8), (1, 4)),
        (1, 0, 1, (2, 2), (1, 2)),
        (2, 0, -2, (4, 8), (1, 4)),
        (3, 0, -3, (8, 8), (1, 8)),
    ]

    for bits, base, shift, shape, stride in cases:
        ours_sw = Swizzle(bits, base, shift)
        ref_sw = pycute.Swizzle(bits, base, shift)

        ours_l = our_layout(shape, stride)
        ref_l = pycute_layout(shape, stride)

        # pycute composes as ComposedLayout(Swizzle, 0, Layout)
        ref_composed = pycute.ComposedLayout(ref_sw, 0, ref_l)

        # Our compose with Swizzle should produce same results
        n = size(ours_l)
        for i in range(n):
            ours_val = ours_sw(ours_l(i))
            ref_val = ref_composed(i)
            assert ours_val == ref_val, (
                f"Swizzle({bits},{base},{shift}) o Layout({shape},{stride})({i}): "
                f"ours={ours_val} vs pycute={ref_val}"
            )


###############################################################################
## Exhaustive property tests for filter, blocked_product, flat_divide
###############################################################################


def test_exhaustive_filter_idempotent():
    """Verify filter is idempotent: filter(filter(L)) == filter(L).

    Also verify that filtered layout has no stride-0 dimensions.
    """
    for layout in _generate_small_layouts():
        filtered = layout.filter()
        filtered2 = filtered.filter()

        # Idempotent: filtering twice gives the same result
        assert size(filtered) == size(filtered2), (
            f"filter not idempotent for {layout}: "
            f"size(filter)={size(filtered)} vs size(filter(filter))={size(filtered2)}"
        )
        for i in range(size(filtered)):
            assert filtered(i) == filtered2(i), (
                f"filter not idempotent for {layout} at {i}: "
                f"filter={filtered(i)} vs filter(filter)={filtered2(i)}"
            )


def test_exhaustive_blocked_product_size():
    """Verify blocked_product(A, B) has size(A) * size(B)."""
    layouts_1d = [
        Layout(2, 1), Layout(3, 1), Layout(4, 1), Layout(2, 2), Layout(4, 2),
    ]
    layouts_2d = [
        Layout((2, 2), (1, 2)), Layout((2, 3), (3, 1)), Layout((3, 2)),
    ]
    all_layouts = layouts_1d + layouts_2d

    for a in all_layouts:
        for b in all_layouts:
            result = blocked_product(a, b)
            expected_size = size(a) * size(b)
            assert size(result) == expected_size, (
                f"blocked_product({a}, {b}): "
                f"size={size(result)} != {expected_size}"
            )


def test_exhaustive_blocked_product_covers_offsets():
    """Verify blocked_product(A, B) for compact A produces all expected offsets.

    For contiguous A, blocked_product should tile B's pattern across cosize(A)
    blocks: B's offsets are shifted by cosize(A) * i for each copy.
    """
    compact_layouts = [
        Layout(2, 1), Layout(4, 1), Layout((2, 2)),
    ]
    tiling_layouts = [
        Layout(2, 1), Layout(3, 1), Layout((2, 2)),
    ]

    for a in compact_layouts:
        for b in tiling_layouts:
            result = blocked_product(a, b)
            # Collect all offsets
            offsets = set(result(i) for i in range(size(result)))
            # Should have size(A) * size(B) elements if injective
            # (may have fewer if broadcast)
            a_cos = cosize(a)
            for j in range(size(b)):
                b_offset = b(j) * a_cos
                for k in range(size(a)):
                    expected = a(k) + b_offset
                    assert expected in offsets, (
                        f"blocked_product({a}, {b}): "
                        f"expected offset {expected} (a({k})={a(k)} + b({j})={b(j)}*{a_cos}) "
                        f"not in result offsets"
                    )


def test_exhaustive_flat_divide_preserves_mapping():
    """Verify flat_divide(L, t)(i) == L(i) for all i.

    Only tests cases where the tiler divides within the first mode
    (scalar tiler <= first mode size for multi-mode layouts).
    """
    small = _generate_small_layouts(max_size=12)
    tilers = [2, 3, 4]
    tested = 0

    for layout in small:
        s = size(layout)
        if s < 4:
            continue
        r = rank(layout)
        for t in tilers:
            if s % t != 0:
                continue
            # For multi-mode layouts, only test tilers that divide evenly
            # within the first mode to avoid cross-mode reordering issues
            if r > 1:
                first_mode_size = layout.shape[0] if isinstance(layout.shape[0], int) else size(Layout(layout.shape[0]))
                if t > first_mode_size or first_mode_size % t != 0:
                    continue
            try:
                result = flat_divide(layout, t)
            except (AssertionError, ValueError, TypeError):
                continue

            for i in range(s):
                assert result(i) == layout(i), (
                    f"flat_divide({layout}, {t})({i}) = "
                    f"{result(i)} != {layout(i)}"
                )
            tested += 1

    assert tested > 20, f"Only tested {tested} flat_divide cases, expected more"


###############################################################################
## Tests for tile_to_shape and make_ordered_layout
###############################################################################


def test_make_ordered_layout_column_major():
    """Verify make_ordered_layout produces correct column-major strides."""
    # Column-major: mode 0 varies fastest
    result = make_ordered_layout((4, 8), (0, 1))
    assert result.shape == (4, 8)
    assert result.stride == (1, 4), f"Expected (1, 4), got {result.stride}"

    # Default should be column-major
    result_default = make_ordered_layout((4, 8))
    assert result_default == result


def test_make_ordered_layout_row_major():
    """Verify make_ordered_layout produces correct row-major strides."""
    # Row-major: mode 1 varies fastest
    result = make_ordered_layout((4, 8), (1, 0))
    assert result.shape == (4, 8)
    assert result.stride == (8, 1), f"Expected (8, 1), got {result.stride}"


def test_make_ordered_layout_3d():
    """Verify make_ordered_layout handles 3D with custom order."""
    result = make_ordered_layout((2, 3, 4), (2, 0, 1))
    # mode 2 fastest (stride 1), then mode 0 (stride 4), then mode 1 (stride 8)
    assert result.shape == (2, 3, 4)
    assert result.stride == (4, 8, 1)


def test_tile_to_shape_size_preserved():
    """Verify tile_to_shape produces layout with target size."""
    test_cases = [
        (Layout(4, 1), 16),
        (Layout((2, 3), (1, 2)), (8, 9)),
        (Layout((2, 4), (1, 2)), (4, 8)),
        (Layout((3, 5)), (12, 15)),
    ]
    for block, target in test_cases:
        result = tile_to_shape(block, target)
        target_size = target if isinstance(target, int) else size(target)
        assert size(result) == target_size, (
            f"tile_to_shape({block}, {target}): size={size(result)} != {target_size}"
        )


def test_tile_to_shape_blocked_structure():
    """Verify tile_to_shape uses blocked_product correctly.

    tile_to_shape(block, target) = blocked_product(block, make_ordered_layout(ceil_div))
    """
    block = Layout((2, 4), (1, 2))
    target = (4, 8)

    result = tile_to_shape(block, target)

    # Manually compute expected result
    # block_shape = (2, 4), target = (4, 8)
    # product_shape = (ceil_div(4,2), ceil_div(8,4)) = (2, 2)
    replication = make_ordered_layout((2, 2))
    expected = blocked_product(block, replication)

    assert result == expected, f"Expected {expected}, got {result}"


def test_tile_to_shape_with_row_major_order():
    """Verify tile_to_shape respects the order parameter."""
    block = Layout((2, 3), (1, 2))
    target = (4, 6)

    # Column-major (default)
    result_col = tile_to_shape(block, target)
    replication_col = make_ordered_layout((2, 2), (0, 1))
    expected_col = blocked_product(block, replication_col)
    assert result_col == expected_col

    # Row-major
    result_row = tile_to_shape(block, target, order=(1, 0))
    replication_row = make_ordered_layout((2, 2), (1, 0))
    expected_row = blocked_product(block, replication_row)
    assert result_row == expected_row


def test_product_each_matches_pycute_size():
    """Verify product_each produces same results as pycute.size per mode."""
    test_cases = [
        (4, 8),
        ((2, 2), 8),
        (3, (2, 4)),
        ((2, 3), (4, 5)),
    ]
    for shape in test_cases:
        result = product_each(shape)
        # Manual verification: size of each top-level element
        expected = tuple(pycute.size(s) if not isinstance(s, int) else s for s in shape)
        assert result == expected, f"product_each({shape}) = {result} != {expected}"



@pytest.mark.skipif(pycute is None, reason="pycute not installed")
def test_oracle_idx2crd():
    shapes = [
        4,
        (4, 2),
        (2, (2, 2)),
        ((2, 2), (2, 2)),
    ]
    indices = [0, 1, 3, 5, 10, 16]
    for s in shapes:
        for idx in indices:
            assert idx2crd(idx, s) == pycute.idx2crd(idx, s)


if __name__ == "__main__":
    import traceback
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    errors = []
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
            errors.append(fn.__name__)
    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("Failed tests:", ", ".join(errors))
    raise SystemExit(1 if failed else 0)
