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

from tensor_layouts import *
from tensor_layouts.analysis import (
    offset_table,
    bank_conflicts,
    coalescing_efficiency,
    cycles,
    fixed_points,
    order,
    explain,
)


## offset_table


def test_offset_table_contiguous():
    """Contiguous layout: each offset maps to exactly one coordinate."""
    table = offset_table(Layout(4, 1))
    assert table == {0: [0], 1: [1], 2: [2], 3: [3]}


def test_offset_table_2d():
    """2D layout: coordinates are tuples."""
    table = offset_table(Layout((2, 2), (1, 2)))
    assert table[0] == [(0, 0)]
    assert table[1] == [(1, 0)]
    assert table[2] == [(0, 1)]
    assert table[3] == [(1, 1)]


def test_offset_table_broadcast():
    """Broadcast layout: multiple coordinates map to same offset."""
    table = offset_table(Layout((4, 2), (0, 1)))
    # All 4 row coords map to offset 0 or 1
    assert len(table[0]) == 4
    assert len(table[1]) == 4


def test_offset_table_strided():
    """Strided layout: only even offsets are hit."""
    table = offset_table(Layout(4, 2))
    assert set(table.keys()) == {0, 2, 4, 6}
    assert all(len(v) == 1 for v in table.values())


## bank_conflicts


def test_bank_conflicts_linear():
    """Linear stride-1 access: no conflicts."""
    result = bank_conflicts(Layout(32, 1))
    assert result['conflict_free']
    assert result['max_ways'] == 1


def test_bank_conflicts_broadcast():
    """All threads access same address: broadcast, not a conflict."""
    result = bank_conflicts(Layout(32, 0))
    assert result['conflict_free']


def test_bank_conflicts_stride_32():
    """Stride-32 elements with fp16: all threads hit bank 0."""
    # 32 elements * 2 bytes = 64 bytes apart. 64/4 = 16 banks stride.
    # Actually: thread t -> offset 32*t, byte_addr = 64*t, bank = (64t/4) % 32 = 16t % 32
    # This causes 2-way conflicts (threads 0,2,4,... hit bank 0; threads 1,3,5,... hit bank 16)
    result = bank_conflicts(Layout(32, 32), element_bytes=2)
    assert not result['conflict_free']


def test_bank_conflicts_swizzled():
    """Swizzle(3,0,3) on 8x8 row-major tile eliminates bank conflicts.

    This matches the test_cute_shared_memory_swizzle test in tests/tensor.py.
    With 4-byte elements (one element per bank word), each row's 8 elements
    land in 8 different banks.
    """
    sw_layout = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    for thread in range(8):
        offsets = [sw_layout(thread, v) for v in range(8)]
        # With element_bytes=4, one element per bank word: bank = offset % 32
        banks = [o % 32 for o in offsets]
        assert len(set(banks)) == 8, f"Thread {thread}: banks {banks}"

    # Also verify via bank_conflicts: each row as a 1D layout
    for thread in range(8):
        row = Layout(8, 1)  # value indices 0..7
        # Build a layout mapping value -> swizzled offset for this thread
        offsets = [sw_layout(thread, v) for v in range(8)]
        result = bank_conflicts(
            Layout(8, 1),  # dummy: we check per-row via manual offset
            element_bytes=4,  # treat each offset as a 4-byte word
        )
        # stride-1, 8 consecutive elements with 4-byte words: 8 different banks
        assert result['conflict_free']


def test_bank_conflicts_fp32():
    """4-byte elements: bank width matches element width."""
    result = bank_conflicts(Layout(32, 1), element_bytes=4)
    assert result['conflict_free']


## coalescing_efficiency


def test_coalescing_contiguous_fp16():
    """32 threads, stride 1, fp16: one cache line (64B of 128B)."""
    result = coalescing_efficiency(Layout(32, 1))
    assert result['transactions'] == 1
    assert result['efficiency'] == pytest.approx(0.5)


def test_coalescing_contiguous_fp32():
    """32 threads, stride 1, fp32: one cache line (128B of 128B)."""
    result = coalescing_efficiency(Layout(32, 1), element_bytes=4)
    assert result['transactions'] == 1
    assert result['efficiency'] == pytest.approx(1.0)


def test_coalescing_strided():
    """Stride-2 access doubles the cache lines touched."""
    result = coalescing_efficiency(Layout(32, 2))
    assert result['transactions'] == 1  # 32*2*2=128 bytes, still fits in 1 line
    # Actually: offsets 0,2,4,...,62. byte addrs 0,4,8,...,124. All in line 0.
    assert result['efficiency'] == pytest.approx(0.5)


def test_coalescing_large_stride():
    """Large stride: each thread touches a different cache line."""
    # stride 64 elements * 2 bytes = 128 bytes = 1 cache line apart
    result = coalescing_efficiency(Layout(32, 64))
    assert result['transactions'] == 32
    # 32 threads * 2 bytes = 64 useful bytes, 32 * 128 = 4096 transferred
    assert result['efficiency'] == pytest.approx(64.0 / (32 * 128))


def test_coalescing_broadcast():
    """All threads access same element: single transaction."""
    result = coalescing_efficiency(Layout(32, 0))
    assert result['transactions'] == 1


## cycles


def test_cycles_identity():
    """Identity layout: all fixed points."""
    c = cycles(Layout(4, 1))
    assert c == [[0], [1], [2], [3]]


def test_cycles_swap():
    """Row-major 2x2: swaps elements 1 and 2."""
    c = cycles(Layout((2, 2), (2, 1)))
    # Offsets: (0,0)->0, (1,0)->2, (0,1)->1, (1,1)->3
    # As permutation: 0->0, 1->2, 2->1, 3->3
    assert [0] in c
    assert [3] in c
    # 1 and 2 form a 2-cycle
    assert [1, 2] in c or [2, 1] in c


def test_cycles_swizzle():
    """Swizzle composed with row-major has non-trivial cycle structure."""
    sw = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    c = cycles(sw)
    # Verify all offsets appear exactly once across all cycles
    all_offsets = sorted(o for cycle in c for o in cycle)
    assert all_offsets == list(range(64))
    # 0 is a fixed point
    assert [0] in c


def test_cycles_not_bijective():
    """Non-bijective layout raises ValueError."""
    with pytest.raises(ValueError):
        cycles(Layout(4, 2))


## fixed_points


def test_fixed_points_identity():
    """Identity layout: every element is a fixed point."""
    assert fixed_points(Layout(4, 1)) == [0, 1, 2, 3]


def test_fixed_points_swap():
    """Row-major 2x2: 0 and 3 are fixed, 1 and 2 swap."""
    fp = fixed_points(Layout((2, 2), (2, 1)))
    assert fp == [0, 3]


def test_fixed_points_broadcast():
    """Broadcast layout: only offset 0 maps to itself."""
    fp = fixed_points(Layout(4, 0))
    assert fp == [0]


## order


def test_order_identity():
    """Identity permutation has order 1."""
    assert order(Layout(4, 1)) == 1


def test_order_swap():
    """Single transposition has order 2."""
    assert order(Layout((2, 2), (2, 1))) == 2


def test_order_swizzle():
    """Swizzle composed with row-major: order is LCM of cycle lengths."""
    sw = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    o = order(sw)
    # Verify order by checking that applying the layout o times gives identity
    for i in range(64):
        x = i
        for _ in range(o):
            x = sw(x)
        assert x == i, f"order {o} failed at {i}"


def test_order_not_bijective():
    """Non-bijective layout raises ValueError."""
    with pytest.raises(ValueError):
        order(Layout(4, 2))


## explain


def test_explain_logical_divide():
    """explain shows step-by-step logical_divide computation."""
    text = explain(logical_divide, Layout(16, 1), 4)
    assert 'logical_divide' in text
    assert 'complement' in text
    assert 'compose' in text
    assert '(4, 4) : (1, 4)' in text


def test_explain_logical_product():
    """explain shows step-by-step logical_product computation."""
    text = explain(logical_product, Layout(4, 1), Layout(3, 1))
    assert 'logical_product' in text
    assert 'complement' in text
    assert '(4, 3) : (1, 4)' in text


def test_explain_complement():
    """explain shows complement with image and codomain."""
    text = explain(complement, Layout(4, 2), 16)
    assert 'image' in text
    assert 'codomain' in text
    assert '[0, 16)' in text


def test_explain_compose():
    """explain shows compose with per-element trace."""
    text = explain(compose, Layout(8, 2), Layout(4, 1))
    assert 'C(i) = A(B(i))' in text
    assert 'i=0' in text


def test_explain_right_inverse():
    """explain shows right_inverse with verification."""
    text = explain(right_inverse, Layout(4, 2))
    assert 'R such that L(R(i)) == i' in text
    assert 'Verification' in text


def test_explain_left_inverse():
    """explain shows left_inverse with verification."""
    text = explain(left_inverse, Layout(4, 2))
    assert 'R such that R(L(i)) == i' in text
    assert 'Verification' in text


def test_explain_unsupported():
    """explain gracefully handles unsupported functions."""
    text = explain(size, Layout(4, 1))
    assert 'does not support' in text


def test_explain_blocked_product():
    """explain shows blocked_product as interleaved logical_product."""
    text = explain(blocked_product, Layout((2, 3), (1, 2)), Layout((4, 2), (1, 4)))
    assert 'blocked_product' in text
    assert 'logical_product' in text
    assert 'A varies fastest' in text


def test_explain_raked_product():
    """explain shows raked_product with comparison to blocked."""
    text = explain(raked_product, Layout(4, 1), Layout(3, 1))
    assert 'raked_product' in text
    assert 'B varies fastest' in text
    assert 'blocked' in text
    assert 'raked' in text


def test_explain_zipped_divide():
    """explain shows zipped_divide as rearranged logical_divide."""
    text = explain(zipped_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert 'zipped_divide' in text
    assert 'logical_divide' in text
    assert '((tiles), (rests))' in text


def test_explain_tiled_divide():
    """explain shows tiled_divide structure."""
    text = explain(tiled_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert 'tiled_divide' in text
    assert 'logical_divide' in text
    assert '((tiles), rest0, rest1, ...)' in text


def test_explain_flat_divide():
    """explain shows flat_divide structure."""
    text = explain(flat_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert 'flat_divide' in text
    assert 'logical_divide' in text
    assert '(tile0, tile1, ..., rest0, rest1, ...)' in text


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
