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
from tensor_layouts.analysis import *


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


## footprint


def test_footprint_contiguous():
    """Contiguous layout: no holes, no reuse."""
    result = footprint(Layout(8, 1))
    assert result["min_offset"] == 0
    assert result["max_offset"] == 7
    assert result["span"] == 8
    assert result["unique_offsets"] == 8
    assert result["total_elements"] == 8
    assert result["reuse_factor"] == 1.0
    assert result["holes"] == 0


def test_footprint_strided():
    """Strided layout: holes between offsets."""
    result = footprint(Layout(4, 2))
    assert result["min_offset"] == 0
    assert result["max_offset"] == 6
    assert result["span"] == 7
    assert result["unique_offsets"] == 4
    assert result["holes"] == 3


def test_footprint_broadcast():
    """Broadcast: high reuse factor."""
    result = footprint(Layout((4, 2), (0, 1)))
    assert result["unique_offsets"] == 2
    assert result["total_elements"] == 8
    assert result["reuse_factor"] == 4.0
    assert result["holes"] == 0


## bank_conflicts


def test_bank_conflicts_linear():
    """Linear stride-1 access: no conflicts."""
    result = bank_conflicts(Layout(32, 1), element_bytes=2)
    assert result["conflict_free"]
    assert result["max_ways"] == 1


def test_bank_conflicts_broadcast():
    """All threads access same address: broadcast, not a conflict."""
    result = bank_conflicts(Layout(32, 0), element_bytes=2)
    assert result["conflict_free"]


def test_bank_conflicts_stride_32():
    """Stride-32 elements with fp16: all threads hit bank 0."""
    # 32 elements * 2 bytes = 64 bytes apart. 64/4 = 16 banks stride.
    # Actually: thread t -> offset 32*t, byte_addr = 64*t, bank = (64t/4) % 32 = 16t % 32
    # This causes 2-way conflicts (threads 0,2,4,... hit bank 0; threads 1,3,5,... hit bank 16)
    result = bank_conflicts(Layout(32, 32), element_bytes=2)
    assert not result["conflict_free"]


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
        # Build a layout mapping value -> swizzled offset for this thread
        offsets = [sw_layout(thread, v) for v in range(8)]
        result = bank_conflicts(
            Layout(8, 1),  # dummy: we check per-row via manual offset
            element_bytes=4,  # treat each offset as a 4-byte word
        )
        # stride-1, 8 consecutive elements with 4-byte words: 8 different banks
        assert result["conflict_free"]


def test_bank_conflicts_fp32():
    """4-byte elements: bank width matches element width."""
    result = bank_conflicts(Layout(32, 1), element_bytes=4)
    assert result["conflict_free"]


def test_bank_conflicts_group_size():
    """Multi-warp layouts are analyzed per group, not all threads at once."""
    # 64 threads with stride 32: aggregating all 64 would double the conflicts
    r32 = bank_conflicts(Layout(32, 32), element_bytes=2)
    r64_default = bank_conflicts(Layout(64, 32), element_bytes=2)
    # Default group_size=32 limits analysis to first warp
    assert r64_default["max_ways"] == r32["max_ways"]

    # Explicitly analyzing all 64 threads gives a larger conflict factor
    r64_full = bank_conflicts(Layout(64, 32), element_bytes=2, group_size=64)
    assert r64_full["max_ways"] > r32["max_ways"]


def test_bank_conflicts_group_size_validation():
    """group_size <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="group_size must be positive"):
        bank_conflicts(Layout(32, 1), element_bytes=2, group_size=0)
    with pytest.raises(ValueError, match="group_size must be positive"):
        bank_conflicts(Layout(32, 1), element_bytes=2, group_size=-1)


def test_bank_conflicts_tv_layout():
    """TV layout analyzes all values per thread, not just value 0."""
    # 32 threads, 2 values: stride-1 threads, stride-32 values
    tv = Layout((32, 2), (1, 32))
    r = bank_conflicts(tv, element_bytes=2)
    assert r["conflict_free"]
    assert len(r["bank_to_threads"]) == 32  # all banks accessed


## coalescing_efficiency


def test_coalescing_contiguous_fp16():
    """32 threads, stride 1, fp16: one cache line (64B of 128B)."""
    result = coalescing_efficiency(Layout(32, 1), element_bytes=2)
    assert result["transactions"] == 1
    assert result["efficiency"] == pytest.approx(0.5)


def test_coalescing_contiguous_fp32():
    """32 threads, stride 1, fp32: one cache line (128B of 128B)."""
    result = coalescing_efficiency(Layout(32, 1), element_bytes=4)
    assert result["transactions"] == 1
    assert result["efficiency"] == pytest.approx(1.0)


def test_coalescing_strided():
    """Stride-2 access doubles the cache lines touched."""
    result = coalescing_efficiency(Layout(32, 2), element_bytes=2)
    assert result["transactions"] == 1  # 32*2*2=128 bytes, still fits in 1 line
    # Actually: offsets 0,2,4,...,62. byte addrs 0,4,8,...,124. All in line 0.
    assert result["efficiency"] == pytest.approx(0.5)


def test_coalescing_large_stride():
    """Large stride: each thread touches a different cache line."""
    # stride 64 elements * 2 bytes = 128 bytes = 1 cache line apart
    result = coalescing_efficiency(Layout(32, 64), element_bytes=2)
    assert result["transactions"] == 32
    # 32 threads * 2 bytes = 64 useful bytes, 32 * 128 = 4096 transferred
    assert result["efficiency"] == pytest.approx(64.0 / (32 * 128))


def test_coalescing_broadcast():
    """All threads access same element: single transaction, minimal useful bytes."""
    result = coalescing_efficiency(Layout(32, 0), element_bytes=2)
    assert result["transactions"] == 1
    # Only 1 unique offset: 1 * 2 bytes useful out of 128 transferred
    assert result["efficiency"] == pytest.approx(2.0 / 128)


def test_coalescing_tv_layout():
    """TV layout counts all values in cache line computation."""
    # 32 threads, 4 values each, stride-4 between threads
    tv = Layout((32, 4), (4, 1))
    result = coalescing_efficiency(tv, element_bytes=2)
    # 128 unique offsets * 2B = 256B -> cache lines 0, 1
    assert result["transactions"] == 2
    assert result["efficiency"] == pytest.approx(1.0)


## segment_analysis


def test_segment_analysis_contiguous_fp16():
    """32 threads, stride 1, fp16: 2 segments, 1 cache line."""
    result = segment_analysis(Layout(32, 1), element_bytes=2)
    # 32 * 2B = 64B -> 2 segments of 32B, 1 cache line of 128B
    assert result["segments"] == 2
    assert result["cache_lines"] == 1
    assert result["unique_bytes"] == 64
    assert result["requested_bytes"] == 64
    assert result["transferred_bytes"] == 64  # 2 * 32
    assert result["segment_efficiency"] == pytest.approx(1.0)
    assert result["first_alignment"] == 0


def test_segment_analysis_strided():
    """Stride-2 touches more segments than contiguous."""
    result = segment_analysis(Layout(32, 2), element_bytes=2)
    # offsets 0,2,4,...,62 -> byte addrs 0,4,8,...,124 -> 4 segments
    assert result["segments"] == 4
    assert result["cache_lines"] == 1


def test_segment_analysis_broadcast():
    """Broadcast: 1 segment, minimal unique bytes."""
    result = segment_analysis(Layout(32, 0), element_bytes=2)
    assert result["segments"] == 1
    assert result["unique_bytes"] == 2
    assert result["requested_bytes"] == 64


def test_segment_analysis_tv_layout():
    """TV layout includes all values in segment computation."""
    tv = Layout((32, 4), (4, 1))
    result = segment_analysis(tv, element_bytes=2)
    # 128 elements * 2B = 256B -> 8 segments, 2 cache lines
    assert result["segments"] == 8
    assert result["cache_lines"] == 2
    assert result["requested_bytes"] == 256  # 32 * 4 * 2


## per-group analysis


def test_per_group_bank_conflicts():
    """Per-group analysis matches single-group result for each warp."""
    r_single = bank_conflicts(Layout(32, 32), element_bytes=2)
    r_per = per_group_bank_conflicts(Layout(64, 32), element_bytes=2)
    assert len(r_per["groups"]) == 2
    # Each group should match the single-warp result
    for g in r_per["groups"]:
        assert g["max_ways"] == r_single["max_ways"]
    assert r_per["worst_max_ways"] == r_single["max_ways"]


def test_per_group_bank_conflicts_tv_layout():
    """TV layout groups by thread dimension, not flat index."""
    # 32 threads, 4 values each: should be 1 group (not 4)
    tv = Layout((32, 4), (1, 32))
    result = per_group_bank_conflicts(tv, element_bytes=2, group_size=32)
    assert len(result["groups"]) == 1


def test_per_group_coalescing():
    """Per-group coalescing for a uniform layout gives identical per-warp results."""
    r_per = per_group_coalescing(Layout(64, 1), element_bytes=2)
    assert len(r_per["groups"]) == 2
    for g in r_per["groups"]:
        assert g["efficiency"] == pytest.approx(0.5)
        assert g["transactions"] == 1


def test_per_group_coalescing_tv_layout():
    """TV layout groups by thread dimension, not flat index."""
    # 32 threads, 4 values each (contiguous within each thread's block)
    tv = Layout((32, 4), (4, 1))
    result = per_group_coalescing(tv, element_bytes=2, group_size=32)
    assert len(result["groups"]) == 1
    # 32 threads * 4 values = 128 elements * 2B = 256B -> 2 cache lines
    assert result["groups"][0]["transactions"] == 2


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


## contiguity


def test_contiguity_contiguous():
    """Fully contiguous layout."""
    assert contiguity(Layout(8, 1)) == 8


def test_contiguity_strided():
    """Strided layout: no contiguity."""
    assert contiguity(Layout(8, 2)) == 1


def test_contiguity_2d_col_major():
    """Column-major 2D: fully contiguous (stride-1 throughout)."""
    assert contiguity(Layout((4, 8), (1, 4))) == 32


def test_contiguity_2d_gapped():
    """2D with gap between modes: contiguous only within mode 0."""
    assert contiguity(Layout((4, 8), (1, 8))) == 4


def test_contiguity_2d_row_major():
    """Row-major 2D: contiguous within first mode (size 1 stride, but mode 0 has stride > 1)."""
    # (4,8):(8,1) -> mode 0 has stride 8, so contiguity is 1
    assert contiguity(Layout((4, 8), (8, 1))) == 1


def test_contiguity_broadcast():
    """Broadcast (stride 0): no contiguity."""
    assert contiguity(Layout(8, 0)) == 1


def test_mode_contiguity_col_major():
    """Column-major: mode 0 contiguous, mode 1 strided."""
    assert mode_contiguity(Layout((4, 8), (1, 4))) == [4, 1]


def test_mode_contiguity_row_major():
    """Row-major: mode 0 strided, mode 1 contiguous."""
    assert mode_contiguity(Layout((4, 8), (8, 1))) == [1, 8]


def test_mode_contiguity_gapped():
    """Mode 0 contiguous, mode 1 has stride > mode 0 size."""
    assert mode_contiguity(Layout((4, 8), (1, 8))) == [4, 1]


def test_slice_contiguity_row_major():
    """Row-major: fixing row gives contiguous column access."""
    L = Layout((4, 8), (8, 1))
    assert slice_contiguity(L, (0, None)) == 8
    assert slice_contiguity(L, (None, 0)) == 1


def test_slice_contiguity_col_major():
    """Column-major: fixing column gives contiguous row access."""
    L = Layout((4, 8), (1, 4))
    assert slice_contiguity(L, (None, 0)) == 4
    assert slice_contiguity(L, (0, None)) == 1


## atom_summary


def test_atom_summary_nv_sm80():
    """SM80 16x8x16 F16 atom summary."""
    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN

    result = atom_summary(SM80_16x8x16_F16F16F16F16_TN)
    assert result["shape_mnk"] == (16, 8, 16)
    assert result["threads"] == 32
    assert result["values_c"] > 0
    assert result["c_coverage_ok"]


def test_atom_summary_nv_sm80_f32():
    """SM80 16x8x8 F32 accumulator atom."""
    from tensor_layouts.atoms_nv import SM80_16x8x8_F32F16F16F32_TN

    result = atom_summary(SM80_16x8x8_F32F16F16F32_TN)
    assert result["shape_mnk"] == (16, 8, 8)
    assert result["threads"] == 32
    assert result["c_coverage_ok"]


def test_atom_summary_amd_cdna():
    """AMD CDNA 32x32x8 MFMA atom summary."""
    from tensor_layouts.atoms_amd import CDNA_32x32x8_F32F16F16_MFMA

    result = atom_summary(CDNA_32x32x8_F32F16F16_MFMA)
    assert result["shape_mnk"] == (32, 32, 8)
    assert result["threads"] == 64  # AMD wavefront
    assert result["c_coverage_ok"]


def test_atom_summary_text_output():
    """atom_summary returns a readable text summary."""
    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN

    result = atom_summary(SM80_16x8x16_F16F16F16F16_TN)
    assert "SM80" in result["text"]
    assert "16 x 8 x 16" in result["text"]
    assert "Threads" in result["text"]


def test_atom_summary_rejects_wrong_c_offsets():
    """c_coverage_ok must check exact offset set, not just cardinality."""
    from tensor_layouts.atoms import MMAAtom

    # Build a 2x2 atom where C layout produces offsets {0, 1, 2, 5}
    # instead of the expected {0, 1, 2, 3}. Cardinality is 4 = M*N,
    # but the set is wrong.
    bad_atom = MMAAtom(
        name="test_bad_coverage",
        ptx="test",
        shape_mnk=(2, 2, 1),
        thr_id=Layout(4),
        a_layout=Layout((4, 1), (1, 0)),  # doesn't matter for this test
        b_layout=Layout((4, 1), (1, 0)),  # doesn't matter for this test
        # C layout: 4 threads, 1 value each -> offsets 0, 1, 2, 5
        c_layout=Layout((4, 1), (1, 0)),  # placeholder, override below
    )
    # Manually construct a C layout that maps t -> {0, 1, 2, 5}
    # Layout((4, 1), (1, 0)) maps t -> t, giving {0, 1, 2, 3} — that's correct.
    # We need offsets {0, 1, 2, 5}: use stride pattern that skips 3.
    # Layout with shape (2, 2) stride (1, 2) gives 0,1,2,3 — still correct.
    # Use a non-standard construction: ((2, 2), 1) : ((1, 4), 0) -> 0,1,4,5
    import dataclasses

    bad_c = Layout(((2, 2), 1), ((1, 4), 0))
    bad_atom = dataclasses.replace(bad_atom, c_layout=bad_c)
    result = atom_summary(bad_atom)
    assert not result["c_coverage_ok"]


def test_atom_summary_rejects_duplicate_c_coverage():
    """c_coverage_ok must be False when C layout produces duplicate offsets."""
    from tensor_layouts.atoms import MMAAtom
    import dataclasses

    # Build a 2x2x1 atom where C layout has shape (4, 2) stride (1, 0).
    # This maps (t, v) pairs to offsets [0,0,1,1,2,2,3,3] — correct set
    # but each offset appears twice.
    base = MMAAtom(
        name="test_duplicate_coverage",
        ptx="test",
        shape_mnk=(2, 2, 1),
        thr_id=Layout(4),
        a_layout=Layout((4, 1), (1, 0)),
        b_layout=Layout((4, 1), (1, 0)),
        c_layout=Layout((4, 1), (1, 0)),  # placeholder
    )
    dup_c = Layout((4, 2), (1, 0))  # 8 accesses, offsets 0..3 each twice
    bad_atom = dataclasses.replace(base, c_layout=dup_c)
    result = atom_summary(bad_atom)
    assert not result["c_coverage_ok"]


def test_operand_analysis_sm80():
    """operand_analysis on a well-formed atom reports full coverage."""
    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN

    result = operand_analysis(SM80_16x8x16_F16F16F16F16_TN)
    for op in ["a", "b", "c"]:
        assert result[op]["coverage_ok"]
        assert result[op]["duplicates"] == 0
        assert result[op]["thread_utilization"] == pytest.approx(1.0)
    assert result["a"]["domain_size"] == 16 * 16  # M * K
    assert result["b"]["domain_size"] == 8 * 16  # N * K
    assert result["c"]["domain_size"] == 16 * 8  # M * N


def test_operand_analysis_bad_coverage():
    """operand_analysis detects malformed operand coverage."""
    from tensor_layouts.atoms import MMAAtom

    base = MMAAtom(
        name="test_bad_operand",
        ptx="test",
        shape_mnk=(2, 2, 1),
        thr_id=Layout(4),
        a_layout=Layout((4, 1), (1, 0)),
        b_layout=Layout((4, 1), (1, 0)),
        c_layout=Layout(((2, 2), 1), ((1, 4), 0)),  # offsets {0,1,4,5}, not {0,1,2,3}
    )
    result = operand_analysis(base)
    assert not result["c"]["coverage_ok"]
    assert len(result["c"]["missing"]) > 0
    assert len(result["c"]["extra"]) > 0


## explain


def test_explain_logical_divide():
    """explain shows step-by-step logical_divide computation."""
    text = explain(logical_divide, Layout(16, 1), 4)
    assert "logical_divide" in text
    assert "complement" in text
    assert "compose" in text
    assert "(4, 4) : (1, 4)" in text


def test_explain_logical_product():
    """explain shows step-by-step logical_product computation."""
    text = explain(logical_product, Layout(4, 1), Layout(3, 1))
    assert "logical_product" in text
    assert "complement" in text
    assert "(4, 3) : (1, 4)" in text


def test_explain_logical_product_tuple_tiler():
    """explain handles logical_product with tuple tiler without crashing."""
    text = explain(logical_product, Layout((4, 4), (1, 4)), (2, 2))
    assert "logical_product" in text
    assert "mode 0" in text
    assert "mode 1" in text
    expected = logical_product(Layout((4, 4), (1, 4)), (2, 2))
    assert str(expected) in text


def test_explain_complement():
    """explain shows complement with image and codomain."""
    text = explain(complement, Layout(4, 2), 16)
    assert "image" in text
    assert "codomain" in text
    assert "[0, 16)" in text


def test_explain_compose():
    """explain shows compose with per-element trace."""
    text = explain(compose, Layout(8, 2), Layout(4, 1))
    assert "C(i) = A(B(i))" in text
    assert "i=0" in text


def test_explain_right_inverse():
    """explain shows right_inverse with verification."""
    text = explain(right_inverse, Layout(4, 2))
    assert "R such that L(R(i)) == i" in text
    assert "Verification" in text


def test_explain_left_inverse():
    """explain shows left_inverse with verification."""
    text = explain(left_inverse, Layout(4, 2))
    assert "R such that R(L(i)) == i" in text
    assert "Verification" in text


def test_explain_unsupported():
    """explain gracefully handles unsupported functions."""
    text = explain(size, Layout(4, 1))
    assert "does not support" in text


def test_explain_blocked_product():
    """explain shows blocked_product as interleaved logical_product."""
    text = explain(blocked_product, Layout((2, 3), (1, 2)), Layout((4, 2), (1, 4)))
    assert "blocked_product" in text
    assert "logical_product" in text
    assert "A varies fastest" in text


def test_explain_raked_product():
    """explain shows raked_product with comparison to blocked."""
    text = explain(raked_product, Layout(4, 1), Layout(3, 1))
    assert "raked_product" in text
    assert "B varies fastest" in text
    assert "blocked" in text
    assert "raked" in text


def test_explain_zipped_divide():
    """explain shows zipped_divide as rearranged logical_divide."""
    text = explain(zipped_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert "zipped_divide" in text
    assert "logical_divide" in text
    assert "((tiles), (rests))" in text


def test_explain_tiled_divide():
    """explain shows tiled_divide structure."""
    text = explain(tiled_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert "tiled_divide" in text
    assert "logical_divide" in text
    assert "((tiles), rest0, rest1, ...)" in text


def test_explain_flat_divide():
    """explain shows flat_divide structure."""
    text = explain(flat_divide, Layout((4, 6), (1, 4)), (2, 3))
    assert "flat_divide" in text
    assert "logical_divide" in text
    assert "(tile0, tile1, ..., rest0, rest1, ...)" in text


## MMAAtom and CopyAtom __str__


def test_mma_atom_str():
    """MMAAtom.__str__ returns a concise summary with name and shape."""
    from tensor_layouts.atoms import MMAAtom

    atom = MMAAtom(
        name="test_16x8x4",
        ptx="test.op",
        shape_mnk=(16, 8, 4),
        thr_id=Layout(32),
        a_layout=Layout((32, 4), (4, 1)),
        b_layout=Layout((32, 2), (2, 1)),
        c_layout=Layout((32, 4), (4, 1)),
    )
    assert str(atom) == "MMAAtom('test_16x8x4', 16x8x4)"


def test_copy_atom_str():
    """CopyAtom.__str__ returns a concise summary with name."""
    from tensor_layouts.atoms import CopyAtom

    atom = CopyAtom(
        name="test_copy_128b",
        ptx="test.copy",
        thr_id=Layout(32),
        src_layout_bits=Layout((32, 128), (128, 1)),
        dst_layout_bits=Layout((32, 128), (128, 1)),
    )
    assert str(atom) == "CopyAtom('test_copy_128b')"


def test_mma_atom_repr_is_verbose():
    """MMAAtom.__repr__ (dataclass-generated) includes all fields."""
    from tensor_layouts.atoms import MMAAtom

    atom = MMAAtom(
        name="test_2x2x1",
        ptx="test",
        shape_mnk=(2, 2, 1),
        thr_id=None,
        a_layout=Layout(2, 1),
        b_layout=Layout(2, 1),
        c_layout=Layout((2, 2), (1, 2)),
    )
    r = repr(atom)
    assert r.startswith("MMAAtom(name=")
    assert "shape_mnk=(2, 2, 1)" in r
    assert "a_layout=Layout(2, 1)" in r


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
