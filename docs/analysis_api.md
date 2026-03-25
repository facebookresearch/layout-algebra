<!--
MIT License

Copyright (c) 2026 Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Analysis API

GPU kernel performance lives or dies by memory access patterns.  Two
hardware constraints dominate:

1. **Shared memory bank conflicts.**  Shared memory is divided into 32
   banks, each 4 bytes wide.  When threads in a warp simultaneously
   access different words in the same bank, the accesses serialize ---
   turning one cycle into N cycles.  Swizzling rearranges offsets to
   spread accesses across banks.

2. **Global memory coalescing.**  The memory controller fetches 128-byte
   cache lines.  When a warp's 32 threads access a contiguous region,
   one transaction suffices.  Scattered accesses fetch multiple cache
   lines, wasting bandwidth.

The `tensor_layouts.analysis` module lets you quantify both, inspect
aliasing via inverse mappings, study the permutation structure of
bijective layouts, and trace the algebra step by step.

```python
from tensor_layouts.analysis import (
    offset_table, bank_conflicts, coalescing_efficiency,
    cycles, fixed_points, order, explain,
)
```

---

## offset_table(layout)

Inverse mapping: `{offset: [coord, ...]}`.  Reveals aliasing --- when
multiple coordinates map to the same offset, they share the same memory
location.  This answers "who writes to address X?"

```python
# Contiguous: each offset has exactly one coordinate
offset_table(Layout(4, 1))
# {0: [0], 1: [1], 2: [2], 3: [3]}

# Broadcast (stride 0): four coordinates alias to each offset
offset_table(Layout((4, 2), (0, 1)))
# {0: [(0,0), (1,0), (2,0), (3,0)],
#  1: [(0,1), (1,1), (2,1), (3,1)]}
```

## bank_conflicts(layout, *, element_bytes, num_banks=32, bank_width_bytes=4, group_size=32)

Analyze shared memory bank conflicts for a thread-to-offset layout.

Only the first `group_size` threads are analyzed, matching the hardware
issue granularity (warp on NVIDIA, wavefront on AMD).  This avoids
overstating conflicts when the layout spans multiple warps.  Pass
`group_size=64` for AMD wavefronts.

Consider an 8x8 row-major tile in shared memory.  Reading rows is fast
(stride 1 hits consecutive banks), but reading a column means stride-8
access --- threads land in the same banks and serialize:

```python
# 8 threads reading a column of an 8x8 tile (stride 8, 4-byte elements).
# Threads land in only 4 of 32 banks -> 2-way conflict.
result = bank_conflicts(Layout(8, 8), element_bytes=4)
result['conflict_free']  # False
result['max_ways']       # 2

# Swizzle fixes it: each thread now hits a different bank.
sw = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
col_layout = sw(None, 0)   # slice column 0: maps thread -> swizzled offset
result = bank_conflicts(col_layout, element_bytes=4)
result['conflict_free']  # True
```

The `max_ways` value is the worst-case serialization factor: 1 means no
conflicts, N means N-way serialization.  Two threads accessing the
*same* word get a broadcast (no conflict on NVIDIA hardware).

For multi-mode (TV) layouts where mode 0 is the thread dimension and
mode 1+ are value dimensions, all values per thread are included in the
analysis.  This models vectorized loads where each thread accesses
multiple elements:

```python
# TV layout: 32 threads, each loading 2 fp16 elements
tv = Layout((32, 2), (1, 32))
result = bank_conflicts(tv, element_bytes=2)
result['conflict_free']  # True: values land in distinct banks
```

Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `conflict_free` | bool | True if `max_ways` is 1 |
| `max_ways` | int | Worst-case serialization factor across all banks |
| `bank_to_threads` | dict | `{bank_id: [thread_ids...]}` for all accessed banks |

## coalescing_efficiency(layout, *, element_bytes, warp_size=32, cache_line_bytes=128)

Analyze global memory coalescing for a thread-to-offset layout.

When 32 threads access 32 consecutive fp32 elements, everything fits in
one 128-byte cache line --- perfect coalescing.  But if each thread
strides 64 elements apart, each access triggers a separate transaction:

```python
# Perfectly coalesced: 32 threads, stride 1, fp32
result = coalescing_efficiency(Layout(32, 1), element_bytes=4)
result['transactions']  # 1
result['efficiency']    # 1.0  (128 unique useful bytes / 128 transferred)

# Worst case: each thread hits a separate cache line
result = coalescing_efficiency(Layout(32, 64), element_bytes=2)
result['transactions']  # 32
result['efficiency']    # 0.016  (64 unique useful bytes / 4096 transferred)
```

Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `transactions` | int | Number of cache line fetches needed |
| `efficiency` | float | Unique useful bytes / transferred bytes (1.0 = perfect) |
| `cache_lines` | list | Sorted cache line indices touched |

For multi-mode (TV) layouts, all values per thread are included,
modeling vectorized loads:

```python
# TV layout: 32 threads, 4 values each, contiguous within each thread
tv = Layout((32, 4), (4, 1))
result = coalescing_efficiency(tv, element_bytes=2)
result['transactions']  # 2  (256 bytes spans 2 cache lines)
result['efficiency']    # 1.0  (256 unique bytes / 256 transferred)
```

## Permutation Analysis

When a layout is bijective (every offset is hit exactly once), it defines
a permutation.  Understanding its cycle structure reveals how data moves
through memory --- transpositions, rotations, and fixed points all have
distinct performance implications.

### cycles(layout)

Decompose the permutation into disjoint cycles.  Fixed points (elements
that map to themselves) appear as length-1 cycles.

Raises `ValueError` if the layout is not bijective.

```python
# Row-major 3x2: the transpose permutation on a 3x2 matrix.
# Flat index i maps to offset 2*(i%3) + i//3.
rm = Layout((3, 2), (2, 1))
# 0->0, 1->2, 2->4, 3->1, 4->3, 5->5

cycles(rm)
# [[0], [1, 2, 4, 3], [5]]
# Corners 0 and 5 are fixed; the four interior elements form a 4-cycle:
# 1 -> 2 -> 4 -> 3 -> 1
```

### fixed_points(layout)

Return offsets where `layout(i) == i`.  Does not require bijectivity.

```python
fixed_points(Layout((3, 2), (2, 1)))  # [0, 5]
fixed_points(Layout(4, 1))            # [0, 1, 2, 3]  (identity)
```

### order(layout)

The permutation order: smallest `k > 0` such that applying the layout
`k` times returns to the identity.  Equals the LCM of all cycle lengths.

Raises `ValueError` if the layout is not bijective.

```python
order(Layout(4, 1))              # 1  (identity)
order(Layout((2, 2), (2, 1)))    # 2  (single transposition)
order(Layout((3, 2), (2, 1)))    # 4  (has a 4-cycle)
```

## contiguity(layout)

The longest contiguous vector width from the start of a layout.  Counts
how many consecutive elements starting from flat index 0 map to
consecutive memory offsets.  This tells you the maximum vectorized
load/store width.

```python
contiguity(Layout(8, 1))            # 8  (fully contiguous)
contiguity(Layout(8, 2))            # 1  (strided, no contiguity)
contiguity(Layout((4, 8), (1, 8)))  # 4  (contiguous within mode 0)
```

## atom_summary(atom)

Human-readable summary of an MMA atom's key properties.  Works with
both NVIDIA and AMD atoms.

```python
from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
atom_summary(SM80_16x8x16_F16F16F16F16_TN)
# SM80_16x8x16_F16F16F16F16_TN
#   Shape (M, N, K): 16 x 8 x 16
#   Threads:          32
#   Values per thread: A=8, B=4, C=4
#   C covers M*N:     True
```

Returns a dict:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Atom name |
| `shape_mnk` | tuple | (M, N, K) logical shape |
| `threads` | int | Number of active threads |
| `values_a` | int | Registers per thread for A operand |
| `values_b` | int | Registers per thread for B operand |
| `values_c` | int | Registers per thread for C accumulator |
| `c_coverage_ok` | bool | True if C layout covers all M*N elements exactly once |
| `a_broadcast` | bool | True if A layout has stride-0 (broadcast) modes |
| `b_broadcast` | bool | True if B layout has stride-0 (broadcast) modes |

## explain(fn, *args)

Show step-by-step how an algebra operation computes its result.  Expands
the mathematical definition with concrete values, showing each
intermediate layout.

This is the "show your work" for the layout algebra.  The four core
operations (compose, complement, divide, product) are tightly
interconnected --- `logical_divide` is defined in terms of `complement`
and `compose`, and `logical_product` is too.  `explain` makes these
connections visible.

Supported operations: `logical_divide`, `logical_product`, `complement`,
`compose`, `right_inverse`, `left_inverse`, `blocked_product`,
`raked_product`, `zipped_divide`, `tiled_divide`, `flat_divide`.

```python
explain(logical_divide, Layout(16, 1), 4)
# logical_divide(16 : 1, 4 : 1)
#   = compose(L, Layout(T, complement(T, size(L))))
#
#   L = 16 : 1
#   T = 4 : 1
#   size(L) = 16
#   complement(T, 16) = 4 : 4
#   Layout(T, complement) = (4, 4) : (1, 4)
#   compose(L, (4, 4) : (1, 4)) = (4, 4) : (1, 4)
#
#   result = (4, 4) : (1, 4)

explain(complement, Layout(4, 2), 16)
# complement(4 : 2, 16)
#   Fills the gaps in L's codomain up to bound=16.
#
#   L = 4 : 2
#   image(L) = [0, 2, 4, 6]
#   codomain = [0, 16)
#   complement = (2, 2) : (1, 8)
#   image(complement) = [0, 1, 8, 9]
```
