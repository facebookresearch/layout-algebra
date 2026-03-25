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

# ruff: noqa: F405

"""GPU layout analysis: bank conflicts, coalescing, permutation structure.

These functions analyze layouts in GPU-specific contexts --- shared memory
bank conflicts, global memory coalescing, and permutation cycle structure.
They build on the core algebra but encode hardware knowledge (bank count,
cache line size, warp width).

    from tensor_layouts.analysis import bank_conflicts, coalescing_efficiency

Import explicitly; these are not re-exported from the top-level package.
"""

from .layouts import *  # noqa: F401,F403
from .atoms import MMAAtom

__all__ = [
    "offset_table",
    "footprint",
    "bank_conflicts",
    "per_group_bank_conflicts",
    "coalescing_efficiency",
    "segment_analysis",
    "per_group_coalescing",
    "cycles",
    "fixed_points",
    "order",
    "contiguity",
    "mode_contiguity",
    "slice_contiguity",
    "atom_summary",
    "operand_analysis",
    "explain",
]


# =============================================================================
# Inverse mapping
# =============================================================================


def offset_table(layout: Layout) -> dict:
    """Return {offset: [coord, ...]} mapping each offset to its coordinates.

    This is the inverse of the layout function.  It immediately reveals
    aliasing: if an offset maps to more than one coordinate, multiple
    logical positions share the same memory location.

    Examples:
        offset_table(Layout(4, 1))
        # {0: [0], 1: [1], 2: [2], 3: [3]}

        offset_table(Layout((4, 2), (0, 1)))
        # {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}  -- broadcast aliasing
    """
    layout = as_layout(layout)
    table = {}
    for i in range(size(layout)):
        coord = idx2crd(i, layout.shape)
        offset = layout(i)
        table.setdefault(offset, []).append(coord)
    return table


def footprint(layout: Layout) -> dict:
    """Summarize the memory footprint of a layout.

    Answers the questions users most often ask about a layout's memory
    behavior: how big is the footprint, how sparse is it, and how much
    aliasing (reuse) is there?

    Args:
        layout: Layout to analyze.

    Returns:
        dict with:
            min_offset: smallest offset produced
            max_offset: largest offset produced
            span: max_offset - min_offset + 1 (contiguous range needed)
            unique_offsets: number of distinct offsets
            total_elements: size(layout) (number of logical elements)
            reuse_factor: total_elements / unique_offsets (1.0 = no aliasing)
            holes: span - unique_offsets (0 = dense, no gaps)

    Examples:
        footprint(Layout(8, 1))
        # {'min_offset': 0, 'max_offset': 7, 'span': 8,
        #  'unique_offsets': 8, 'total_elements': 8,
        #  'reuse_factor': 1.0, 'holes': 0}

        footprint(Layout((4, 2), (0, 1)))  # broadcast
        # {'min_offset': 0, 'max_offset': 1, 'span': 2,
        #  'unique_offsets': 2, 'total_elements': 8,
        #  'reuse_factor': 4.0, 'holes': 0}
    """
    layout = as_layout(layout)
    offsets = image(layout)  # sorted list of unique offsets
    n_unique = len(offsets)
    n_total = size(layout)
    min_off = offsets[0] if offsets else 0
    max_off = offsets[-1] if offsets else 0
    span = max_off - min_off + 1 if offsets else 0

    return {
        "min_offset": min_off,
        "max_offset": max_off,
        "span": span,
        "unique_offsets": n_unique,
        "total_elements": n_total,
        "reuse_factor": n_total / n_unique if n_unique > 0 else 0.0,
        "holes": span - n_unique,
    }


# =============================================================================
# Bank conflict analysis
# =============================================================================


def bank_conflicts(
    layout: Layout,
    *,
    element_bytes: int,
    num_banks: int = 32,
    bank_width_bytes: int = 4,
    group_size: int = 32,
):
    """Analyze shared memory bank conflicts for a thread-to-offset layout.

    Given a layout that maps thread indices to shared memory offsets,
    compute how many bank conflicts occur when a group of threads
    accesses memory simultaneously.

    Shared memory is divided into banks (typically 32).  Two threads
    conflict when they access the same bank but different addresses
    within that bank.  The conflict factor (max_ways) tells you how
    many times the access must serialize.

    Only the first ``group_size`` threads are analyzed, matching the
    hardware issue granularity (warp on NVIDIA, wavefront on AMD).
    This avoids overstating conflicts when the layout spans multiple
    warps.

    For multi-mode (TV) layouts, mode 0 is the thread dimension and all
    remaining modes are value dimensions.  Each thread's accesses across
    all values are included in the analysis, modeling vectorized loads.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        num_banks: Number of shared memory banks (32 on NVIDIA GPUs).
        element_bytes: Size of each element in bytes (2 for fp16).
        bank_width_bytes: Width of each bank in bytes (4 on NVIDIA GPUs).
        group_size: Number of threads analyzed as one simultaneous group
            (32 = one NVIDIA warp, 64 = one AMD wavefront).

    Returns:
        dict with:
            conflict_free: True if no bank conflicts
            max_ways: worst-case serialization factor (1 = no conflicts)
            bank_to_threads: {bank_id: [thread_ids...]} for all accessed banks

    Examples:
        # Linear layout: threads access consecutive fp16 elements
        bank_conflicts(Layout(32, 1), element_bytes=2)
        # {'conflict_free': True, 'max_ways': 1, ...}

        # All threads hit the same address
        bank_conflicts(Layout(32, 0), element_bytes=2)
        # {'conflict_free': True, 'max_ways': 1, ...}  (broadcast, not a conflict)
    """
    layout = as_layout(layout)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    thread_count, value_count = _tv_dimensions(layout)
    n = min(thread_count, group_size)

    # Map each thread to (bank, word_address)
    # A bank conflict occurs when threads access different 4-byte words in the
    # same bank.  Two threads accessing the same word get a broadcast (no conflict).
    thread_banks = {}  # bank -> [(thread_id, word_address), ...]
    for t in range(n):
        for v in range(value_count):
            flat_idx = v * thread_count + t
            offset = layout(flat_idx)
            byte_addr = offset * element_bytes
            word_addr = byte_addr // bank_width_bytes
            bank = word_addr % num_banks
            thread_banks.setdefault(bank, []).append((t, word_addr))

    # Compute conflicts per bank
    # Two threads conflict if they hit the same bank but different addresses.
    # If they hit the same address, it's a broadcast (no conflict on NVIDIA).
    max_ways = 1
    bank_to_threads = {}
    for bank, accesses in thread_banks.items():
        thread_ids = [t for t, _ in accesses]
        bank_to_threads[bank] = thread_ids

        # Group by address within this bank
        addr_groups = {}
        for t, addr in accesses:
            addr_groups.setdefault(addr, []).append(t)

        # Conflict factor = number of distinct addresses in this bank
        ways = len(addr_groups)
        if ways > max_ways:
            max_ways = ways

    return {
        "conflict_free": max_ways <= 1,
        "max_ways": max_ways,
        "bank_to_threads": bank_to_threads,
    }


# =============================================================================
# Coalescing analysis
# =============================================================================


def coalescing_efficiency(
    layout: Layout,
    *,
    element_bytes: int,
    warp_size: int = 32,
    cache_line_bytes: int = 128,
):
    """Analyze global memory coalescing for a thread-to-offset layout.

    Given a layout that maps thread indices to global memory offsets,
    compute how many cache line transactions a warp needs and the
    resulting bandwidth efficiency.

    In the ideal case (fully coalesced), 32 consecutive threads access
    32 consecutive elements within a single cache line, requiring one
    transaction.  In the worst case, each thread triggers a separate
    transaction.

    For multi-mode (TV) layouts, mode 0 is the thread dimension and all
    remaining modes are value dimensions.  Each thread's accesses across
    all values are included in the analysis, modeling vectorized loads.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        warp_size: Threads per warp (32 on NVIDIA/AMD GPUs).
        element_bytes: Size of each element in bytes.
        cache_line_bytes: Cache line size in bytes (128 on NVIDIA GPUs).

    Returns:
        dict with:
            transactions: number of cache line transactions needed
            efficiency: ratio of unique useful bytes to transferred bytes
                (unique = deduplicated across threads; broadcast accesses
                count only once, not once per thread)
            cache_lines: sorted list of cache line indices touched

    Examples:
        # Perfectly coalesced: 32 threads, stride 1, fp16
        coalescing_efficiency(Layout(32, 1), element_bytes=2)
        # {'transactions': 1, 'efficiency': 0.5, ...}  -- 64B used of 128B line

        # Strided access: each thread 2 elements apart, fp32
        coalescing_efficiency(Layout(32, 2), element_bytes=4)
        # {'transactions': 2, 'efficiency': 0.5, ...}
    """
    layout = as_layout(layout)
    thread_count, value_count = _tv_dimensions(layout)
    n = min(thread_count, warp_size)

    # Find which cache lines are touched and count unique offsets
    cache_lines = set()
    unique_offsets = set()
    for t in range(n):
        for v in range(value_count):
            flat_idx = v * thread_count + t
            offset = layout(flat_idx)
            unique_offsets.add(offset)
            byte_addr = offset * element_bytes
            cache_line = byte_addr // cache_line_bytes
            cache_lines.add(cache_line)

    transactions = len(cache_lines)
    useful_bytes = len(unique_offsets) * element_bytes
    transferred_bytes = transactions * cache_line_bytes
    efficiency = useful_bytes / transferred_bytes if transferred_bytes > 0 else 0.0

    return {
        "transactions": transactions,
        "efficiency": efficiency,
        "cache_lines": sorted(cache_lines),
    }


def segment_analysis(
    layout: Layout,
    *,
    element_bytes: int,
    warp_size: int = 32,
    segment_bytes: int = 32,
    cache_line_bytes: int = 128,
):
    """Segment- and alignment-aware global memory transaction analysis.

    A more detailed model than ``coalescing_efficiency()``.  NVIDIA GPUs
    transfer memory in 32-byte segments within 128-byte cache lines.  A
    warp access may touch fewer cache lines than segments when accesses
    cluster within a line but span multiple segments.

    For multi-mode (TV) layouts, mode 0 is the thread dimension and all
    remaining modes are value dimensions.  Each thread's accesses across
    all values are included in the analysis, modeling vectorized loads.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        warp_size: Threads per warp.
        element_bytes: Size of each element in bytes.
        segment_bytes: Memory segment size in bytes (32 on NVIDIA GPUs).
        cache_line_bytes: Cache line size in bytes (128 on NVIDIA GPUs).

    Returns:
        dict with:
            segments: number of 32B segments touched
            cache_lines: number of 128B cache lines touched
            unique_bytes: bytes for unique addressed elements
            requested_bytes: bytes requested by all threads (including duplicates)
            transferred_bytes: total bytes transferred (segments * segment_bytes)
            segment_efficiency: unique_bytes / transferred_bytes
            first_byte_addr: byte address of the first thread's access
            first_alignment: alignment of first_byte_addr to segment_bytes
    """
    layout = as_layout(layout)
    thread_count, value_count = _tv_dimensions(layout)
    n = min(thread_count, warp_size)

    segments = set()
    lines = set()
    unique_offsets = set()
    first_byte = None

    for t in range(n):
        for v in range(value_count):
            flat_idx = v * thread_count + t
            offset = layout(flat_idx)
            unique_offsets.add(offset)
            byte_addr = offset * element_bytes
            if first_byte is None:
                first_byte = byte_addr
            segments.add(byte_addr // segment_bytes)
            lines.add(byte_addr // cache_line_bytes)

    first_byte = first_byte if first_byte is not None else 0
    n_segments = len(segments)
    n_lines = len(lines)
    unique_bytes = len(unique_offsets) * element_bytes
    requested_bytes = n * value_count * element_bytes
    transferred_bytes = n_segments * segment_bytes

    return {
        "segments": n_segments,
        "cache_lines": n_lines,
        "unique_bytes": unique_bytes,
        "requested_bytes": requested_bytes,
        "transferred_bytes": transferred_bytes,
        "segment_efficiency": (unique_bytes / transferred_bytes if transferred_bytes > 0 else 0.0),
        "first_byte_addr": first_byte,
        "first_alignment": first_byte % segment_bytes,
    }


# =============================================================================
# Per-group analysis
# =============================================================================


def _tv_dimensions(layout: Layout):
    """Extract (thread_count, value_count) from a layout.

    For rank-1 (scalar shape) layouts: thread_count = size, value_count = 1.
    For rank>1 (TV) layouts: thread_count = size(mode 0), value_count =
    product of remaining modes.
    """
    if is_int(layout.shape):
        return size(layout), 1
    return size(mode(layout, 0)), size(layout) // size(mode(layout, 0))


def per_group_bank_conflicts(
    layout: Layout,
    *,
    element_bytes: int,
    group_size: int = 32,
    num_banks: int = 32,
    bank_width_bytes: int = 4,
) -> dict:
    """Analyze bank conflicts per warp/wavefront group across a full layout.

    Splits the layout into groups of ``group_size`` threads and analyzes
    bank conflicts for each group independently.

    For multi-mode (TV) layouts, groups are formed along mode 0 (the thread
    dimension).  Each thread's accesses across all value modes (mode 1+) are
    included in its group's analysis.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        group_size: Threads per group (32 = NVIDIA warp, 64 = AMD wave).
        num_banks: Number of shared memory banks.
        element_bytes: Size of each element in bytes.
        bank_width_bytes: Width of each bank in bytes.

    Returns:
        dict with:
            groups: list of per-group result dicts (conflict_free, max_ways,
                bank_to_threads)
            worst_group: index of group with highest max_ways
            worst_max_ways: the highest max_ways across all groups
    """
    layout = as_layout(layout)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    thread_count, value_count = _tv_dimensions(layout)
    num_groups = (thread_count + group_size - 1) // group_size

    groups = []
    worst_idx = 0
    worst_ways = 0

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, thread_count)

        thread_banks = {}
        for t in range(start, end):
            for v in range(value_count):
                flat_idx = v * thread_count + t
                offset = layout(flat_idx)
                byte_addr = offset * element_bytes
                word_addr = byte_addr // bank_width_bytes
                bank = word_addr % num_banks
                thread_banks.setdefault(bank, []).append((t, word_addr))

        max_ways = 1
        bank_to_threads = {}
        for bank, accesses in thread_banks.items():
            bank_to_threads[bank] = [t for t, _ in accesses]
            addr_groups = {}
            for t, addr in accesses:
                addr_groups.setdefault(addr, []).append(t)
            ways = len(addr_groups)
            if ways > max_ways:
                max_ways = ways

        result = {
            "conflict_free": max_ways <= 1,
            "max_ways": max_ways,
            "bank_to_threads": bank_to_threads,
        }
        groups.append(result)
        if max_ways > worst_ways:
            worst_ways = max_ways
            worst_idx = g

    return {
        "groups": groups,
        "worst_group": worst_idx,
        "worst_max_ways": worst_ways,
    }


def per_group_coalescing(
    layout: Layout,
    *,
    element_bytes: int,
    group_size: int = 32,
    cache_line_bytes: int = 128,
) -> dict:
    """Analyze coalescing efficiency per warp/wavefront group across a full layout.

    Splits the layout into groups of ``group_size`` threads and analyzes
    coalescing for each group independently.

    For multi-mode (TV) layouts, groups are formed along mode 0 (the thread
    dimension).  Each thread's accesses across all value modes (mode 1+) are
    included in its group's analysis.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        group_size: Threads per group (32 = NVIDIA warp, 64 = AMD wave).
        element_bytes: Size of each element in bytes.
        cache_line_bytes: Cache line size in bytes.

    Returns:
        dict with:
            groups: list of per-group coalescing_efficiency() results
            worst_group: index of group with lowest efficiency
            worst_efficiency: the lowest efficiency across all groups
    """
    layout = as_layout(layout)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    thread_count, value_count = _tv_dimensions(layout)
    num_groups = (thread_count + group_size - 1) // group_size

    groups = []
    worst_idx = 0
    worst_eff = float("inf")

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, thread_count)

        cache_lines = set()
        unique_offsets = set()
        for t in range(start, end):
            for v in range(value_count):
                flat_idx = v * thread_count + t
                offset = layout(flat_idx)
                unique_offsets.add(offset)
                byte_addr = offset * element_bytes
                cache_line = byte_addr // cache_line_bytes
                cache_lines.add(cache_line)

        transactions = len(cache_lines)
        useful_bytes = len(unique_offsets) * element_bytes
        transferred_bytes = transactions * cache_line_bytes
        efficiency = useful_bytes / transferred_bytes if transferred_bytes > 0 else 0.0

        result = {
            "transactions": transactions,
            "efficiency": efficiency,
            "cache_lines": sorted(cache_lines),
        }
        groups.append(result)
        if efficiency < worst_eff:
            worst_eff = efficiency
            worst_idx = g

    return {
        "groups": groups,
        "worst_group": worst_idx,
        "worst_efficiency": worst_eff,
    }


# =============================================================================
# Permutation analysis
# =============================================================================


def cycles(layout: Layout) -> list:
    """Return the cycle decomposition of a bijective layout.

    When a layout is a bijection on [0, cosize), it defines a permutation.
    This function decomposes that permutation into disjoint cycles.

    Fixed points (cycles of length 1) are included.

    Raises ValueError if the layout is not bijective.

    Examples:
        # Identity: all fixed points
        cycles(Layout(4, 1))
        # [[0], [1], [2], [3]]

        # Swizzle(1, 0, 1) on 4 elements: XOR with bit 1
        cycles(compose(Swizzle(1, 0, 1), Layout(4, 1)))
        # [[0], [1, 2], [3]]  -- 0 and 3 are fixed, 1<->2 swap
    """
    layout = as_layout(layout)
    if not is_bijective(layout):
        raise ValueError(
            f"Layout is not bijective (size={size(layout)}, "
            f"cosize={cosize(layout)}, image_size={len(image(layout))})"
        )

    n = cosize(layout)
    visited = [False] * n
    result = []

    for start in range(n):
        if visited[start]:
            continue
        cycle = []
        current = start
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = layout(current)
        if cycle:
            result.append(cycle)

    return result


def fixed_points(layout: Layout) -> list:
    """Return offsets that map to themselves: layout(i) == i.

    Does not require the layout to be bijective.

    Examples:
        fixed_points(Layout(4, 1))          # [0, 1, 2, 3]
        fixed_points(Layout((2, 2), (2, 1)))  # [0, 3]
    """
    layout = as_layout(layout)
    return [i for i in range(size(layout)) if layout(i) == i]


def order(layout: Layout) -> int:
    """Return the permutation order: smallest k > 0 where layout^k = identity.

    The order is the LCM of all cycle lengths.

    Raises ValueError if the layout is not bijective.

    Examples:
        order(Layout(4, 1))    # 1 (identity)

        # Swizzle is its own inverse (XOR twice = identity), so order = 2
        # (unless it has fixed points only, then order = 1)
    """
    layout = as_layout(layout)
    from math import gcd

    cycle_list = cycles(layout)
    if not cycle_list:
        return 1

    result = 1
    for cycle in cycle_list:
        length = len(cycle)
        result = result * length // gcd(result, length)

    return result


# =============================================================================
# Contiguity
# =============================================================================


def contiguity(layout: Layout) -> int:
    """Return the longest contiguous vector width from the start of the layout.

    Counts how many consecutive elements starting from flat index 0 map
    to consecutive memory offsets.  This tells you the maximum vector
    load/store width: if contiguity is 4, you can issue a 4-wide
    vectorized access.

    Formally, this is max_common_vector(layout, identity_of_same_size),
    but the name makes the intent clear.

    Examples:
        contiguity(Layout(8, 1))              # 8  (fully contiguous)
        contiguity(Layout(8, 2))              # 1  (strided, no contiguity)
        contiguity(Layout((4, 8), (1, 4)))    # 32 (column-major = contiguous in flat order)
        contiguity(Layout((4, 8), (1, 8)))    # 4  (contiguous within mode 0)
    """
    layout = as_layout(layout)
    return max_common_vector(layout, Layout(size(layout)))


def mode_contiguity(layout: Layout) -> list:
    """Return contiguous vector width for each top-level mode.

    For each mode, fixes all other modes to coordinate 0 and measures
    the contiguity of that single-mode slice.  This tells you the
    vectorizable width along each axis independently.

    Args:
        layout: Layout to analyze (must be rank >= 1).

    Returns:
        List of ints, one per top-level mode.

    Examples:
        mode_contiguity(Layout((4, 8), (1, 8)))    # [4, 1] (mode 0 contiguous, mode 1 strided)
        mode_contiguity(Layout((4, 8), (1, 4)))     # [4, 1] (mode 0 contiguous, mode 1 stride-4)
        mode_contiguity(Layout((4, 8), (8, 1)))     # [1, 8] (row-major: mode 1 contiguous)
    """
    layout = as_layout(layout)
    r = rank(layout)
    if r == 0:
        return [1]

    result = []
    for i in range(r):
        mode_layout = mode(layout, i)
        if is_int(mode_layout):
            # Scalar mode (size 1) — trivially contiguous
            result.append(1)
        else:
            result.append(contiguity(mode_layout))
    return result


def slice_contiguity(layout: Layout, coord) -> int:
    """Contiguity of a layout after fixing some coordinates.

    Slices the layout by ``coord`` (using None for free dimensions)
    and returns the contiguity of the resulting sublayout.  This answers
    "given these fixed coordinates, what's the vectorizable width of
    the remaining free dimensions?"

    Args:
        layout: Layout to slice.
        coord: Coordinate tuple with None for free dims and ints for
            fixed dims. E.g. ``(None, 3)`` fixes mode 1 to 3.

    Returns:
        Contiguous vector width of the sliced sublayout.

    Examples:
        # Fix row, measure column contiguity
        slice_contiguity(Layout((4, 8), (8, 1)), (0, None))  # 8

        # Fix column, measure row contiguity
        slice_contiguity(Layout((4, 8), (8, 1)), (None, 0))  # 1
    """
    layout = as_layout(layout)
    sublayout = layout(*coord) if isinstance(coord, tuple) else layout(coord)
    if isinstance(sublayout, int):
        return 1  # fully fixed, scalar result
    return contiguity(sublayout)


# =============================================================================
# Atom analysis
# =============================================================================


def atom_summary(atom: MMAAtom) -> dict:
    """Summarize an MMA atom's key properties.

    Extracts the numbers a kernel developer cares about: how many threads,
    how many registers per thread for each operand, and whether the layouts
    are well-formed.

    Args:
        atom: An MMAAtom (NVIDIA or AMD).

    Returns:
        dict with shape_mnk, threads, values_a/b/c, c_coverage_ok,
        a_broadcast, b_broadcast.  Also prints a human-readable summary.

    Examples:
        from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
        atom_summary(SM80_16x8x16_F16F16F16F16_TN)
        # SM80_16x8x16_F16F16F16F16_TN
        #   Shape (M, N, K): 16 x 8 x 16
        #   Threads:          32
        #   Values per thread: A=8, B=4, C=4
        #   C covers M*N:     True
    """
    M, N, K = atom.shape_mnk

    # Thread count from C layout's thread mode (mode 0)
    c_thr = mode(atom.c_layout, 0)
    threads = size(c_thr)

    # Values per thread from each layout's value mode (mode 1)
    values_a = size(mode(atom.a_layout, 1))
    values_b = size(mode(atom.b_layout, 1))
    values_c = size(mode(atom.c_layout, 1))

    # Check C layout coverage: every element of M*N should be hit exactly once
    c_offset_list = []
    num_t = size(mode(atom.c_layout, 0))
    num_v = size(mode(atom.c_layout, 1))
    for t in range(num_t):
        for v in range(num_v):
            c_offset_list.append(atom.c_layout(t, v))
    c_offsets = set(c_offset_list)
    c_coverage_ok = c_offsets == set(range(M * N)) and len(c_offset_list) == M * N

    # Check for broadcast (stride-0) in A and B
    a_broadcast = atom.a_layout.filter() != atom.a_layout
    b_broadcast = atom.b_layout.filter() != atom.b_layout

    result = {
        "name": atom.name,
        "shape_mnk": atom.shape_mnk,
        "threads": threads,
        "values_a": values_a,
        "values_b": values_b,
        "values_c": values_c,
        "c_coverage_ok": c_coverage_ok,
        "a_broadcast": a_broadcast,
        "b_broadcast": b_broadcast,
    }

    lines = [
        atom.name,
        f"  Shape (M, N, K): {M} x {N} x {K}",
        f"  Threads:          {threads}",
        f"  Values per thread: A={values_a}, B={values_b}, C={values_c}",
        f"  C covers M*N:     {c_coverage_ok}",
    ]
    if a_broadcast:
        lines.append("  A has broadcast (stride-0) modes")
    if b_broadcast:
        lines.append("  B has broadcast (stride-0) modes")

    text = "\n".join(lines)
    print(text)
    result["text"] = text
    return result


def _operand_coverage(layout: Layout, domain_size: int) -> dict:
    """Analyze a single operand layout's coverage of its logical domain."""
    num_t = size(mode(layout, 0))
    num_v = size(mode(layout, 1))
    total_accesses = num_t * num_v

    offsets = []
    for t in range(num_t):
        for v in range(num_v):
            offsets.append(layout(t, v))

    unique = set(offsets)
    expected = set(range(domain_size))
    missing = expected - unique
    extra = unique - expected
    duplicates = total_accesses - len(unique)

    return {
        "domain_size": domain_size,
        "unique_offsets": len(unique),
        "total_accesses": total_accesses,
        "duplicates": duplicates,
        "coverage_ok": unique == expected,
        "missing": sorted(missing) if missing else [],
        "extra": sorted(extra) if extra else [],
        "thread_utilization": (len(unique) / total_accesses if total_accesses > 0 else 0.0),
    }


def operand_analysis(atom: MMAAtom) -> dict:
    """Detailed per-operand analysis of an MMA atom.

    Extends ``atom_summary()`` with exact coverage checks, duplicate
    load counts, and utilization metrics for each operand (A, B, C).

    Args:
        atom: An MMAAtom (NVIDIA or AMD).

    Returns:
        dict with keys 'a', 'b', 'c', each containing:
            domain_size: expected number of unique offsets (M*K, N*K, M*N)
            unique_offsets: actual number of unique offsets produced
            total_accesses: threads * values (total layout evaluations)
            duplicates: total_accesses - unique_offsets
            coverage_ok: True if offsets exactly cover range(domain_size)
            missing: sorted list of missing offsets (empty if coverage_ok)
            extra: sorted list of out-of-range offsets (empty if coverage_ok)
            thread_utilization: unique_offsets / total_accesses (1.0 = no waste)
    """
    M, N, K = atom.shape_mnk

    return {
        "a": _operand_coverage(atom.a_layout, M * K),
        "b": _operand_coverage(atom.b_layout, N * K),
        "c": _operand_coverage(atom.c_layout, M * N),
    }


# =============================================================================
# Algebra explanation
# =============================================================================


def explain(fn, *args):
    """Show step-by-step how an algebra operation computes its result.

    Expands the mathematical definition with concrete values, showing
    each intermediate layout.  This is the "show your work" for the
    layout algebra --- useful for building intuition about how compose,
    complement, divide, and product relate to each other.

    Supported operations: logical_divide, logical_product, complement,
    compose, right_inverse, left_inverse, blocked_product, raked_product,
    zipped_divide, tiled_divide, flat_divide.

    Examples:
        explain(logical_divide, Layout(16, 1), 4)
        explain(logical_product, Layout(4, 1), Layout(3, 1))
        explain(complement, Layout(4, 2), 16)
    """
    name = fn.__name__
    lines = []

    if name == "logical_divide":
        L, T = args
        if isinstance(T, int):
            T = Layout(T)
        lines.append(f"logical_divide({L}, {T})")
        actual = logical_divide(L, T)

        if is_layout(T):
            lines.append("  = compose(L, Layout(T, complement(T, size(L))))")
            lines.append("")
            lines.append(f"  L = {L}")
            lines.append(f"  T = {T}")
            lines.append(f"  size(L) = {size(L)}")
            comp = complement(T, size(L))
            lines.append(f"  complement(T, {size(L)}) = {comp}")
            intermediate = Layout(T, comp)
            lines.append(f"  Layout(T, complement) = {intermediate}")
            result = compose(L, intermediate)
            lines.append(f"  compose(L, {intermediate}) = {result}")
        else:
            lines.append("  Divides each mode of L by the corresponding tiler element.")
            lines.append("")
            lines.append(f"  L = {L}")
            lines.append(f"  T = {T}")

        lines.append("")
        lines.append(f"  result = {actual}")

    elif name == "logical_product":
        A, B = args
        if isinstance(B, int):
            B = Layout(B)
        lines.append(f"logical_product({A}, {B})")

        if is_layout(B):
            lines.append("  = Layout(A, compose(complement(A, size(A)*size(B)), B))")
            lines.append("")
            lines.append(f"  A = {A}")
            lines.append(f"  B = {B}")
            bound = size(A) * size(B)
            lines.append(f"  size(A) * size(B) = {bound}")
            comp = complement(A, bound)
            lines.append(f"  complement(A, {bound}) = {comp}")
            comp_b = compose(comp, B)
            lines.append(f"  compose(complement, B) = {comp_b}")
            result = Layout(A, comp_b)
            lines.append(f"  Layout(A, {comp_b}) = {result}")
        else:
            # Tuple tiler: mode-by-mode decomposition
            lines.append("  For tuple tilers, applies logical_product mode-by-mode.")
            lines.append("")
            lines.append(f"  A = {A}")
            lines.append(f"  B = {B}")
            for i in range(len(B)):
                mi = mode(A, i)
                bi = B[i]
                ri = logical_product(mi, bi)
                lines.append(f"  mode {i}: logical_product({mi}, {bi}) = {ri}")

        lines.append("")
        actual = logical_product(A, B)
        lines.append(f"  result = {actual}")

    elif name == "complement":
        L = args[0]
        bound = args[1] if len(args) > 1 else None
        if bound is not None:
            lines.append(f"complement({L}, {bound})")
        else:
            lines.append(f"complement({L})")
            bound = cosize(L)
        lines.append(f"  Fills the gaps in L's codomain up to bound={bound}.")
        lines.append("")
        lines.append(f"  L = {L}")
        lines.append(f"  image(L) = {image(L)}")
        lines.append(f"  codomain = [0, {bound})")
        comp = complement(*args)
        lines.append(f"  complement = {comp}")
        lines.append(f"  image(complement) = {image(comp)}")

    elif name == "compose":
        A, B = args
        lines.append(f"compose({A}, {B})")
        lines.append("  C(i) = A(B(i))")
        lines.append("")
        lines.append(f"  A = {A}")
        lines.append(f"  B = {B}")
        result = compose(A, B)
        lines.append(f"  result = {result}")
        lines.append("")
        n = min(size(result), 8)
        lines.append(f"  First {n} values:")
        for i in range(n):
            lines.append(f"    i={i}: B({i})={B(i)}, A({B(i)})={result(i)}")

    elif name == "right_inverse":
        L = args[0]
        lines.append(f"right_inverse({L})")
        lines.append("  R such that L(R(i)) == i")
        lines.append("")
        R = right_inverse(L)
        lines.append(f"  L = {L}")
        lines.append(f"  R = {R}")
        n = min(size(R), 8)
        lines.append("")
        lines.append(f"  Verification (first {n}):")
        for i in range(n):
            lines.append(f"    R({i})={R(i)}, L(R({i}))={L(R(i))}")

    elif name == "left_inverse":
        L = args[0]
        lines.append(f"left_inverse({L})")
        lines.append("  R such that R(L(i)) == i")
        lines.append("")
        R = left_inverse(L)
        lines.append(f"  L = {L}")
        lines.append(f"  R = {R}")
        n = min(size(L), 8)
        lines.append("")
        lines.append(f"  Verification (first {n}):")
        for i in range(n):
            lines.append(f"    L({i})={L(i)}, R(L({i}))={R(L(i))}")

    elif name == "blocked_product":
        A, B = args
        lines.append(f"blocked_product({A}, {B})")
        lines.append("  Like logical_product, but interleaves corresponding modes:")
        lines.append("  ((A0, B0), (A1, B1), ...) — A varies fastest (block-first).")
        lines.append("")
        lp = logical_product(A, B)
        lines.append(f"  logical_product(A, B) = {lp}")
        actual = blocked_product(A, B)
        lines.append(f"  blocked_product(A, B) = {actual}")
        lines.append("")
        lines.append("  Mode structure:")
        for i in range(max(1, len(actual.shape) if isinstance(actual.shape, tuple) else 1)):
            m = mode(actual, i) if isinstance(actual.shape, tuple) else actual
            lines.append(f"    mode {i}: {m.shape} : {m.stride}")

    elif name == "raked_product":
        A, B = args
        lines.append(f"raked_product({A}, {B})")
        lines.append("  Like blocked_product, but B varies fastest (rake-first):")
        lines.append("  ((B0, A0), (B1, A1), ...) — elements are interleaved.")
        lines.append("")
        bp = blocked_product(A, B)
        lines.append(f"  blocked_product(A, B) = {bp}")
        actual = raked_product(A, B)
        lines.append(f"  raked_product(A, B)   = {actual}")
        lines.append("")
        lines.append("  Compare first 8 offsets:")
        n = min(size(actual), 8)
        bp_vals = [bp(i) for i in range(n)]
        rp_vals = [actual(i) for i in range(n)]
        lines.append(f"    blocked: {bp_vals}")
        lines.append(f"    raked:   {rp_vals}")

    elif name in ("zipped_divide", "tiled_divide", "flat_divide"):
        L, T = args
        lines.append(f"{name}({L}, {T})")
        lines.append("  Rearrangement of logical_divide result.")
        lines.append("")
        ld = logical_divide(L, T)
        lines.append(f"  logical_divide({L}, {T})")
        lines.append(f"    = {ld}")
        actual = fn(L, T)
        lines.append(f"  {name}:")
        lines.append(f"    = {actual}")
        lines.append("")
        if name == "zipped_divide":
            lines.append("  Structure: ((tiles), (rests))")
        elif name == "tiled_divide":
            lines.append("  Structure: ((tiles), rest0, rest1, ...)")
        else:
            lines.append("  Structure: (tile0, tile1, ..., rest0, rest1, ...)")

    else:
        lines.append(f"explain() does not support {name}.")
        lines.append("Supported: logical_divide, logical_product, complement,")
        lines.append("           compose, right_inverse, left_inverse,")
        lines.append("           blocked_product, raked_product,")
        lines.append("           zipped_divide, tiled_divide, flat_divide.")

    text = "\n".join(lines)
    print(text)
    return text
