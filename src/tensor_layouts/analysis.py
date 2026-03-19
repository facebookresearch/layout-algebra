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

"""GPU layout analysis: bank conflicts, coalescing, permutation structure.

These functions analyze layouts in GPU-specific contexts --- shared memory
bank conflicts, global memory coalescing, and permutation cycle structure.
They build on the core algebra but encode hardware knowledge (bank count,
cache line size, warp width).

    from tensor_layouts.analysis import bank_conflicts, coalescing_efficiency

Import explicitly; these are not re-exported from the top-level package.
"""

from .layouts import (
    Layout, size, cosize, mode, idx2crd, is_layout, is_int,
    image, is_bijective,
    compose, complement, logical_divide, logical_product,
    right_inverse, left_inverse,
    blocked_product, raked_product,
    zipped_divide, tiled_divide, flat_divide,
)


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
    table = {}
    for i in range(size(layout)):
        coord = idx2crd(i, layout.shape)
        offset = layout(i)
        table.setdefault(offset, []).append(coord)
    return table


# =============================================================================
# Bank conflict analysis
# =============================================================================

def bank_conflicts(layout: Layout, *, num_banks: int = 32,
                   element_bytes: int = 2, bank_width_bytes: int = 4):
    """Analyze shared memory bank conflicts for a thread-to-offset layout.

    Given a layout that maps thread indices to shared memory offsets,
    compute how many bank conflicts occur when all threads access memory
    simultaneously.

    Shared memory is divided into banks (typically 32).  Two threads
    conflict when they access the same bank but different addresses
    within that bank.  The conflict factor (max_ways) tells you how
    many times the access must serialize.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        num_banks: Number of shared memory banks (32 on NVIDIA GPUs).
        element_bytes: Size of each element in bytes (2 for fp16).
        bank_width_bytes: Width of each bank in bytes (4 on NVIDIA GPUs).

    Returns:
        dict with:
            conflict_free: True if no bank conflicts
            max_ways: worst-case serialization factor (1 = no conflicts)
            bank_to_threads: {bank_id: [thread_ids...]} for all accessed banks

    Examples:
        # Linear layout: threads access consecutive elements
        bank_conflicts(Layout(32, 1))
        # {'conflict_free': True, 'max_ways': 1, ...}

        # All threads hit the same address
        bank_conflicts(Layout(32, 0))
        # {'conflict_free': True, 'max_ways': 1, ...}  (broadcast, not a conflict)
    """
    elements_per_bank = bank_width_bytes // element_bytes
    if elements_per_bank < 1:
        elements_per_bank = 1

    n = size(layout)

    # Map each thread to (bank, word_address)
    # A bank conflict occurs when threads access different 4-byte words in the
    # same bank.  Two threads accessing the same word get a broadcast (no conflict).
    thread_banks = {}  # bank -> [(thread_id, word_address), ...]
    for t in range(n):
        offset = layout(t)
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
        'conflict_free': max_ways <= 1,
        'max_ways': max_ways,
        'bank_to_threads': bank_to_threads,
    }


# =============================================================================
# Coalescing analysis
# =============================================================================

def coalescing_efficiency(layout: Layout, *, warp_size: int = 32,
                          element_bytes: int = 2,
                          cache_line_bytes: int = 128):
    """Analyze global memory coalescing for a thread-to-offset layout.

    Given a layout that maps thread indices to global memory offsets,
    compute how many cache line transactions a warp needs and the
    resulting bandwidth efficiency.

    In the ideal case (fully coalesced), 32 consecutive threads access
    32 consecutive elements within a single cache line, requiring one
    transaction.  In the worst case, each thread triggers a separate
    transaction.

    Args:
        layout: Maps thread_id -> memory offset (in elements).
        warp_size: Threads per warp (32 on NVIDIA/AMD GPUs).
        element_bytes: Size of each element in bytes.
        cache_line_bytes: Cache line size in bytes (128 on NVIDIA GPUs).

    Returns:
        dict with:
            transactions: number of cache line transactions needed
            efficiency: ratio of useful bytes to transferred bytes (1.0 = perfect)
            cache_lines: sorted list of cache line indices touched

    Examples:
        # Perfectly coalesced: 32 threads, stride 1, fp16
        coalescing_efficiency(Layout(32, 1))
        # {'transactions': 1, 'efficiency': 0.5, ...}  -- 64B used of 128B line

        # Strided access: each thread 2 elements apart
        coalescing_efficiency(Layout(32, 2))
        # {'transactions': 2, 'efficiency': 0.5, ...}
    """
    n = min(size(layout), warp_size)

    # Find which cache lines are touched
    cache_lines = set()
    for t in range(n):
        offset = layout(t)
        byte_addr = offset * element_bytes
        cache_line = byte_addr // cache_line_bytes
        cache_lines.add(cache_line)

    transactions = len(cache_lines)
    useful_bytes = n * element_bytes
    transferred_bytes = transactions * cache_line_bytes
    efficiency = useful_bytes / transferred_bytes if transferred_bytes > 0 else 0.0

    return {
        'transactions': transactions,
        'efficiency': efficiency,
        'cache_lines': sorted(cache_lines),
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

    if name == 'logical_divide':
        L, T = args
        if isinstance(T, int):
            T = Layout(T)
        lines.append(f'logical_divide({L}, {T})')
        actual = logical_divide(L, T)

        if is_layout(T):
            lines.append(f'  = compose(L, Layout(T, complement(T, size(L))))')
            lines.append(f'')
            lines.append(f'  L = {L}')
            lines.append(f'  T = {T}')
            lines.append(f'  size(L) = {size(L)}')
            comp = complement(T, size(L))
            lines.append(f'  complement(T, {size(L)}) = {comp}')
            intermediate = Layout(T, comp)
            lines.append(f'  Layout(T, complement) = {intermediate}')
            result = compose(L, intermediate)
            lines.append(f'  compose(L, {intermediate}) = {result}')
        else:
            lines.append(f'  Divides each mode of L by the corresponding tiler element.')
            lines.append(f'')
            lines.append(f'  L = {L}')
            lines.append(f'  T = {T}')

        lines.append(f'')
        lines.append(f'  result = {actual}')

    elif name == 'logical_product':
        A, B = args
        if isinstance(B, int):
            B = Layout(B)
        lines.append(f'logical_product({A}, {B})')
        lines.append(f'  = Layout(A, compose(complement(A, size(A)*size(B)), B))')
        lines.append(f'')
        lines.append(f'  A = {A}')
        lines.append(f'  B = {B}')
        bound = size(A) * size(B)
        lines.append(f'  size(A) * size(B) = {bound}')
        comp = complement(A, bound)
        lines.append(f'  complement(A, {bound}) = {comp}')
        comp_b = compose(comp, B)
        lines.append(f'  compose(complement, B) = {comp_b}')
        result = Layout(A, comp_b)
        lines.append(f'  Layout(A, {comp_b}) = {result}')
        lines.append(f'')
        actual = logical_product(A, B)
        lines.append(f'  result = {actual}')

    elif name == 'complement':
        L = args[0]
        bound = args[1] if len(args) > 1 else None
        if bound is not None:
            lines.append(f'complement({L}, {bound})')
        else:
            lines.append(f'complement({L})')
            bound = cosize(L)
        lines.append(f'  Fills the gaps in L\'s codomain up to bound={bound}.')
        lines.append(f'')
        lines.append(f'  L = {L}')
        lines.append(f'  image(L) = {image(L)}')
        lines.append(f'  codomain = [0, {bound})')
        comp = complement(*args)
        lines.append(f'  complement = {comp}')
        lines.append(f'  image(complement) = {image(comp)}')

    elif name == 'compose':
        A, B = args
        lines.append(f'compose({A}, {B})')
        lines.append(f'  C(i) = A(B(i))')
        lines.append(f'')
        lines.append(f'  A = {A}')
        lines.append(f'  B = {B}')
        result = compose(A, B)
        lines.append(f'  result = {result}')
        lines.append(f'')
        n = min(size(result), 8)
        lines.append(f'  First {n} values:')
        for i in range(n):
            lines.append(f'    i={i}: B({i})={B(i)}, A({B(i)})={result(i)}')

    elif name == 'right_inverse':
        L = args[0]
        lines.append(f'right_inverse({L})')
        lines.append(f'  R such that L(R(i)) == i')
        lines.append(f'')
        R = right_inverse(L)
        lines.append(f'  L = {L}')
        lines.append(f'  R = {R}')
        n = min(size(R), 8)
        lines.append(f'')
        lines.append(f'  Verification (first {n}):')
        for i in range(n):
            lines.append(f'    R({i})={R(i)}, L(R({i}))={L(R(i))}')

    elif name == 'left_inverse':
        L = args[0]
        lines.append(f'left_inverse({L})')
        lines.append(f'  R such that R(L(i)) == i')
        lines.append(f'')
        R = left_inverse(L)
        lines.append(f'  L = {L}')
        lines.append(f'  R = {R}')
        n = min(size(L), 8)
        lines.append(f'')
        lines.append(f'  Verification (first {n}):')
        for i in range(n):
            lines.append(f'    L({i})={L(i)}, R(L({i}))={R(L(i))}')

    elif name == 'blocked_product':
        A, B = args
        lines.append(f'blocked_product({A}, {B})')
        lines.append(f'  Like logical_product, but interleaves corresponding modes:')
        lines.append(f'  ((A0, B0), (A1, B1), ...) — A varies fastest (block-first).')
        lines.append(f'')
        lp = logical_product(A, B)
        lines.append(f'  logical_product(A, B) = {lp}')
        actual = blocked_product(A, B)
        lines.append(f'  blocked_product(A, B) = {actual}')
        lines.append(f'')
        lines.append(f'  Mode structure:')
        for i in range(max(1, len(actual.shape) if isinstance(actual.shape, tuple) else 1)):
            m = mode(actual, i) if isinstance(actual.shape, tuple) else actual
            lines.append(f'    mode {i}: {m.shape} : {m.stride}')

    elif name == 'raked_product':
        A, B = args
        lines.append(f'raked_product({A}, {B})')
        lines.append(f'  Like blocked_product, but B varies fastest (rake-first):')
        lines.append(f'  ((B0, A0), (B1, A1), ...) — elements are interleaved.')
        lines.append(f'')
        bp = blocked_product(A, B)
        lines.append(f'  blocked_product(A, B) = {bp}')
        actual = raked_product(A, B)
        lines.append(f'  raked_product(A, B)   = {actual}')
        lines.append(f'')
        lines.append(f'  Compare first 8 offsets:')
        n = min(size(actual), 8)
        bp_vals = [bp(i) for i in range(n)]
        rp_vals = [actual(i) for i in range(n)]
        lines.append(f'    blocked: {bp_vals}')
        lines.append(f'    raked:   {rp_vals}')

    elif name in ('zipped_divide', 'tiled_divide', 'flat_divide'):
        L, T = args
        lines.append(f'{name}({L}, {T})')
        lines.append(f'  Rearrangement of logical_divide result.')
        lines.append(f'')
        ld = logical_divide(L, T)
        lines.append(f'  logical_divide({L}, {T})')
        lines.append(f'    = {ld}')
        actual = fn(L, T)
        lines.append(f'  {name}:')
        lines.append(f'    = {actual}')
        lines.append(f'')
        if name == 'zipped_divide':
            lines.append(f'  Structure: ((tiles), (rests))')
        elif name == 'tiled_divide':
            lines.append(f'  Structure: ((tiles), rest0, rest1, ...)')
        else:
            lines.append(f'  Structure: (tile0, tile1, ..., rest0, rest1, ...)')

    else:
        lines.append(f'explain() does not support {name}.')
        lines.append(f'Supported: logical_divide, logical_product, complement,')
        lines.append(f'           compose, right_inverse, left_inverse,')
        lines.append(f'           blocked_product, raked_product,')
        lines.append(f'           zipped_divide, tiled_divide, flat_divide.')

    text = '\n'.join(lines)
    print(text)
    return text
