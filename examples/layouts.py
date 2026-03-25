#!/usr/bin/env python3
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

"""Layout Algebra Examples — no dependencies.

Demonstrates the core tensor-layouts API: construction, querying,
coordinate mapping, and the four algebraic operations (compose,
complement, divide, product).

Run:
    python layouts.py

See also:
    viz.py       — visualization examples (requires matplotlib)
    viz.ipynb    — Jupyter notebook gallery
"""

from tensor_layouts import * # noqa: F401,F403,F405
from tensor_layouts.analysis import * # noqa: F401,F403,F405


# =============================================================================
# Section 1: Layout Construction
# =============================================================================


def example_construction():
    """Building layouts from shape and stride.

    A Layout is a function: offset = sum(coord_i * stride_i).
    The shape defines the logical domain; the stride defines the
    memory step for each dimension.
    """
    print("\n" + "=" * 60)
    print("1. Layout Construction")
    print("=" * 60)

    # --- Explicit shape and stride ---
    col_major = Layout((4, 8), (1, 4))
    print(f"  Column-major 4x8: {col_major}")
    # offset(i,j) = i*1 + j*4

    row_major = Layout((4, 8), (8, 1))
    print(f"  Row-major 4x8:    {row_major}")
    # offset(i,j) = i*8 + j*1

    # --- Shape only: column-major strides are computed automatically ---
    auto = Layout((4, 8))
    print(f"  Auto-stride (4,8): {auto}")
    assert auto == col_major

    # --- 1D layouts ---
    contiguous = Layout(8, 1)
    print(f"  1D contiguous: {contiguous}")

    strided = Layout(8, 2)
    print(f"  1D strided:    {strided}")
    # Visits offsets 0, 2, 4, 6, 8, 10, 12, 14

    # --- Broadcast (stride 0) ---
    broadcast = Layout((4, 8), (0, 1))
    print(f"  Broadcast rows: {broadcast}")
    # All rows map to the same offsets: offset(i,j) = j

    # --- Hierarchical (nested) shapes ---
    hier = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    print(f"  Hierarchical:  {hier}")
    # Describes 2x2 tiles arranged in a 3x4 grid


# =============================================================================
# Section 2: Querying Layouts
# =============================================================================


def example_querying():
    """Query functions for shape, size, rank, and depth.

    These are free functions, not methods, following the CuTe convention.
    They also work on plain tuples and ints.
    """
    print("\n" + "=" * 60)
    print("2. Querying Layouts")
    print("=" * 60)

    layout = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))

    print(f"  Layout: {layout}")
    print(f"  shape:  {layout.shape}")
    print(f"  stride: {layout.stride}")
    print(f"  size:   {size(layout)}")  # Total number of elements
    print(f"  cosize: {cosize(layout)}")  # Span: max offset + 1
    print(f"  rank:   {rank(layout)}")  # Number of top-level modes
    print(f"  depth:  {depth(layout)}")  # Maximum nesting depth

    # mode() extracts a single mode as a Layout
    print(f"  mode 0: {mode(layout, 0)}")
    print(f"  mode 1: {mode(layout, 1)}")

    # Query functions also work on tuples and ints
    print(f"\n  size((4, 8)):   {size((4, 8))}")
    print(f"  rank((4, 8)):   {rank((4, 8))}")
    print(f"  size(32):       {size(32)}")


# =============================================================================
# Section 3: Coordinate Mapping
# =============================================================================


def example_coordinate_mapping():
    """Calling a layout to map coordinates to memory offsets.

    layout(i, j) computes offset = i*stride_i + j*stride_j.
    layout(flat_idx) treats the layout as a 1D function over its flat domain.
    """
    print("\n" + "=" * 60)
    print("3. Coordinate Mapping")
    print("=" * 60)

    layout = Layout((4, 8), (1, 4))
    print(f"  Layout: {layout}")

    # Multi-dimensional coordinates
    print(f"  (0, 0) -> {layout(0, 0)}")  # 0
    print(f"  (2, 3) -> {layout(2, 3)}")  # 2 + 12 = 14
    print(f"  (3, 7) -> {layout(3, 7)}")  # 3 + 28 = 31

    # Flat index: column-major traversal of the domain
    print(f"  flat 0 -> {layout(0)}")  # Same as (0,0) -> 0
    print(f"  flat 5 -> {layout(5)}")  # Same as (1,1) -> 5
    print(f"  flat 31 -> {layout(31)}")  # Same as (3,7) -> 31

    # idx2crd: convert flat index to multi-dimensional coordinate
    print(f"\n  idx2crd(5, (4, 8)):  {idx2crd(5, (4, 8))}")  # (1, 1)
    print(f"  idx2crd(14, (4, 8)): {idx2crd(14, (4, 8))}")  # (2, 3)

    # crd2flat: convert coordinate to flat index
    print(f"  crd2flat((2, 3), (4, 8)): {crd2flat((2, 3), (4, 8))}")  # 14


# =============================================================================
# Section 4: Tuple Arithmetic
# =============================================================================


def example_tuple_arithmetic():
    """Arithmetic on nested tuples — the foundation of layout algebra.

    CuTe shapes and strides are nested integer tuples. These functions
    operate element-wise or with accumulation across nested structure.
    """
    print("\n" + "=" * 60)
    print("4. Tuple Arithmetic")
    print("=" * 60)

    # prefix_product: running product (exclusive prefix), used to compute
    # column-major strides from a shape
    shape = (2, 3, 4)
    pp = prefix_product(shape)
    print(f"  prefix_product({shape}): {pp}")  # (1, 2, 6)
    # This is exactly the column-major stride for shape (2,3,4)

    # suffix_product: running product from the right
    sp = suffix_product(shape)
    print(f"  suffix_product({shape}): {sp}")  # (12, 4, 1)
    # This is exactly the row-major stride for shape (2,3,4)

    # inner_product: sum of element-wise products
    a = (2, 3)
    b = (4, 5)
    print(f"  inner_product({a}, {b}): {inner_product(a, b)}")  # 2*4 + 3*5 = 23

    # elem_scale: element-wise multiply
    print(f"  elem_scale({a}, {b}): {elem_scale(a, b)}")  # (8, 15)

    # Works on nested tuples too
    nested = ((2, 3), 4)
    pp_nested = prefix_product(nested)
    print(f"  prefix_product({nested}): {pp_nested}")  # ((1, 2), 6)


# =============================================================================
# Section 5: Layout Manipulation
# =============================================================================


def example_manipulation():
    """Reshape and reorganize layouts without changing the mapping.

    flatten, coalesce, sort, append, prepend, group.
    """
    print("\n" + "=" * 60)
    print("5. Layout Manipulation")
    print("=" * 60)

    hier = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    print(f"  Original:  {hier}")

    # flatten: remove all nesting, produce a flat tuple of (shape, stride) pairs
    flat = flatten(hier)
    print(f"  flatten:   {flat}")

    # coalesce: merge contiguous modes (adjacent modes with compatible strides)
    coal = coalesce(flat)
    print(f"  coalesce:  {coal}")

    # sort: reorder modes by stride (ascending)
    unsorted = Layout((4, 2), (2, 1))  # stride 2 before stride 1
    sorted_layout = sort(unsorted)
    print(f"\n  Before sort: {unsorted}")
    print(f"  After sort:  {sorted_layout}")

    # append / prepend: add a mode
    base = Layout((4, 8), (1, 4))
    extended = append(base, Layout(3, 32))
    print(f"\n  append: {base} + 3:32 = {extended}")

    pre = prepend(base, Layout(2, 0))
    print(f"  prepend: 2:0 + {base} = {pre}")

    # group: merge a range of modes into one nested mode
    flat3 = Layout((2, 3, 4), (1, 2, 6))
    grouped = group(flat3, 0, 2)
    print(f"\n  group(0..2): {flat3} -> {grouped}")


# =============================================================================
# Section 6: Composition
# =============================================================================


def example_composition():
    """compose(A, B) — function composition: C(i) = A(B(i)).

    B selects which elements of A to visit, and in what order.
    The result has B's shape.

    This is the fundamental operation of CuTe layout algebra.
    """
    print("\n" + "=" * 60)
    print("6. Composition")
    print("=" * 60)

    # Simple 1D: A visits offsets 0,2,4,6 (4 elts, stride 2)
    # B visits indices 0,1,2,3 (4 elts, stride 1 — trivial)
    # compose(A, B)(i) = A(B(i)) = A(i) — B just selects A's first 4 elements
    a = Layout(8, 2)
    b = Layout(4, 1)
    c = compose(a, b)
    print(f"  compose({a}, {b}) = {c}")
    # Result: Layout(4, 2) — visits offsets 0,2,4,6

    # B selects every other element of A
    a2 = Layout(8, 1)
    b2 = Layout(4, 2)
    c2 = compose(a2, b2)
    print(f"  compose({a2}, {b2}) = {c2}")
    # Result: Layout(4, 2) — B picks indices 0,2,4,6 from A

    # 2D compose: mode-by-mode via Tile
    a3 = Layout((4, 8), (8, 1))
    tiler = Tile(Layout(2, 1), Layout(4, 1))
    c3 = compose(a3, tiler)
    print(f"\n  compose({a3}, Tile(2:1, 4:1)) = {c3}")
    # Selects the top-left 2x4 subblock

    # Shape as tiler (shorthand for Tile of contiguous layouts)
    c4 = compose(a3, (2, 4))
    print(f"  compose({a3}, (2, 4)) = {c4}")
    assert c3 == c4


# =============================================================================
# Section 7: Complement
# =============================================================================


def example_complement():
    """complement(L) — the layout that fills in L's gaps.

    If L visits offsets {0, 2, 4, 6} in a range of 8, complement(L, 8)
    visits {0, 1} (stride 1 within each stride-2 gap).
    Together, Layout(L, complement(L)) covers every offset exactly once.
    """
    print("\n" + "=" * 60)
    print("7. Complement")
    print("=" * 60)

    # L visits 0, 2, 4, 6  (4 elements, stride 2)
    layout = Layout(4, 2)
    comp = complement(layout, 16)
    print(f"  complement({layout}, 16) = {comp}")
    # Complement fills the odd offsets and extends to 16

    # Contiguous layout: complement starts after the layout's codomain
    layout2 = Layout(4, 1)
    comp2 = complement(layout2, 16)
    print(f"  complement({layout2}, 16) = {comp2}")

    # 2D example
    layout3 = Layout((2, 2), (1, 4))
    comp3 = complement(layout3, 16)
    print(f"  complement({layout3}, 16) = {comp3}")

    # complement is the key to logical_divide:
    # logical_divide(A, T) = compose(A, Layout(T, complement(T, size(A))))
    print(f"\n  The divide connection:")
    a = Layout(16, 1)
    t = Layout(4, 1)
    bundled = Layout(t, complement(t, size(a)))
    print(f"    T = {t}")
    print(f"    complement(T, 16) = {complement(t, size(a))}")
    print(f"    Layout(T, complement) = {bundled}")
    print(f"    compose(A, bundled) = {compose(a, bundled)}")
    print(f"    logical_divide(A, T) = {logical_divide(a, t)}")


# =============================================================================
# Section 8: Division
# =============================================================================


def example_division():
    """logical_divide — split a layout into (tile, rest).

    Division answers: "if I process this layout in tiles of size T,
    how do I organize the iteration?"

    Result structure: (within-tile coords, across-tile coords).
    zipped/tiled/flat variants control how the result modes are organized.
    """
    print("\n" + "=" * 60)
    print("8. Division")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))
    print(f"  Layout: {layout}")

    # logical_divide: hierarchical result
    ld = logical_divide(layout, (2, 4))
    print(f"\n  logical_divide by (2, 4): {ld}")
    # Each mode becomes (tile_part, rest_part)

    # zipped_divide: ((tiles), (rests))
    zd = zipped_divide(layout, (2, 4))
    print(f"  zipped_divide by (2, 4):  {zd}")

    # tiled_divide: ((tiles), rest0, rest1, ...)
    td = tiled_divide(layout, (2, 4))
    print(f"  tiled_divide by (2, 4):   {td}")

    # flat_divide: (tile0, tile1, rest0, rest1, ...)
    fd = flat_divide(layout, (2, 4))
    print(f"  flat_divide by (2, 4):    {fd}")

    # Integer tiler divides the first mode
    print(f"\n  1D division:")
    layout_1d = Layout(16, 1)
    ld_1d = logical_divide(layout_1d, 4)
    print(f"  logical_divide({layout_1d}, 4) = {ld_1d}")
    # (4, 4) : (1, 4) — 4-element tiles, 4 tiles

    # Layout tiler for non-contiguous tiling
    ld_strided = logical_divide(layout_1d, Layout(4, 2))
    print(f"  logical_divide({layout_1d}, 4:2) = {ld_strided}")


# =============================================================================
# Section 9: Product
# =============================================================================


def example_product():
    """logical_product — replicate A's pattern across B's domain.

    Product is the reverse of division: if division splits a layout into
    tiles, product creates a layout from a tile replicated across copies.

    blocked_product and raked_product control the interleaving pattern.
    """
    print("\n" + "=" * 60)
    print("9. Product")
    print("=" * 60)

    # logical_product: replicate a 4-element pattern 3 times
    a = Layout(4, 1)
    b = Layout(3, 1)
    lp = logical_product(a, b)
    print(f"  logical_product({a}, {b}) = {lp}")
    # (4, 3) : (1, 4) — tile of 4, repeated 3 times at offsets 0, 4, 8

    # Verify: the tile pattern repeats
    for tile_idx in range(3):
        offsets = [lp(i, tile_idx) for i in range(4)]
        print(f"    tile {tile_idx}: offsets {offsets}")

    # blocked_product: interleave by mode
    a2 = Layout((2, 4), (1, 2))
    b2 = Layout((3, 2), (1, 3))
    bp = blocked_product(a2, b2)
    print(f"\n  blocked_product({a2}, {b2}) = {bp}")
    # Each mode of A is paired with corresponding mode of B

    # raked_product: A's pattern is interleaved within B's
    rp = raked_product(a, b)
    print(f"\n  raked_product({a}, {b}) = {rp}")


# =============================================================================
# Section 10: Inverse
# =============================================================================


def example_inverse():
    """right_inverse, left_inverse — undo a layout's mapping.

    right_inverse(L) gives R such that L(R(i)) = i for all valid i.
    left_inverse(L) gives R such that R(L(i)) = i for all valid i.
    max_common_layout finds the largest common prefix of two layouts.
    """
    print("\n" + "=" * 60)
    print("10. Inverse")
    print("=" * 60)

    layout = Layout((4, 2), (2, 1))
    ri = right_inverse(layout)
    print(f"  right_inverse({layout}) = {ri}")

    # Verify: layout(ri(i)) == i
    for i in range(size(layout)):
        assert layout(ri(i)) == i, f"right_inverse failed at {i}"
    print(f"  Verified: layout(ri(i)) == i for all i")

    li = left_inverse(layout)
    print(f"  left_inverse({layout}) = {li}")

    # Verify: li(layout(i)) == i
    for i in range(size(layout)):
        assert li(layout(i)) == i, f"left_inverse failed at {i}"
    print(f"  Verified: li(layout(i)) == i for all i")

    # max_common_layout: largest layout that divides both
    a = Layout((4, 8), (1, 4))
    b = Layout((8, 4), (1, 8))
    mcl = max_common_layout(a, b)
    print(f"\n  max_common_layout({a}, {b}) = {mcl}")


# =============================================================================
# Section 11: Swizzle
# =============================================================================


def example_swizzle():
    """Swizzle(bits, base, shift) — XOR-based bank conflict avoidance.

    A Swizzle is a function: swizzle(offset) = offset XOR ((offset >> shift) & mask)
    where mask covers 'bits' bits starting at position 'base'.

    GPU shared memory has 32 banks. Without swizzling, threads reading the
    same column of a row-major matrix all hit the same bank. Swizzling
    permutes each row's offsets so that column accesses spread across banks.

    Composing a Swizzle with a Layout embeds the swizzle inside the Layout
    so it is applied after computing the linear offset.
    """
    print("\n" + "=" * 60)
    print("11. Swizzle")
    print("=" * 60)

    # Swizzle(3, 0, 3) is the canonical CuTe example for 128-byte rows.
    # It XORs bits 0-2 with bits 3-5 of each offset.
    sw = Swizzle(3, 0, 3)
    print(f"  {sw}")
    print(f"  XORs bits 0-2 with bits 3-5 of the offset")

    # Compose with an 8x8 row-major layout
    base = Layout((8, 8), (8, 1))
    swizzled = compose(sw, base)
    print(f"\n  compose(sw, {base}) = {swizzled}")

    # Row 0 (offsets 0-7) is unchanged because bits 3-5 are all zero.
    # Each subsequent row is permuted differently, which is the point:
    # column accesses now hit different banks in each row.
    # Show offset % 8 so each row is visibly a permutation of 0-7.
    print(f"\n  Swizzled offsets mod 8 (each row is a permutation of 0-7):")
    for row in range(8):
        swz = [swizzled(row, j) % 8 for j in range(8)]
        tag = " (identity)" if swz == list(range(8)) else ""
        print(f"    row {row}: {swz}{tag}")


# =============================================================================
# Section 12: Tensor
# =============================================================================


def example_tensor():
    """Tensor — a Layout combined with a base offset.

    In CuTe C++, a Tensor is (Pointer, Layout). Here the pointer is an
    integer offset. Tensor supports slicing: fixing a coordinate
    accumulates its offset contribution and returns a new Tensor with
    a reduced Layout.
    """
    print("\n" + "=" * 60)
    print("12. Tensor")
    print("=" * 60)

    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout)
    print(f"  {tensor}")

    # Coordinate mapping (same as layout)
    print(f"  tensor(2, 5) = {tensor(2, 5)}")

    # Tensor with a base offset
    tensor_offset = Tensor(layout, offset=100)
    print(f"  {tensor_offset}")
    print(f"  tensor_offset(0, 0) = {tensor_offset(0, 0)}")  # 100
    print(f"  tensor_offset(2, 5) = {tensor_offset(2, 5)}")  # 100 + 21

    # Slicing: fix one mode, keep the other
    row2 = tensor[2, :]
    print(f"\n  tensor[2, :] = {row2}")
    print(f"    offsets: {[row2(j) for j in range(8)]}")

    col5 = tensor[:, 5]
    print(f"  tensor[:, 5] = {col5}")
    print(f"    offsets: {[col5(i) for i in range(4)]}")

    # Fix all modes: returns an integer
    val = tensor[2, 5]
    print(f"\n  tensor[2, 5] = {val}")
    assert val == tensor(2, 5)


# =============================================================================
# Section 13: Tile
# =============================================================================


def example_tile():
    """Tile — a tuple of Layouts for mode-by-mode composition.

    When passed as the second argument to compose or logical_divide,
    a Tile applies each of its elements to the corresponding mode of
    the first layout.
    """
    print("\n" + "=" * 60)
    print("13. Tile")
    print("=" * 60)

    layout = Layout((12, 8), (8, 1))
    print(f"  Layout: {layout}")

    # Tile specifies how to slice each mode independently
    tiler = Tile(Layout(3, 1), Layout(4, 1))
    print(f"  Tile:   {tiler}")

    # compose with Tile: mode-by-mode
    composed = compose(layout, tiler)
    print(f"  compose(layout, tile) = {composed}")

    # Equivalent to passing a shape tuple for contiguous tiles
    composed2 = compose(layout, (3, 4))
    print(f"  compose(layout, (3, 4)) = {composed2}")
    assert composed == composed2

    # Tile with non-contiguous access patterns
    strided_tile = Tile(Layout(3, 4), Layout(4, 2))
    composed3 = compose(layout, strided_tile)
    print(f"\n  Strided Tile: {strided_tile}")
    print(f"  compose(layout, strided_tile) = {composed3}")


# =============================================================================
# Section 14: Iteration
# =============================================================================


def example_iteration():
    """Iterating over layouts.

    iter_layout(L) yields (coordinate, offset) pairs for every element
    in colexicographic order.  Layout also supports __iter__ (yields
    coordinates) and __len__.
    """
    print("\n" + "=" * 60)
    print("14. Iteration")
    print("=" * 60)

    # --- 1D layout ---
    layout = Layout(4, 2)
    print(f"  Layout: {layout}")
    print(f"  len(layout) = {len(layout)}")
    print(f"  list(layout) = {list(layout)}")

    # --- 2D column-major: __iter__ yields coordinates ---
    layout2d = Layout((3, 4), (1, 3))
    print(f"\n  Layout: {layout2d}")
    print(f"  Iteration (coord -> offset):")
    for coord in layout2d:
        print(f"    {coord} -> {layout2d(coord)}")

    # --- iter_layout yields (coord, offset) pairs ---
    print(f"\n  iter_layout gives (coord, offset) pairs:")
    for coord, offset in iter_layout(layout2d):
        print(f"    {coord} -> {offset}")

    # --- Collect unique offsets with a set comprehension ---
    broadcast = Layout((4, 2), (0, 1))
    offsets = {broadcast(c) for c in broadcast}
    print(f"\n  Broadcast layout {broadcast}")
    print(f"  Unique offsets: {sorted(offsets)}")
    print(f"  (only {len(offsets)} unique out of {len(broadcast)} elements)")


# =============================================================================
# Section 15: Image and Injectivity
# =============================================================================


def example_image_injectivity():
    """Analyzing a layout as a function.

    image(L) returns the sorted list of distinct offsets the layout
    produces.  is_injective, is_surjective, and is_bijective answer
    basic questions about the layout viewed as a function from
    coordinates to offsets.
    """
    print("\n" + "=" * 60)
    print("15. Image and Injectivity")
    print("=" * 60)

    # --- Bijective: contiguous col-major ---
    col = Layout((4, 8), (1, 4))
    print(f"  Layout: {col}")
    print(f"  image size: {len(image(col))}, domain size: {size(col)}")
    print(f"  injective: {is_injective(col)}, surjective: {is_surjective(col)}")
    print(f"  bijective: {is_bijective(col)}")

    # --- Not injective: broadcast ---
    bcast = Layout((4, 8), (0, 1))
    print(f"\n  Broadcast: {bcast}")
    print(f"  image: {image(bcast)}")
    print(f"  injective: {is_injective(bcast)} (stride-0 causes aliasing)")

    # --- Not surjective: strided ---
    strided = Layout(4, 2)
    print(f"\n  Strided: {strided}")
    print(f"  image: {image(strided)}")
    print(f"  surjective: {is_surjective(strided)} (gaps in codomain)")

    # --- Swizzle: bijective permutation ---
    sw = compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1)))
    print(f"\n  Swizzled: {sw}")
    print(f"  bijective: {is_bijective(sw)}")


# =============================================================================
# Section 16: Functional Equivalence
# =============================================================================


def example_functional_equivalence():
    """Checking if two layouts compute the same mapping.

    Layout.__eq__ checks structural equality (same shape and stride).
    functionally_equal() checks whether two layouts produce the same
    offset for every index, regardless of representation.
    """
    print("\n" + "=" * 60)
    print("16. Functional Equivalence")
    print("=" * 60)

    # --- Coalescing preserves function ---
    layout = Layout(((2, 2), (2, 4)), ((1, 4), (2, 8)))
    coalesced = coalesce(layout)
    print(f"  Original:   {layout}")
    print(f"  Coalesced:  {coalesced}")
    print(f"  Structurally equal: {layout == coalesced}")
    print(f"  Functionally equal: {functionally_equal(layout, coalesced)}")

    # --- Flattening also preserves function ---
    flat = flatten(layout)
    print(f"\n  Flattened:  {flat}")
    print(f"  Functionally equal: {functionally_equal(layout, flat)}")

    # --- Row-major vs column-major: same shape, different function ---
    col = Layout((3, 4), (1, 3))
    row = Layout((3, 4), (4, 1))
    print(f"\n  Col-major: {col}")
    print(f"  Row-major: {row}")
    print(f"  Functionally equal: {functionally_equal(col, row)}")


# =============================================================================
# Section 17: GPU Analysis
# =============================================================================


def example_analysis():
    """Analyzing layouts for GPU performance.

    GPU kernels live or die by memory access patterns.  Two critical
    properties are:

    1. Shared memory bank conflicts: shared memory has 32 banks, each
       4 bytes wide.  When threads in a warp access different words in
       the same bank, the accesses serialize (N-way conflict).

    2. Global memory coalescing: the memory controller fetches 128-byte
       cache lines.  When a warp's threads access scattered locations,
       multiple cache lines are fetched, wasting bandwidth.

    The analysis module quantifies both, and also provides permutation
    analysis for understanding swizzle patterns.
    """
    print("\n" + "=" * 60)
    print("17. GPU Analysis")
    print("=" * 60)

    # --- Bank conflict problem ---
    # 8x8 row-major tile in shared memory.
    # Reading a column means stride-8 access: thread t reads offset 8*t.
    print("  Bank Conflicts")
    print("  " + "-" * 40)
    tile = Layout((8, 8), (8, 1))
    col_access = Layout(8, 8)  # thread t -> offset 8*t (column access)
    result = bank_conflicts(col_access, element_bytes=4)
    print(f"  Column access (stride 8): {col_access}")
    print(f"    conflict_free: {result['conflict_free']}")
    print(
        f"    max_ways: {result['max_ways']} (threads serialize {result['max_ways']}x)"
    )

    # --- Swizzle fix ---
    sw_tile = compose(Swizzle(3, 0, 3), tile)
    sw_col = sw_tile(None, 0)  # slice column 0
    result_sw = bank_conflicts(sw_col, element_bytes=4)
    print(f"\n  After Swizzle(3,0,3):")
    print(f"    Column 0 layout: {sw_col}")
    print(f"    conflict_free: {result_sw['conflict_free']}")

    # --- Who writes where? (offset_table) ---
    print(f"\n  Offset Table (Aliasing)")
    print("  " + "-" * 40)
    broadcast = Layout((4, 2), (0, 1))
    table = offset_table(broadcast)
    print(f"  Broadcast layout {broadcast}:")
    for offset, coords in sorted(table.items()):
        print(f"    offset {offset} <- {coords}")

    # --- Global memory coalescing ---
    print(f"\n  Global Memory Coalescing")
    print("  " + "-" * 40)
    coalesced = Layout(32, 1)
    r1 = coalescing_efficiency(coalesced, element_bytes=4)
    print(
        f"  Stride-1 (fp32): {r1['transactions']} transaction, "
        f"efficiency {r1['efficiency']:.0%}"
    )

    scattered = Layout(32, 64)
    r2 = coalescing_efficiency(scattered, element_bytes=2)
    print(
        f"  Stride-64 (fp16): {r2['transactions']} transactions, "
        f"efficiency {r2['efficiency']:.1%}"
    )

    # --- Permutation structure ---
    print(f"\n  Permutation Structure")
    print("  " + "-" * 40)

    # Row-major 3x2 is the transpose permutation
    rm = Layout((3, 2), (2, 1))
    print(f"  Row-major 3x2: {rm}")
    print(f"    mapping: {[rm(i) for i in range(6)]}")
    print(f"    cycles: {cycles(rm)}")
    print(f"    fixed_points: {fixed_points(rm)}")
    print(f"    order: {order(rm)} (apply 4 times to return to identity)")

    # --- Step-by-step algebra trace ---
    print(f"\n  Step-by-Step Algebra (explain)")
    print("  " + "-" * 40)
    print()
    explain(logical_divide, Layout(16, 1), 4)
    print()
    explain(logical_product, Layout(4, 1), Layout(3, 1))

    # --- Contiguity ---
    print(f"\n  Contiguity (vector width)")
    print("  " + "-" * 40)

    layouts = [
        ("Contiguous 1D", Layout(8, 1)),
        ("Strided 1D", Layout(8, 2)),
        ("Col-major 4x8", Layout((4, 8), (1, 4))),
        ("Gapped 4x8", Layout((4, 8), (1, 8))),
        ("Row-major 4x8", Layout((4, 8), (8, 1))),
    ]
    for label, l in layouts:
        print(f"  {label:20s} {str(l):25s} contiguity={contiguity(l)}")

    # --- Atom summary ---
    print(f"\n  Atom Summary")
    print("  " + "-" * 40)

    from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
    from tensor_layouts.atoms_amd import CDNA_32x32x8_F32F16F16_MFMA

    print()
    atom_summary(SM80_16x8x16_F16F16F16F16_TN)
    print()
    atom_summary(CDNA_32x32x8_F32F16F16_MFMA)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all layout algebra examples."""
    print("=" * 70)
    print("CuTe Layout Algebra Examples")
    print("=" * 70)

    example_construction()
    example_querying()
    example_coordinate_mapping()
    example_tuple_arithmetic()
    example_manipulation()
    example_composition()
    example_complement()
    example_division()
    example_product()
    example_inverse()
    example_swizzle()
    example_tensor()
    example_tile()
    example_iteration()
    example_image_injectivity()
    example_functional_equivalence()
    example_analysis()

    print("\n" + "=" * 70)
    print("All examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
