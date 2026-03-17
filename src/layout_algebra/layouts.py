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

"""Pure-Python implementation of NVIDIA CuTe layout algebra.

A Layout is a function from logical coordinates to memory offsets, defined by
a pair (shape, stride). The shape describes the logical domain (how many
elements along each dimension), and the stride describes how coordinates map
to offsets: offset = dot(coord, stride).

    Layout((4, 8), (1, 4))  maps a 4x8 column-major matrix.
    Layout((4, 8), (8, 1))  maps a 4x8 row-major matrix.
    Layout(32, 1)           maps 32 contiguous elements.

What makes CuTe's algebra powerful is that shapes can be hierarchical ---
nested tuples like ((2, 4), (3, 2)) describe multi-level coordinate spaces.
This lets you represent complex GPU memory access patterns (tiles within tiles,
swizzled shared memory banks) as simple shape/stride pairs.

The algebra is built on four key operations:

  compose(A, B)      Function composition: compose(A, B)(i) = A(B(i)).
                     B selects which elements of A to visit, and in what order.

  complement(L)      The "other half": a layout that visits the offsets L skips,
                     so Layout(L, complement(L)) covers every offset once.

  logical_divide(L, T)   Factor L into (tile, rest) using T as the tile shape.
                         Defined as compose(L, Layout(T, complement(T))).

  logical_product(A, B)  Reproduce A's pattern at each position B describes.
                         Defined as Layout(A, compose(complement(A), B)).

Division answers "how do I iterate in tiles?", product answers "how do I
replicate a pattern?", and both are defined in terms of compose + complement.
"""

from __future__ import annotations

import math
from collections.abc import Iterable as IterableType
from typing import Any, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

# Tuple of int | tuple
IntOrIntTuple = Union[int, tuple["IntOrIntTuple", ...]]

__all__ = [
    # Type alias
    "IntOrIntTuple",
    # Type predicates
    "is_tuple", "is_int", "is_scalar", "is_iterable", "is_layout",
    "is_pure_shape", "has_none",
    # Shape conversions
    "as_tuple", "as_shape", "unwrap", "normalize",
    # Core types
    "Layout", "Tile", "Swizzle", "make_swizzle",
    # Stride computation
    "compute_col_major_strides", "compute_row_major_strides",
    # Query functions
    "size", "cosize", "rank", "depth", "mode",
    # Tuple operations
    "concat", "congruent", "compatible",
    "tuple_max", "transform_tuple", "zip_transform",
    "fold", "fold_accumulate", "elem_scale", "inner_product",
    "prefix_product", "suffix_product", "product_each",
    # Layout manipulation
    "append", "prepend", "replace", "group",
    "flatten", "unflatten", "sort", "coalesce",
    # Coordinate conversion
    "idx2crd", "crd2flat", "crd2offset", "crd2idx", "crd2crd",
    "slice_modes", "dice_modes", "slice_and_offset",
    # Core algebra
    "compose", "complement", "logical_divide", "logical_product",
    # Division variants
    "zipped_divide", "tiled_divide", "flat_divide",
    # Product variants
    "zipped_product", "tiled_product", "hier_unzip",
    "blocked_product", "raked_product", "flat_product",
    # Inverse and related
    "right_inverse", "left_inverse", "nullspace",
    "max_common_layout", "max_common_vector",
    # Shape arithmetic
    "safe_div", "shape_div", "shape_mod",
    # Upcast / downcast
    "upcast", "downcast",
]


# =============================================================================
# Type predicates
# =============================================================================
#
# Simple type checks used throughout the algebra.
#


def is_tuple(x) -> bool:
    """Check if x is a tuple (matches CuTe's is_tuple convention)."""
    return isinstance(x, tuple)


def is_int(x) -> bool:
    """Check if x is an integer (excluding booleans which are int subclasses in Python)."""
    return isinstance(x, int) and not isinstance(x, bool)


def is_scalar(x) -> bool:
    """Check if x represents a scalar shape (int, not tuple)."""
    return is_int(x)


def is_iterable(x) -> bool:
    """Check if x is an iterable collection (excluding strings and bytes)."""
    return isinstance(x, IterableType) and not isinstance(x, (str, bytes))


def is_layout(x) -> bool:
    """Check if x is a Layout (matches CuTe's is_layout convention)."""
    return isinstance(x, Layout)


def is_pure_shape(t) -> bool:
    """Check if t is a pure shape (nested ints with no Layouts).

    A pure shape is an int or a tuple containing only ints (recursively).
    This is used to distinguish shape tuples from tiler tuples that may
    contain Layouts.

    Examples:
        is_pure_shape(4) -> True
        is_pure_shape((2, 3)) -> True
        is_pure_shape(((2, 3), 4)) -> True
        is_pure_shape(Layout(4, 1)) -> False
        is_pure_shape((Layout(4, 1), 3)) -> False
    """
    if isinstance(t, Layout):
        return False
    if is_int(t):
        return True
    if is_tuple(t):
        return all(is_pure_shape(elem) for elem in t)
    return False


def has_none(a) -> bool:
    """Determine if None appears at any terminal of an int-tuple.

    Used to detect slice operations in coordinate arguments.

    Examples:
        has_none(3) -> False
        has_none(None) -> True
        has_none((1, None, 3)) -> True
        has_none((1, (2, None))) -> True
    """
    return fold(a, False, lambda acc, v: acc or v is None)

# =============================================================================
# Shape conversions
# =============================================================================
#
#   Function          Direction              When to use
#   ────────────────  ─────────────────────  ──────────────────────────────────
#   as_tuple(x)       int|tuple → tuple      Iterate uniformly over modes
#   as_shape(items)   list → int|tuple       Build result, preserving rank
#   unwrap(t)         (x,) → x               Extract single composed mode
#   normalize(x)      any → int|tuple        Sanitize user input
#


def as_tuple(x) -> tuple:
    """Ensure x is a tuple for uniform iteration.

    Scalars become single-element tuples; tuples pass through unchanged.
    Use this to iterate over modes uniformly:

        for s, d in zip(as_tuple(shape), as_tuple(stride)):
            ...

    Examples:
        as_tuple(8)       → (8,)
        as_tuple((4, 8))  → (4, 8)
    """
    if isinstance(x, int):
        return (x,)
    return tuple(x)


def as_shape(items) -> IntOrIntTuple:
    """Convert a list of modes back to a shape, preserving rank semantics.

    Single-element lists become scalars (rank-0); multi-element become tuples.
    Use this when building computed results from a list of modes:

        result_shapes = [...]  # built up during computation
        return Layout(as_shape(result_shapes), as_shape(result_strides))

    Examples:
        as_shape([8])        → 8        (scalar)
        as_shape([(2, 4)])   → (2, 4)   (still a tuple, just unwrapped from list)
        as_shape([4, 8])     → (4, 8)   (tuple)
    """
    if len(items) == 1:
        return items[0]
    return tuple(items)


def unwrap(t):
    """Unwrap a single-element tuple to its element; pass through otherwise.

    Use this when extracting a single mode from composition, where the result
    might be wrapped in a spurious outer tuple:

        composed = compose(mode_layout, other)
        result_shapes.append(unwrap(composed.shape))

    Examples:
        unwrap((4,))    → 4
        unwrap((4, 8))  → (4, 8)
        unwrap(4)       → 4
    """
    if is_tuple(t) and len(t) == 1:
        return t[0]
    return t


def normalize(x: Any) -> IntOrIntTuple:
    """Normalize user input to a canonical shape: int | tuple[int | tuple, ...].

    - int passes through unchanged
    - iterables (lists, generators) become tuples with normalized elements
    - single-element tuples are preserved (user intent is explicit)

    Used by Layout.__init__ to sanitize user-provided shapes/strides.

    Examples:
        normalize(8)           → 8
        normalize([4, 8])      → (4, 8)
        normalize((4,))        → (4,)      # preserved!
        normalize([[2, 4], 8]) → ((2, 4), 8)
    """
    if is_int(x):
        return x
    if is_iterable(x):
        return tuple(normalize(elem) for elem in x)
    raise TypeError(f"Cannot normalize shape: {type(x).__name__}")


# =============================================================================
# Layout
# =============================================================================
#
# A Layout is a function from logical coordinates to memory offsets, defined by
# a pair (shape, stride). Each "mode" (dimension) contributes coord_i * stride_i
# to the offset. When a shape element is itself a tuple, that mode has sub-modes,
# creating the hierarchical coordinate spaces that are CuTe's key innovation.
#


class Layout:
    """A function from logical coordinates to memory offsets: offset = sum(coord_i * stride_i).

    A Layout is defined by (shape, stride) where shape describes the logical
    domain and stride describes the memory step for each dimension.

    Examples:
        Layout((4, 8), (1, 4))   -- 4x8 column-major matrix
        Layout((4, 8), (8, 1))   -- 4x8 row-major matrix
        Layout(32, 1)            -- 32 contiguous elements
        Layout((4, 8), (2, 0))   -- strided rows, broadcast columns

    Shapes can be hierarchical (nested tuples):
        Layout(((2, 4), 8), ((1, 2), 8))   -- a 2x4 tile within an 8-column layout

    This hierarchy lets you describe complex GPU memory patterns --- tiles within
    tiles, swizzled banks, interleaved threads --- as simple shape/stride pairs.

    Shapes and strides are stored as int | tuple:
    - int for scalar (1D) shapes
    - tuple for multi-dimensional shapes

    Swizzled layouts:
        When composed with a Swizzle, a Layout stores the swizzle function and
        applies it after computing the linear offset. This keeps composition
        closed: compose(Swizzle, Layout) returns a Layout.

    Construction:
        Layout(shape)              -- column-major strides computed automatically
        Layout(shape, stride)      -- explicit shape and stride
        Layout(layout_a, layout_b) -- bundle two layouts as modes of a new layout
    """

    def __init__(self, *args, swizzle: "Swizzle | None" = None):
        self._swizzle = swizzle

        if len(args) == 0:
            self._shape = ()
            self._stride = ()

        elif all(isinstance(arg, Layout) for arg in args):
            if len(args) == 1:
                # Wrap the inner layout's shape/stride to add one level of nesting
                inner = args[0]
                self._shape = normalize((inner.shape,))
                self._stride = normalize((inner.stride,))
            else:
                shapes = tuple(layout.shape for layout in args)
                strides = tuple(layout.stride for layout in args)
                self._shape = shapes
                self._stride = strides

        elif len(args) == 1:
            shape = args[0]
            self._shape = normalize(shape)
            self._stride = compute_col_major_strides(self._shape)

        elif len(args) == 2:
            shape, stride = args
            self._shape = normalize(shape)
            self._stride = normalize(stride)

        else:
            raise TypeError(
                "Layout() takes shapes/stride arguments or multiple Layout arguments for bundling"
            )

        if not congruent(self._shape, self._stride):
            raise ValueError(
                f"Shape {self._shape} and Stride {self._stride} are not congruent"
            )

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return False
        return (
            self.shape == other.shape
            and self.stride == other.stride
            and self._swizzle == other._swizzle
        )

    def __hash__(self):
        swizzle_hash = None
        if self._swizzle is not None:
            swizzle_hash = (self._swizzle.bits, self._swizzle.base, self._swizzle.shift)
        return hash((self.shape, self.stride, swizzle_hash))

    def __repr__(self):
        def fmt(x):
            """Format shape/stride: int as-is, tuple with parens."""
            if isinstance(x, int):
                return str(x)
            return repr(x)
        base_repr = f"{fmt(self._shape)} : {fmt(self._stride)}"
        if self._swizzle is not None:
            return f"({self._swizzle}) o ({base_repr})"
        return base_repr

    @property
    def shape(self) -> IntOrIntTuple:
        return self._shape

    @property
    def stride(self) -> IntOrIntTuple:
        return self._stride

    @property
    def swizzle(self) -> "Swizzle | None":
        """The swizzle function applied after computing linear offset, or None."""
        return self._swizzle

    @staticmethod
    def _calculate_max_offset(shape: Any, stride: Any) -> int:
        if is_tuple(shape):
            return sum(
                Layout._calculate_max_offset(s, d) for s, d in zip(shape, stride)
            )
        return (shape - 1) * stride

    def __call__(self, *args):
        """Map a logical coordinate to a linear index, or slice the layout.

        If any coordinate is None, returns a sublayout (the sliced dimensions).
        Otherwise returns the integer offset.

        For swizzled layouts, the swizzle function is applied after computing
        the linear offset.

        Examples:
            Layout((4,8))((2,3)) -> 26       # coordinate to index
            Layout((4,8))(None, 3) -> (4,) : (1,)  # slice: fix dim 1 to 3, keep dim 0
        """
        if len(args) == 1:
            coords = args[0]
        else:
            coords = args
        if has_none(coords):
            sliced_shape = slice_modes(coords, self.shape)
            sliced_stride = slice_modes(coords, self.stride)
            # slice_modes returns tuples; preserve structure (matching pycute behavior)
            if not sliced_shape:
                return Layout((), (), swizzle=self._swizzle)
            return Layout(sliced_shape, sliced_stride, swizzle=self._swizzle)
        linear_offset = crd2offset(coords, self.shape, self.stride)
        if self._swizzle is not None:
            return self._swizzle(linear_offset)
        return linear_offset

    def squeeze(self) -> Self:
        """Removes all dimensions of size 1 and their corresponding strides."""
        new_shape, new_stride = self.filter_shapes(self.shape, self.stride, 1)
        return Layout(new_shape, new_stride, swizzle=self._swizzle)

    def filter(self) -> "Layout":
        """Removes all dimensions with a stride of 0."""
        new_shape, new_stride = self.filter_strides(self.shape, self.stride, 0)
        return Layout(new_shape, new_stride, swizzle=self._swizzle)

    def filter_shapes(self, shape, stride, target):
        """Removes all dimensions of size 'target', and their corresponding strides."""
        if is_int(shape):
            if shape == target:
                return (), ()
            return shape, stride

        s_out = []
        d_out = []
        for s, d in zip(shape, stride):
            if is_tuple(s):
                sub_s, sub_d = self.filter_shapes(s, d, target)
                if sub_s != ():
                    s_out.append(sub_s)
                    d_out.append(sub_d)
            elif s != target:
                s_out.append(s)
                d_out.append(d)
        return as_shape(s_out) if s_out else (), as_shape(d_out) if d_out else ()

    def filter_strides(self, shape, stride, target):
        """Removes all dimensions with a stride of 'target', and their corresponding shapes."""
        if is_int(shape):
            if stride == target:
                return (), ()
            return shape, stride

        s_out = []
        d_out = []
        for s, d in zip(shape, stride):
            if is_tuple(s):

                sub_s, sub_d = self.filter_strides(s, d, target)
                if sub_s != ():
                    s_out.append(sub_s)
                    d_out.append(sub_d)
            elif d != target:
                s_out.append(s)
                d_out.append(d)
        return as_shape(s_out) if s_out else (), as_shape(d_out) if d_out else ()


def compute_col_major_strides(shape: IntOrIntTuple) -> IntOrIntTuple:
    """Compute column-major (leftmost-fastest) strides for a shape.

    Each element gets stride equal to the product of all preceding elements,
    making the first (leftmost) mode vary fastest --- like Fortran/column-major order.
    """
    strides = prefix_product(shape)
    return _zero_leading_unit_strides(shape, strides)


def compute_row_major_strides(shape: IntOrIntTuple) -> IntOrIntTuple:
    """Compute row-major (rightmost-fastest) strides for a shape.

    Each element gets stride equal to the product of all following elements,
    making the last (rightmost) mode vary fastest --- like C/row-major order.
    """
    return suffix_product(shape)


def _zero_leading_unit_strides(shape, strides):
    """CuTe convention: leading size-1 modes get stride 0 instead of 1."""
    if is_int(shape):
        if shape == 1 and strides == 1:
            return 0
        return strides

    result = []
    still_leading = True
    for s, d in zip(shape, strides):
        if is_tuple(s):
            if still_leading:
                sub = _zero_leading_unit_strides(s, d)
                result.append(sub)
                if size(s) != 1:
                    still_leading = False
            else:
                result.append(d)
        else:
            if still_leading and s == 1 and d == 1:
                result.append(0)
            else:
                result.append(d)
                if s != 1:
                    still_leading = False
    return tuple(result)

# =============================================================================
# Query functions: size, rank, depth, mode
# =============================================================================
#
# These functions query properties of shapes and layouts:
#   size  -- total number of elements (product of all shape elements)
#   cosize -- memory span (max offset + 1, the size of the codomain)
#   rank  -- number of top-level modes (dimensions)
#   depth -- nesting depth of the shape hierarchy
#   mode  -- extract a single mode (dimension) from a shape or layout
#

def size(obj: Any) -> int:
    """Returns the logical number of elements (product of shape)."""
    if isinstance(obj, Layout):
        return size(obj.shape)
    if is_tuple(obj) or is_int(obj):
        return fold(obj, 1, lambda acc, x: acc * x)
    raise TypeError(f"Cannot calculate size of {type(obj).__name__}")


def cosize(obj: Layout) -> int:
    """Returns the memory span (max_offset + 1)."""
    if is_int(obj.shape):
        return obj._calculate_max_offset(obj.shape, obj.stride) + 1
    if len(obj.shape) == 0:
        return 1
    return obj._calculate_max_offset(obj.shape, obj.stride) + 1


def rank(obj: Any) -> int:
    if is_tuple(obj):
        return len(obj)
    if isinstance(obj, Layout):
        if is_int(obj.shape):
            return 0
        return len(obj.shape)
    if is_int(obj):
        return 0
    raise TypeError(f"Cannot calculate rank of {type(obj).__name__}")


def depth(obj: Any) -> int:
    """Calculate nesting depth of a shape/layout.

    - int has depth 0
    - tuple has depth 1 + max depth of its elements
    - Layout delegates to its shape
    """
    if isinstance(obj, Layout):
        return depth(obj.shape)
    if is_int(obj):
        return 0
    if is_tuple(obj):
        if not obj:
            return 0
        return 1 + max((depth(elem) for elem in obj), default=0)
    raise TypeError(f"Cannot calculate depth of {type(obj).__name__}")


def mode(obj: Any, idx):
    if is_tuple(obj):
        if not obj:
            return ()
        return obj[idx]
    if isinstance(obj, Layout):
        if is_int(obj.shape):
            if idx != 0:
                raise IndexError(f"Index {idx} out of range for scalar layout")
            return obj
        return Layout(obj.shape[idx], obj.stride[idx])
    raise TypeError(f"Cannot get mode of {type(obj).__name__}")


def concat(t1: Any, t2: Any):
    if is_tuple(t1) and is_tuple(t2):
        return t1 + t2
    if isinstance(t1, Layout) and isinstance(t2, Layout):
        return Layout(as_tuple(t1.shape) + as_tuple(t2.shape),
                      as_tuple(t1.stride) + as_tuple(t2.stride))
    raise TypeError(
        f"Cannot concatenate objects of {type(t1).__name__} and {type(t2).__name__}"
    )


def congruent(a: IntOrIntTuple, b: IntOrIntTuple) -> bool:
    """Returns True if two layouts have the same rank and structure.

    Matches CuTe's congruent(): tests if two tuples have the same profile
    (hierarchical rank division).  Congruent shapes can be element-wise
    zipped (like zip_transform).

    Examples:
        congruent((2, 3), (4, 5))     -> True   (same rank)
        congruent((2, 3), 6)          -> False  (int vs tuple)
        congruent(((2, 3), 4), ((5, 6), 7))  -> True   (same nesting)
    """
    if isinstance(a, int) and isinstance(b, int):
        return True
    if is_tuple(a) and is_tuple(b):
        return len(a) == len(b) and all(congruent(sa, sb) for sa, sb in zip(a, b))
    return False


def compatible(a: IntOrIntTuple, b: IntOrIntTuple) -> bool:
    """Checks if shape A is compatible with shape B.

    Matches CuTe's compatible(): A is compatible with B if size(A) == size(B)
    and any coordinate into A can also be used as a coordinate into B.
    This is a partial order: A <= B.

    A is compatible with B if A's modes can be grouped to match B's structure.

    Examples:
        compatible((2, 2, 3), (4, 3))  -> True   (2*2 groups into 4)
        compatible(12, (2, 2, 3))      -> True   (scalar is compatible with any shape)
        compatible((2, 2, 3), (5, 2))  -> False  (sizes don't match)
    """
    if size(a) != size(b):
        return False

    if is_scalar(a):
        return True
    if is_scalar(b):
        return False

    if len(a) == len(b):
        return all(compatible(sa, sb) for sa, sb in zip(a, b))

    return _can_group_a_into_b(list(a), b)


def _can_group_a_into_b(a_modes: list, b) -> bool:
    """Check if A's modes can be consumed/grouped to match B's structure."""
    if is_scalar(b):
        target_size = size(b)
        acc_size = 1
        while acc_size < target_size and a_modes:
            acc_size *= size(a_modes.pop(0))
        return acc_size == target_size

    if is_tuple(b):
        return all(_can_group_a_into_b(a_modes, sub_b) for sub_b in b) and len(a_modes) == 0

    return False


# =============================================================================
# Layout manipulation: append, prepend, replace, group, flatten, sort
# =============================================================================
#
# These functions restructure a layout's modes without changing the underlying
# mapping. Flatten removes hierarchy, sort reorders by stride, group nests
# adjacent modes. They are the structural building blocks for composition
# and coalescing.
#


def append(a: Layout, b: Layout) -> Layout:
    """Appends layout b as a new mode at the end of layout a.

    append(3:1, 4:3) -> (3,4):(1,3)
    append((3,4):(1,3), (3,4):(1,3)) -> (3,4,(3,4)):(1,3,(1,3))
    """
    return Layout(as_tuple(a.shape) + (b.shape,), as_tuple(a.stride) + (b.stride,))


def prepend(a: Layout, b: Layout) -> Layout:
    """Prepends layout b as a new mode at the beginning of layout a.

    prepend(3:1, 4:3) -> (4,3):(3,1)
    """
    return Layout((b.shape,) + as_tuple(a.shape), (b.stride,) + as_tuple(a.stride))


def replace(layout: Layout, idx: int, new_layout: Layout) -> Layout:
    """Replaces the mode at index idx with new_layout.

    replace((3,4,(3,4)):(1,3,(1,3)), 2, 4:3) -> (3,4,4):(1,3,3)
    """
    shapes = list(as_tuple(layout.shape))
    strides = list(as_tuple(layout.stride))

    shapes[idx] = new_layout.shape
    strides[idx] = new_layout.stride

    return Layout(tuple(shapes), tuple(strides))


def group(layout: Layout, start: int, end: int) -> Layout:
    """Groups modes from index start to end (exclusive) into a nested tuple.

    group((2,3,5,7):(1,2,6,30), 0, 2) -> ((2,3),5,7):((1,2),6,30)
    group(((2,3),5,7):((1,2),6,30), 1, 3) -> ((2,3),(5,7)):((1,2),(6,30))
    """
    r = rank(layout)
    if start < 0 or end > r or start >= end:
        raise ValueError(
            f"Invalid group range [{start}, {end}) for layout of rank {r}"
        )

    shapes = list(as_tuple(layout.shape))
    strides = list(as_tuple(layout.stride))

    # Extract the modes to group
    grouped_shape = tuple(shapes[start:end])
    grouped_stride = tuple(strides[start:end])

    # Build new layout: [0:start] + [grouped] + [end:]
    new_shapes = shapes[:start] + [grouped_shape] + shapes[end:]
    new_strides = strides[:start] + [grouped_stride] + strides[end:]

    return Layout(tuple(new_shapes), tuple(new_strides))


def flatten(obj: Any) -> Any:
    """Flattens a hierarchical layout into a rank-N flat layout."""

    def _flatten(s):
        if is_int(s):
            return (s,)
        flat = []
        for si in s:
            if is_tuple(si):
                s_rec = _flatten(si)
                flat.extend(s_rec)
            else:
                flat.append(si)
        return tuple(flat)

    if is_int(obj):
        return (obj,)
    if is_tuple(obj):
        return _flatten(obj)
    elif isinstance(obj, Layout):
        flat_shape = _flatten(obj.shape)
        flat_stride = _flatten(obj.stride)
        return Layout(as_shape(list(flat_shape)), as_shape(list(flat_stride)))
    else:
        raise TypeError(f"Cannot flatten object of type {type(obj).__name__}")


def unflatten(obj, target_profile):
    """Unflatten a flat object to match a target's hierarchical structure.

    This is the inverse of flatten: it reshapes a flat tuple or layout into
    a hierarchical structure matching target_profile.

    Args:
        obj: A flat tuple or Layout
        target_profile: A (possibly nested) tuple or Layout defining the
                        desired structure

    Returns:
        A tuple or Layout with the structure of target_profile

    Examples:
        unflatten((1,2,3,4,5), ((0,0), (0,0,0))) -> ((1,2), (3,4,5))
        unflatten(Layout((2,3,5,7), (1,2,6,30)), (4, 3))
            -> Layout((2,3), (5,7)), ((1,2), (6,30)))

    Preconditions:
        flatten(obj) == obj  (obj must already be flat)
        rank(flatten(target_profile)) == rank(obj)
    """
    def _unflatten_helper(flat_tuple, profile):
        """Consume elements from flat_tuple to match profile's structure."""
        if is_tuple(profile):
            result = []
            remaining = list(flat_tuple)
            for elem in profile:
                sub_result, remaining = _unflatten_helper(remaining, elem)
                result.append(sub_result)
            return tuple(result), remaining
        else:
            return flat_tuple[0], flat_tuple[1:]

    if isinstance(target_profile, Layout):
        target_profile = target_profile.shape

    if isinstance(obj, Layout):
        new_shape, remaining_s = _unflatten_helper(tuple(obj.shape), target_profile)
        new_stride, remaining_d = _unflatten_helper(tuple(obj.stride), target_profile)
        if len(remaining_s) != 0:
            raise ValueError(f"Rank mismatch: leftover shape elements {remaining_s}")
        if len(remaining_d) != 0:
            raise ValueError(f"Rank mismatch: leftover stride elements {remaining_d}")
        return Layout(new_shape, new_stride)

    if is_tuple(obj):
        result, remaining = _unflatten_helper(tuple(obj), target_profile)
        if len(remaining) != 0:
            raise ValueError(f"Rank mismatch: leftover elements {remaining}")
        return result

    raise TypeError(f"Cannot unflatten object of type {type(obj).__name__}")


def product_each(shape: Any) -> tuple:
    """Compute the product of each top-level mode of a shape.

    Flattens nested shape elements to get the size of each top-level mode.
    This is useful when you need the "effective" size of each mode after
    flattening any internal structure.

    Args:
        shape: A shape (int or tuple, possibly nested)

    Returns:
        A tuple where each element is the product of the corresponding
        top-level mode. If input is an int, returns (shape,).

    Examples:
        product_each((4, 8))       -> (4, 8)
        product_each(((2, 2), 8))  -> (4, 8)    # 2*2 = 4
        product_each((3, (2, 4)))  -> (3, 8)    # 2*4 = 8
        product_each(16)           -> (16,)
    """
    if is_int(shape):
        return (shape,)
    return tuple(size(s) for s in shape)


def sort(obj: Layout) -> Layout:
    """Returns a new Layout with modes sorted by stride."""
    if rank(obj) <= 1:
        return obj

    flat = flatten(obj)
    combined = list(zip(flat.stride, flat.shape))
    combined.sort()
    new_stride = tuple(item[0] for item in combined)
    new_shape = tuple(item[1] for item in combined)

    return Layout(new_shape, new_stride)


# =============================================================================
# Tuple arithmetic: tuple_max, elem_scale, inner_product, prefix_product
# =============================================================================
#
# Arithmetic operations that work element-wise on hierarchical int-tuples.
# These mirror their scalar counterparts but respect the nested structure.
# prefix_product is particularly important: it computes column-major strides
# from a shape, which is how Layout(shape) auto-computes its strides.
#

def tuple_max(a: Any) -> int:
    """Return the maximum value across all terminals of a (possibly nested) int-tuple.

    Examples:
        tuple_max(5) -> 5
        tuple_max((3, 7, 2)) -> 7
        tuple_max(((1, 9), (4, 2))) -> 9
    """
    return fold(a, -float('inf'), lambda acc, x: max(acc, x))


def transform_tuple(t: Any, f) -> Any:
    """Apply f to each leaf element of a (possibly nested) tuple.

    Recursively descends into nested tuples, applying f only to
    non-tuple elements (integers). Preserves the hierarchical structure.

    Examples:
        transform_tuple(5, lambda x: x*2) -> 10
        transform_tuple((3, 4), lambda x: x*2) -> (6, 8)
        transform_tuple(((2, 3), 4), lambda x: x+1) -> ((3, 4), 5)
    """
    if is_tuple(t):
        return tuple(transform_tuple(elem, f) for elem in t)
    return f(t)


def zip_transform(a: Any, b: Any, f) -> Any:
    """Apply f(a_i, b_i) element-wise to two congruent tuples.

    Both arguments must have the same structure (same nesting and lengths).
    Recursively descends into nested tuples, applying f to paired leaf elements.

    Examples:
        zip_transform(2, 3, lambda x, y: x*y) -> 6
        zip_transform((1, 2), (3, 4), lambda x, y: x+y) -> (4, 6)
        zip_transform(((1, 2), 3), ((4, 5), 6), lambda x, y: x*y) -> ((4, 10), 18)
    """
    if is_tuple(a):
        if not is_tuple(b) or len(a) != len(b):
            raise ValueError(f"Structure mismatch: {a} vs {b}")
        return tuple(zip_transform(ai, bi, f) for ai, bi in zip(a, b))
    return f(a, b)


def fold(t: Any, init: Any, f) -> Any:
    """Left-fold a (possibly nested) tuple with an initial value and binary function.

    Recursively descends into nested tuples, applying f only to leaf elements.
    Reduces from left to right: f(f(f(init, leaf0), leaf1), leaf2)...
    For scalars, returns f(init, t).

    This is useful for accumulating results across all elements of a shape/stride.

    Examples:
        fold(5, 0, lambda acc, x: acc + x) -> 5
        fold((1, 2, 3), 0, lambda acc, x: acc + x) -> 6
        fold(((1, 2), 3), 0, lambda acc, x: acc + x) -> 6
        fold((2, 3, 4), 1, lambda acc, x: acc * x) -> 24
    """
    if is_tuple(t):
        acc = init
        for elem in t:
            acc = fold(elem, acc, f)
        return acc
    return f(init, t)


def fold_accumulate(t: Any, init: Any, f, update) -> Any:
    """Left-fold a tuple, collecting intermediate results while threading state.

    Like fold, but returns a tuple of the same structure containing the result
    at each position. The state is updated via `update` after each element.

    Implements the pattern:
        fold_accumulate((a, b, c), v, f, u) = (f(a, v), f(b, u(a, v)), f(c, u(b, u(a, v))))

    Args:
        t: A (possibly nested) tuple to fold over
        init: Initial state value
        f: (element, state) -> result for each element
        update: (element, state) -> new_state for the next element

    Examples:
        # Prefix product (computing strides from shapes):
        fold_accumulate((2, 3, 4), 1,
                        f=lambda elem, state: state,
                        update=lambda elem, state: state * elem)
        # -> (1, 2, 6)  — each result is the product of all prior elements

        # shape_div uses this to divide a shape by a divisor:
        #   f: ceil(element / divisor)  — divide this mode
        #   update: divisor / size(element)  — carry remainder to next mode
        # shape_div((2, 3, 4), 6) -> (1, 1, 4)
        #   mode 0: ceil(2/6)=1, remaining divisor=6/2=3
        #   mode 1: ceil(3/3)=1, remaining divisor=3/3=1
        #   mode 2: ceil(4/1)=4, done
    """
    if isinstance(t, int):
        return f(t, init)

    if not is_tuple(t) or len(t) == 0:
        return t

    results = []
    state = init
    for elem in t:
        results.append(fold_accumulate(elem, state, f, update))
        state = update(elem, state)

    return tuple(results)


def elem_scale(a: Any, b: Any) -> Any:
    """Element-wise scale of int-tuple a by int-tuple b.

    For scalars: a * b.
    For tuple a, scalar b: error (ambiguous).
    For scalar a, tuple b: a * product(b).
    For tuple a, tuple b: pairwise elem_scale.

    Examples:
        elem_scale(3, 4) -> 12
        elem_scale(2, (3, 4)) -> 24   (2 * 12)
        elem_scale((2, 3), (4, 5)) -> (8, 15)
    """
    if is_tuple(a):
        if is_tuple(b):
            return zip_transform(a, b, elem_scale)
        else:
            raise TypeError("Cannot elem_scale tuple by scalar (ambiguous)")
    else:
        if is_tuple(b):
            return elem_scale(a, size(b))
        else:
            return a * b


def inner_product(a: Any, b: Any) -> int:
    """Compute the inner product of two int-tuples.

    For scalars: a * b
    For tuples: sum of pairwise inner products.

    Examples:
        inner_product(2, 3) -> 6
        inner_product((1, 2), (3, 2)) -> 7
        inner_product(((2, 3), 4), ((2, 1), 2)) -> 15
    """
    if is_tuple(a):
        if not is_tuple(b) or len(a) != len(b):
            raise ValueError(f"Structure mismatch: {a} vs {b}")
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError(f"Expected int, got {type(a).__name__} and {type(b).__name__}")
        return a * b


def prefix_product(a: Any, init: Any = 1) -> Any:
    """Compute the exclusive prefix product of an int-tuple.

    Returns a tuple of the same structure where each element is replaced
    by the product of all preceding elements (starting from init).

    For scalars: returns init (the prefix before the scalar).
    For tuples: recursively computes prefix products with carry.

    Examples:
        prefix_product(2) -> 1
        prefix_product((3, 2)) -> (1, 3)
        prefix_product((3, 2, 4)) -> (1, 3, 6)
        prefix_product(((2, 3), 4)) -> ((1, 2), 6)
        prefix_product(((2, 3), (2, 1, 2), (5, 2, 1))) -> ((1, 2), (6, 12, 12), (24, 120, 240))
    """
    if is_tuple(a):
        if is_tuple(init):
            if len(a) != len(init):
                raise ValueError(f"Length mismatch: {len(a)} vs {len(init)}")
            return zip_transform(a, init, prefix_product)
        else:
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * size(v)
            return tuple(r)
    else:
        if is_tuple(init):
            raise TypeError("Cannot apply tuple init to scalar shape")
        return init


def suffix_product(a: Any, init: Any = 1) -> Any:
    """Compute the exclusive suffix product of an int-tuple.

    Returns a tuple of the same structure where each element is replaced
    by the product of all following elements (ending with init).

    For scalars: returns init (the suffix after the scalar).
    For tuples: recursively computes suffix products with carry from the right.

    Examples:
        suffix_product(2) -> 1
        suffix_product((3, 2)) -> (2, 1)
        suffix_product((3, 2, 4)) -> (8, 4, 1)
        suffix_product(((2, 3), 4)) -> ((12, 4), 1)
        suffix_product((3, (2, 4))) -> (8, (4, 1))
    """
    if is_tuple(a):
        if is_tuple(init):
            if len(a) != len(init):
                raise ValueError(f"Length mismatch: {len(a)} vs {len(init)}")
            return zip_transform(a, init, suffix_product)
        else:
            r = []
            carry = init
            for v in reversed(a):
                r.append(suffix_product(v, carry))
                carry = carry * size(v)
            return tuple(reversed(r))
    else:
        if is_tuple(init):
            raise TypeError("Cannot apply tuple init to scalar shape")
        return init


# =============================================================================
# Coalescing
# =============================================================================
#
# Coalescing merges contiguous modes. Two adjacent modes are contiguous when
# stride[i+1] == shape[i] * stride[i], meaning they cover a contiguous range
# of offsets. Merging them into one larger mode simplifies the layout without
# changing the mapping. Coalescing is the canonical simplification: it is
# always safe and always preserves semantics.
#

def coalesce(obj: Layout, profile: Any = None) -> Layout:
    """Returns a new Layout where contiguous dimensions are merged.

    Args:
        obj: The layout to coalesce
        profile: Optional shape profile that defines mode boundaries.
                 When provided, coalescing happens within each mode independently,
                 preserving the hierarchical structure defined by the profile.

    Examples:
        coalesce(Layout((2,4), (1,2))) -> Layout(8, 1)
        coalesce(Layout((2,4,2,2), (1,2,8,16)), (4,4)) -> Layout((8,4), (1,8))
    """
    if rank(obj) == 0:
        if is_int(obj.shape):
            return Layout(1, 0) if obj.shape == 1 else obj
        return Layout()

    if profile is None:
        return _coalesce_flat(obj)

    return _coalesce_by_mode(obj, profile if is_tuple(profile) else (profile,))


def _coalesce_flat(obj: Layout) -> Layout:
    """Coalesce a layout by filtering trivial modes and merging contiguous ones."""
    flat = flatten(obj)

    if is_int(flat.shape):
        return Layout(1, 0) if flat.shape == 1 else flat

    shapes = list(flat.shape)
    strides = list(flat.stride)

    # Filter and merge in one pass: skip size-1 modes, merge contiguous ones
    merged_s, merged_d = [], []
    for s, d in zip(shapes, strides):
        if s == 1:
            continue
        if merged_s and d == merged_s[-1] * merged_d[-1]:
            merged_s[-1] *= s
        else:
            merged_s.append(s)
            merged_d.append(d)

    if not merged_s:
        return Layout(1, 0)
    return Layout(as_shape(merged_s), as_shape(merged_d))


def _coalesce_by_mode(layout: Layout, profile: tuple) -> Layout:
    """Coalesce a layout respecting mode boundaries defined by profile.

    If profile contains None, coalesce each original mode independently.
    Otherwise, partition the flattened layout by profile sizes and coalesce each partition.
    """
    profile_list = list(profile)

    # None-profile: coalesce each original mode independently
    if any(p is None for p in profile_list):
        result_s, result_d = [], []
        for i in range(len(profile_list)):
            if i >= rank(layout):
                result_s.append(1)
                result_d.append(0)
            else:
                coalesced = _coalesce_flat(Layout(mode(layout.shape, i), mode(layout.stride, i)))
                result_s.append(coalesced.shape)
                result_d.append(coalesced.stride)
        return Layout(as_shape(result_s), as_shape(result_d))

    # Int-profile: partition flattened layout by profile sizes
    flat = flatten(layout)
    flat_shapes, flat_strides = list(flat.shape), list(flat.stride)

    # Flatten profile to get target sizes
    target_sizes = list(flatten(profile))

    result_s, result_d = [], []
    idx = 0

    for target_size in target_sizes:
        # Consume modes until we reach target_size
        mode_s, mode_d = [], []
        accumulated = 1
        while accumulated < target_size and idx < len(flat_shapes):
            mode_s.append(flat_shapes[idx])
            mode_d.append(flat_strides[idx])
            accumulated *= flat_shapes[idx]
            idx += 1

        if not mode_s:
            result_s.append(1)
            result_d.append(0)
            continue

        # Sort by stride, filter size-1, merge contiguous (with nonzero stride check)
        paired = sorted(zip(mode_d, mode_s))
        merged_s, merged_d = [], []
        for d, s in paired:
            if s == 1:
                continue
            if merged_s and merged_d[-1] != 0 and d == merged_s[-1] * merged_d[-1]:
                merged_s[-1] *= s
            else:
                merged_s.append(s)
                merged_d.append(d)

        if not merged_s:
            result_s.append(1)
            result_d.append(0)
        else:
            result_s.append(as_shape(merged_s))
            result_d.append(as_shape(merged_d))

    return Layout(tuple(result_s), tuple(result_d))


# =============================================================================
# Complement, inverse, and slice operations
# =============================================================================
#
# The complement of a layout fills in the gaps. If layout L visits offsets
# {0, 2, 4, 6}, then complement(L) visits {0, 1} (the offsets within each
# stride gap). Together, make_ayout(L, complement(L)) covers every offset
# exactly once. This is the key building block for logical_divide.
#
# The right-inverse R of L satisfies L(R(i)) = i: it "undoes" L.
# The left-inverse R satisfies R(L(i)) = i: it recovers coordinates from offsets.
#
# Slicing fixes some coordinates and returns a sublayout over the remaining
# free dimensions, much like NumPy's array[3, :, :] syntax.
#

def complement(layout: Layout, cosize_bound: Any = None) -> Layout:
    """Compute the complement of a layout: a layout that fills in the gaps.

    If L visits offsets {0, 2, 4, 6} within a range of 8, then complement(L, 8)
    visits the in-between offsets {0, 1} (stride-1 within each stride-2 gap).
    Together, Layout(L, complement(L)) covers every offset exactly once.

    Why "complement"?  Think of L as selecting a subset of [0, cosize).
    The complement fills "the rest" — not by set subtraction, but by filling
    the stride gaps.  The bundled Layout(L, complement(L)) is a bijection
    onto [0, cosize), with L controlling position within each gap, and
    complement(L) controlling which gap.

    This is the key building block for logical_divide: dividing a layout by a
    tiler T is equivalent to composing with Layout(T, complement(T)).

    The algorithm sorts the layout's modes by stride, then folds _step_mode
    over them.  Each step checks for a gap between the current frontier and
    the next mode's stride; gaps become output modes.  A final step fills
    from the last mode's cosize up to cosize_bound.

    Args:
        layout: The layout to compute complement for
        cosize_bound: The target cosize. Defaults to cosize(layout).

    Examples:
        complement(Layout(4, 2), 16) -> Layout((2, 2), (1, 8))
        complement(Layout(4, 1), 16) -> Layout(4, 4)
        complement(Layout((2, 2), (1, 4)), 16) -> Layout((2, 2), (2, 8))
    """

    def _step_mode(current_stride, stride, shape):
        """Emit a gap-fill if there's a gap before this mode, then advance
        past it.  Returns (gap_size, next_current_stride)."""
        gap_size = stride // current_stride if stride > current_stride else 1
        return gap_size, stride * shape

    # Handle cosize_bound as a shape (tuple) - use its size
    if cosize_bound is None:
        cosize_bound = cosize(layout)
    elif is_tuple(cosize_bound):
        cosize_bound = size(cosize_bound)

    # Handle empty layout (empty tuple shape)
    if is_tuple(layout.shape) and len(layout.shape) == 0:
        return Layout(cosize_bound, 1) if cosize_bound > 1 else Layout()

    # Flatten, filter size-1 and stride-0 dims, sort by stride
    flat = flatten(layout)

    # Convert to lists for uniform processing
    if is_int(flat.shape):
        flat_shapes = [flat.shape]
        flat_strides = [flat.stride]
    else:
        flat_shapes = list(flat.shape)
        flat_strides = list(flat.stride)

    modes = sorted(
        ((d, s) for s, d in zip(flat_shapes, flat_strides) if s != 1 and d != 0)
    )

    # Fold _step_mode over sorted modes, collecting gap-fills
    result_shapes = []
    result_strides = []
    current_stride = 1

    for stride, shape in modes:
        gap_size, next_stride = _step_mode(current_stride, stride, shape)
        if gap_size > 1:
            result_shapes.append(gap_size)
            result_strides.append(current_stride)
        current_stride = next_stride

    # Fill remaining space up to cosize_bound (ceiling division).
    # Always append (even if shape-1) to match pycute; coalesce cleans up.
    remaining = _ceil_div(cosize_bound, current_stride)
    result_shapes.append(remaining)
    result_strides.append(current_stride)

    # Coalesce the result (merges contiguous modes, removes size-1 modes)
    return coalesce(Layout(as_shape(result_shapes), as_shape(result_strides)))


def right_inverse(layout: Any) -> Layout:
    """Compute the right-inverse of a layout.

    For a layout L, the right-inverse R satisfies: L(R(i)) == i
    for all i in range(size(R)).

    The algorithm sorts modes by stride and folds _step_mode over them,
    greedily building the longest contiguous prefix.  Each step checks
    whether the mode's stride matches the running frontier; if so the
    mode contributes to the inverse, otherwise iteration stops.

    Examples:
        right_inverse(Layout(4, 1)) -> Layout(4, 1)
        right_inverse(Layout(4, 2)) -> Layout((2, 2), (0, 1))
        right_inverse(Layout((8, 4), (1, 8))) -> Layout((8, 4), (1, 8))
        right_inverse(Layout((8, 4), (4, 1))) -> Layout((4, 8), (1, 4))
    """

    def _step_mode(current_idx, stride, shape):
        """Check if a mode is contiguous with the frontier.
        Returns (contiguous, next_current_idx)."""
        if shape == 1:
            return True, current_idx
        if current_idx != stride:
            return False, current_idx
        return True, shape * stride

    if layout is None:
        return None
    if isinstance(layout, int):
        return Layout(layout)

    flat = flatten(layout)

    # Handle scalar layouts
    if is_int(flat.shape):
        flat_shapes = [flat.shape]
        flat_strides = [flat.stride]
    else:
        flat_shapes = list(flat.shape)
        flat_strides = list(flat.stride)

    # Compute prefix products for inverse strides
    pp = prefix_product(flat.shape)
    if is_int(pp):
        pp = [pp]
    else:
        pp = list(pp)

    # Sort (stride, shape, prefix_prod) triples by stride
    triples = sorted(zip(flat_strides, flat_shapes, pp))

    result_shape = []
    result_stride = []
    current_idx = 1

    for stride, shape, rstride in triples:
        contiguous, current_idx = _step_mode(current_idx, stride, shape)
        if not contiguous:
            break
        if shape != 1:
            result_shape.append(shape)
            result_stride.append(rstride)

    if not result_shape:
        return Layout(1, 0)

    return coalesce(Layout(
        tuple(result_shape),
        tuple(result_stride)
    ))


def left_inverse(layout: Any) -> Layout:
    """Compute the left-inverse of a layout.

    For a layout L, the left-inverse R satisfies: R(L(i)) == i
    for all i in range(size(L)).

    Computed as: right_inverse(Layout(L, complement(L)))

    Examples:
        left_inverse(Layout(4, 1)) -> Layout(4, 1)
        left_inverse(Layout(4, 2)) -> Layout((2, 4), (0, 1))
        left_inverse(Layout((8, 4), (1, 8))) -> Layout(32, 1)
    """
    if layout is None:
        return None
    if isinstance(layout, int):
        return Layout(layout)

    comp = complement(layout)
    combined = Layout(
        (layout.shape, comp.shape),
        (layout.stride, comp.stride),
    )
    return right_inverse(combined)


def nullspace(layout: Layout) -> Layout:
    """Compute the nullspace (kernel) of a layout.

    The nullspace contains all coordinates that map to offset 0. These are
    the stride-0 modes: dimensions along which movement in the logical domain
    produces no movement in memory (broadcast dimensions).

    The result is a layout whose domain enumerates all elements that map to 0:
        layout(nullspace(layout)(i)) == 0  for all i in range(size(result))

    The size of the nullspace is  size(layout) / size(filter(layout)),
    i.e., the total domain divided by the "effective" (non-broadcast) domain.

    Algorithm: flatten the layout, compute column-major strides for the full
    flat shape, then select the shapes and strides at stride-0 positions.
    The column-major strides ensure that nullspace coordinates, when mapped
    back through the layout via idx2crd, land on the broadcast dimensions.

    Examples:
        nullspace(Layout((2,2,2), (0,0,0))) -> (2,2,2):(1,2,4)
        nullspace(Layout((2,2,2), (1,0,2))) -> 2:2
        nullspace(Layout((4,8), (1,4)))      -> 1:0
    """
    flat = flatten(layout)

    # Column-major strides for the full flat shape: these are the strides
    # of a compact column-major layout with the same shape.
    col_major_strides = prefix_product(flat.shape)

    # Normalize to tuples so zip works on scalar (rank-0) layouts
    flat_shapes = as_tuple(flat.shape)
    flat_strides = as_tuple(flat.stride)
    col_strides = as_tuple(col_major_strides)

    # Select shapes and strides at stride-0 positions
    zero_shapes = []
    zero_strides = []
    for s, d, r in zip(flat_shapes, flat_strides, col_strides):
        if d == 0 and s != 1:
            zero_shapes.append(s)
            zero_strides.append(r)

    if not zero_shapes:
        return Layout(1, 0)

    return Layout(as_shape(zero_shapes), as_shape(zero_strides))


def max_common_layout(layout_a: Layout, layout_b: Layout) -> Layout:
    """Return a layout pointing to the maximum contiguous elements common to both.

    Two layouts "logically correspond" when indexing through one produces the
    same offsets as indexing through the other. max_common_layout finds the
    longest contiguous prefix where a(R(i)) == i and b(R(i)) == i.

    Algorithm: compose(a, right_inverse(b)), coalesce, then check if the
    leading mode has stride 1. If so, compose inv_b with that leading mode
    to get the common layout. Otherwise, return Layout(1, 0).

    Args:
        layout_a: First layout
        layout_b: Second layout

    Returns:
        A layout R such that a(R(i)) == i and b(R(i)) == i for all i < size(R)

    Examples:
        max_common_layout(Layout(8, 1), Layout(8, 1))       -> 8:1
        max_common_layout(Layout((4,2), (2,1)), Layout(8,1)) -> 1:0
        max_common_layout(Layout(8, 1), Layout((4,2), (1,4))) -> 4:1
    """
    inv_b = right_inverse(layout_b)
    common = coalesce(compose(layout_a, inv_b))

    # Check if the leading mode has stride 1
    flat_common = flatten(common)
    flat_shape = flat_common.shape
    flat_stride = flat_common.stride

    # Handle scalar layouts (rank 0) - they are effectively rank 1
    if is_int(flat_shape):
        if flat_stride == 1:
            return coalesce(compose(inv_b, Layout(flat_shape, 1)))
        else:
            return Layout(1, 0)

    if rank(flat_common) > 0 and flat_stride[0] == 1:
        leading_shape = flat_shape[0]
        return coalesce(compose(inv_b, Layout(leading_shape, 1)))
    else:
        return Layout(1, 0)


def max_common_vector(layout_a: Layout, layout_b: Layout) -> int:
    """Return the number of contiguous elements that logically correspond in both layouts.

    This is the size of max_common_layout(a, b) — the length of the longest
    contiguous prefix where both layouts agree.

    Args:
        layout_a: First layout
        layout_b: Second layout

    Returns:
        An integer N >= 1 such that for all 0 <= i < N, both layouts map
        element i to offset i.

    Examples:
        max_common_vector(Layout(8, 1), Layout(8, 1))        -> 8
        max_common_vector(Layout((4,2), (2,1)), Layout(8,1)) -> 1
        max_common_vector(Layout(8, 1), Layout((4,2), (1,4))) -> 4
    """
    return size(max_common_layout(layout_a, layout_b))


def slice_and_offset(crd, layout: Layout):
    """Slice a layout by a coordinate and return (sublayout, offset).

    Given a coordinate with None values marking sliced (free) dimensions
    and integer values marking fixed dimensions, returns:
    - sublayout: Layout over the free dimensions
    - offset: The linear offset from the fixed dimensions

    Args:
        crd: Coordinate tuple with None for sliced dims and ints for fixed dims
        layout: The layout to slice

    Returns:
        (sublayout, offset) tuple

    Examples:
        slice_and_offset((None, 3), Layout((4, 8), (1, 4)))
        -> (Layout((4,), (1,)), 12)  # sublayout over dim 0, offset = 3*4
    """
    sliced_shape = slice_modes(crd, layout.shape)
    sliced_stride = slice_modes(crd, layout.stride)
    # slice_modes returns tuples that preserve structure; pass directly to Layout
    sublayout = Layout(
        sliced_shape if sliced_shape else (),
        sliced_stride if sliced_stride else (),
    )
    offset = crd2offset(crd, layout.shape, layout.stride)
    return (sublayout, offset)


# =============================================================================
# Coordinate conversion: idx2crd, crd2flat, crd2offset, crd2idx, crd2crd
# =============================================================================
#
# These convert between the three coordinate representations in CuTe:
#   1D index   -- a single integer (the "flat" position in the domain)
#   nD coord   -- a tuple of per-mode coordinates, e.g. (row, col)
#   offset     -- the memory offset (what the layout computes)
#
# idx2crd:    1D index -> nD coordinate (decompose via shape)
# crd2flat:   nD coordinate -> 1D index (flatten via shape, inverse of idx2crd)
# crd2offset: nD coordinate -> offset (inner product with stride)
# crd2idx:    dispatches to crd2flat (2-arg) or crd2offset (3-arg), matching
#             C++ CuTe's overloaded crd2idx(coord, shape[, stride])
# crd2crd:    convert between two shapes' coordinate spaces
#

def idx2crd(coord: Any, shape: Any) -> Any:
    """Convert index into a hierarchical coordinate."""

    if isinstance(shape, int):
        return coord

    # Case: Input is a single integer index for this entire sub-hierarchy
    if isinstance(coord, int):
        res = []
        index = coord
        for s in shape:
            m_size = size(s)
            # Recurse: expand the index restricted to this mode's sub-shape
            res.append(idx2crd(index % m_size, s))
            index //= m_size
        return tuple(res)

    # Case: Input is a collection (Tuple/tuple)
    # We map the modes of the coordinate to the modes of the shape
    if is_tuple(coord):
        if len(coord) != len(shape):
            raise ValueError(
                f"Coordinate rank {len(coord)} mismatch with Shape rank {len(shape)}"
            )

        return zip_transform(coord, shape, idx2crd)

    raise TypeError(f"Cannot map {type(coord)} to shape {shape}")


def crd2flat(coord: Any, shape: Any = None) -> int:
    """Convert a hierarchical coordinate to a flat 1D index (inverse of idx2crd).

    Example: crd2flat((1, 1), (4, 4)) -> 5
    """

    if isinstance(shape, int):
        if is_tuple(coord):
            raise ValueError(f"Cannot map coordinate {coord} to scalar shape {shape}")
        return int(coord)

    if isinstance(coord, int):
        return coord

    if is_tuple(coord):
        if len(coord) != len(shape):
            raise ValueError(f"Rank mismatch: coord {len(coord)} vs shape {len(shape)}")

        index = 0
        stride = 1
        for c, s in zip(coord, shape):
            tindex = crd2flat(c, s)
            index += tindex * stride
            stride *= size(s)
        return index

    raise TypeError(f"Unsupported coordinate type: {type(coord)}")


# crd2offset((1, 1), Layout((4,4),(1,100))) -> 101
def crd2offset(coord, shape, stride) -> int:
    """Convert coordinate to memory offset (inner product with stride).

    When coord is a 1D integer index and shape is a tuple, the index is
    decomposed across modes from left to right. Each mode (except the last)
    consumes its share via modular arithmetic. The last mode is NOT modded,
    allowing indices beyond the domain to extend through it. This matches
    CuTe's convention that the last mode is implicitly extensible.
    """
    # Case 0: None coordinate contributes 0 offset (used by slice operations)
    if coord is None:
        return 0

    # Case 1: Scalar shape - direct multiplication
    if is_int(shape):
        if is_tuple(coord):
            raise ValueError(f"Cannot map coordinate {coord} to scalar shape {shape}")
        return coord * stride

    # Case 2: 1D index mapping (index -> nD -> offset)
    if isinstance(coord, int):
        offset = 0
        index = coord
        shape_list = list(shape)
        stride_list = list(stride)
        for i, (s, d) in enumerate(zip(shape_list, stride_list)):
            mode_size = size(s)
            if i < len(shape_list) - 1:
                # All modes except last: mod by mode size
                c = index % mode_size
                index //= mode_size
            else:
                # Last mode: do not mod — extend infinitely
                c = index

            # If s is a Tuple, d is also a Tuple. We must recurse.
            if is_tuple(s):
                offset += crd2offset(c, s, d)
            else:
                offset += c * d
        return offset

    # Case 3: nD coordinate mapping (coord tuple -> offset)
    offset = 0
    for c, s, d in zip(coord, shape, stride):
        if c is None:
            continue  # None coordinates contribute 0 (slice marker)
        if is_tuple(s):
            # If the shape element is nested, the coordinate part c
            # must also be nested (or be an int that we treat as a 1D index)
            offset += crd2offset(c, s, d)
        else:
            offset += c * d
    return offset


def crd2idx(coord, shape, stride=None):
    """Dispatch to crd2flat (2-arg) or crd2offset (3-arg).

    Matches C++ CuTe's overloaded crd2idx(coord, shape[, stride]):
      crd2idx(coord, shape)         -> 1D flat index (colexicographic)
      crd2idx(coord, shape, stride) -> memory offset (inner product with stride)
    """
    if stride is None:
        return crd2flat(coord, shape)
    return crd2offset(coord, shape, stride)


def crd2crd(crd: Any, dst_shape: Any, src_shape: Any = None) -> Any:
    """Transform a coordinate into a different shape's iteration space.

    If crd is a tuple and dst_shape is a tuple, recursively transform each mode.
    If crd is a tuple and dst_shape is an int, flatten the coordinate using src_shape.
    If crd is an int and dst_shape is a tuple, expand the index into dst_shape.
    If both are ints, return crd (identity).

    Args:
        crd: The coordinate to transform
        dst_shape: The target shape
        src_shape: The source shape (required when crd is tuple and dst_shape is scalar)

    Examples:
        crd2crd(3, (2, 4)) -> (1, 0)        # expand index 3 into (2,4)
        crd2crd((1, 0), 8, (2, 4)) -> 1     # flatten (1,0) from (2,4) space
        crd2crd((1, 2), (3, 4)) -> (1, 2)   # identity transform
    """
    if is_tuple(crd):
        if is_tuple(dst_shape):
            if len(crd) != len(dst_shape):
                raise ValueError(f"Rank mismatch: crd has {len(crd)} elements, dst_shape has {len(dst_shape)}")
            return zip_transform(crd, dst_shape, crd2crd)
        else:
            # crd is tuple, dst_shape is scalar: flatten using src_shape
            if src_shape is None:
                raise ValueError("src_shape required to flatten tuple coordinate to scalar")
            return crd2flat(crd, src_shape)
    else:
        if is_tuple(dst_shape):
            return idx2crd(crd, dst_shape)
        else:
            return crd


def slice_modes(crd, trg):
    """Filter trg according to crd: keep only elements paired with None.

    This implements CuTe's slice operator. Elements of trg that are paired
    with None in crd are kept (wrapped in a tuple); elements paired with
    concrete integers are dropped.

    Args:
        crd: A coordinate with None values indicating sliced dimensions
        trg: The target (shape or stride) to filter

    Returns:
        A tuple of the kept elements (flattened from nested results)

    Examples:
        slice_modes(None, 4) -> (4,)
        slice_modes(0, 4) -> ()
        slice_modes((None, 0), (3, 4)) -> (3,)
        slice_modes((0, None), (3, 4)) -> (4,)
        slice_modes((None, None), (3, 4)) -> (3, 4)
    """
    if is_tuple(crd):
        if is_tuple(trg):
            if len(crd) != len(trg):
                raise ValueError(f"Rank mismatch: crd has {len(crd)} elements, trg has {len(trg)}")
            # Flatten and concatenate non-empty results
            result = []
            for c, s in zip(crd, trg):
                sub = slice_modes(c, s)
                result.extend(sub)
            return tuple(result)
        else:
            raise TypeError("Cannot slice scalar target with tuple coordinate")
    elif crd is None:
        return (trg,)
    else:
        return ()


def dice_modes(crd, layout):
    """Keep only the modes of a layout that are paired with integers in crd.

    Dice is the complement of slice: slice_modes keeps the None-marked modes
    (the free dimensions), while dice_modes keeps the integer-marked modes
    (the fixed dimensions).

    For layouts: returns a layout over only the "diced" modes.
    For tuples: returns a filtered tuple.

    Note the difference from the C++ entry point: when crd is a plain integer
    (not a tuple), dice_modes returns the target directly (unwrapped), matching
    CuTe's convention that dice(int, b) == b.

    Args:
        crd: A coordinate with None for modes to drop, integers for modes to keep
        layout: The Layout (or tuple) to filter

    Returns:
        A Layout (or value) over only the integer-marked modes

    Examples:
        dice_modes(0, Layout((3,4), (1,4)))       -> (3,4):(1,4)   # scalar crd: identity
        dice_modes((0, None), Layout((3,4),(1,4))) -> 3:1           # keep mode 0
        dice_modes((None, 0), Layout((3,4),(1,4))) -> 4:4           # keep mode 1
    """
    def dice_tuple(crd, trg):
        """Keep elements of trg paired with integers in crd."""
        if is_tuple(crd):
            if is_tuple(trg):
                if len(crd) != len(trg):
                    raise ValueError(f"Rank mismatch: crd has {len(crd)} elements, trg has {len(trg)}")
                result = []
                for c, s in zip(crd, trg):
                    result.extend(dice_tuple(c, s))
                return tuple(result)
            else:
                raise TypeError("Cannot dice scalar target with tuple coordinate")
        elif crd is None:
            return ()
        else:
            return (trg,)

    if isinstance(layout, Layout):
        if is_tuple(crd):
            diced_shape = dice_tuple(crd, layout.shape)
            diced_stride = dice_tuple(crd, layout.stride)
            return Layout(as_shape(diced_shape), as_shape(diced_stride))
        elif crd is None:
            return Layout()
        else:
            return layout
    else:
        # Tuple-level dice
        if is_tuple(crd):
            return dice_tuple(crd, layout)
        elif crd is None:
            return ()
        else:
            return layout


# =============================================================================
# Tile and composition
# =============================================================================
#
# Composition is function composition: compose(A, B) produces a layout C where
# C(i) = A(B(i)). B selects which elements of A to visit, and in what order.
# This is the fundamental operation --- division, product, and tiling are all
# defined in terms of composition.
#
# A Tile is a tuple-of-Layouts used for mode-by-mode composition. When you
# compose a multi-mode layout with a Tile, each Tile element is composed with
# the corresponding mode independently.
#
# Shape arithmetic (shape_div, shape_mod) is the machinery that makes
# composition work on hierarchical shapes: it propagates a divisor through
# nested shape elements, consuming from the innermost (leftmost) modes first.
#

class Tile(tuple):
    """A Tiler is a tuple-of-Layouts used for mode-by-mode composition.

    Tile is semantically distinct from a plain tuple: it signals mode-by-mode
    composition rather than bundling.  When you compose(L, Tile(A, B)), each
    mode of L is composed independently:

        compose(a, tiler) = Layout(compose(mode(a, 0), tiler[0]),
                                   compose(mode(a, 1), tiler[1]), ...)

    This is different from compose(L, Layout((s0, s1), (d0, d1))) where the
    Layout is treated as a single mapping.  Tile makes the intent explicit:
    "apply these tilers to L's modes, one-by-one."

    Examples:
        # (12,(4,8)):(59,(13,1))
        a = Layout((12, (4, 8)), (59, (13, 1)))

        # <3:4, 8:2>
        tiler = Tile(Layout(3, 4), Layout(8, 2))

        # (3,(2,4)):(236,(26,1))
        result = compose(a, tiler)
    """

    def __new__(cls, *layouts):
        """Create a Tile from one or more Layouts.

        Args:
            *layouts: Layout objects to include in the tile
        """
        for i, layout in enumerate(layouts):
            if not isinstance(layout, Layout):
                raise TypeError(
                    f"Tile element {i} must be a Layout, got {type(layout).__name__}"
                )
        return super().__new__(cls, layouts)

    def __repr__(self):
        contents = ", ".join(repr(layout) for layout in self)
        return f"Tile({contents})"


def safe_div(a: int, b: int) -> int:
    """Integer division where b must divide a evenly.

    In CuTe, this is used when we know the division is exact.
    Returns a // b, asserting that b divides a.
    """
    if b == 0:
        raise ValueError("Division by zero")
    if a % b != 0:
        raise ValueError(f"safe_div requires {b} to divide {a} evenly")
    return a // b


def shape_div(shape: Any, divisor: int) -> Any:
    """Divide a shape by a divisor, consuming from the innermost modes first.

    Intuition: shape_div and shape_mod together factor a shape into two
    pieces — the part consumed by the divisor (shape_mod) and the part
    that remains (shape_div). They are the hierarchical analog of
    integer divmod, respecting CuTe's column-major (leftmost-fastest)
    convention: divisors consume from the innermost (leftmost) modes
    first, then carry to outer modes.

    shape_div is to hierarchical shapes what integer division is to integers.
    It divides the shape element-by-element from left to right (innermost first
    in CuTe's column-major convention). When the divisor exceeds a mode's size,
    that mode becomes 1 and the remaining divisor carries to the next mode.

    For scalars: shape_div(a, b) = ceil(a / b), which equals a/b when b|a
    and 1 when a|b (i.e., a/gcd(a,b) when one divides the other).

    The key identity: size(shape_div(s, d)) * size(shape_mod(s, d)) == size(s)

    Examples:
        shape_div(12, 4) -> 3           # 12/4 = 3
        shape_div(12, 3) -> 4           # 12/3 = 4
        shape_div((4, 3), 2) -> (2, 3)  # Divide first mode: 4/2=2, rest untouched
        shape_div((4, 3), 4) -> (1, 3)  # First mode consumed: 4/4=1
        shape_div((4, 6), 8) -> (1, 3)  # Carries into second mode: 8/4=2, 6/2=3
        shape_div((4, 3), 12) -> (1, 1) # All consumed
    """
    if divisor == 1:
        return shape

    def _scalar(s, d):
        if s % d != 0 and d % s != 0:
            raise ValueError(
                f"shape_div({s}, {d}): one must divide the other for clean "
                f"factorization"
            )
        return (s + d - 1) // d

    def _update(first, divisor):
        return shape_div(divisor, size(first))

    return fold_accumulate(shape, divisor, _scalar, _update)


def shape_mod(shape: Any, modulus: int) -> Any:
    """The complement of shape_div: returns the "kept" portion of a shape.

    If shape_div tells you what's left after dividing, shape_mod tells you
    what was consumed. The key identity:
        size(shape_div(s, d)) * size(shape_mod(s, d)) == size(s)

    For scalars: shape_mod(a, m) = min(a, m) when one divides the other (= gcd(a, m)).

    Examples:
        shape_mod(12, 4) -> 4           # gcd(12, 4) = 4
        shape_mod((4, 3), 2) -> (2, 1)  # 2 consumed from first mode, nothing from second
        shape_mod((4, 3), 12) -> (4, 3) # All kept (modulus >= size)
    """
    def _scalar(s, m):
        return s if m >= s else math.gcd(s, m)

    def _update(first, modulus):
        return shape_div(modulus, shape_mod(size(first), modulus))

    return fold_accumulate(shape, modulus, _scalar, _update)


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division: smallest integer >= a/b."""
    return (a + b - 1) // b


def upcast(layout: "Layout", n: int) -> "Layout":
    """Reinterpret a layout from a finer to a coarser coordinate space.

    Mirrors CuTe's upcast<N>(layout).  Use case: GPU memory layouts are
    often defined in bits (to handle mixed-precision types uniformly),
    but you want to work with elements (fp16, int8, etc.).
    upcast(L, 16) converts a bit-addressed layout to an fp16-element layout.

    For the stride-1 mode the shape shrinks by n (the elements are now n×
    bigger, so there are fewer of them).  All strides are divided by n.

    Examples:
        # Bit layout → fp16 elements (÷16)
        upcast(Layout((32, 32), (32, 1)), 16)
        # => Layout((32, 2), (2, 1))

        # Hierarchical value mode
        upcast(Layout((32, (32, 4)), (32, (1, 1024))), 16)
        # => Layout((32, (2, 4)), (2, (1, 64)))

        # Transpose layout with sub-element innermost stride
        upcast(Layout(((4, 8), (16, 2)), ((256, 16), (1, 128))), 16)
        # => Layout(((4, 8), (1, 2)), ((16, 1), (1, 8)))
    """
    if n == 1:
        return layout

    def _upcast_leaf(s, d):
        if d == 0:
            return (s, d)
        shape_divisor = _ceil_div(n, abs(d))
        new_shape = _ceil_div(s, shape_divisor)
        new_stride = (1 if d > 0 else -1) * _ceil_div(abs(d), n)
        return (new_shape, new_stride)

    def _apply(shape, stride):
        if is_tuple(shape):
            if not is_tuple(stride) or len(shape) != len(stride):
                raise ValueError(f"Shape/stride structure mismatch: {shape} vs {stride}")
            pairs = [_apply(s, d) for s, d in zip(shape, stride)]
            new_s = tuple(p[0] for p in pairs)
            new_d = tuple(p[1] for p in pairs)
            return (new_s, new_d)
        return _upcast_leaf(shape, stride)

    new_shape, new_stride = _apply(layout.shape, layout.stride)
    return Layout(new_shape, new_stride)


def downcast(layout: "Layout", n: int) -> "Layout":
    """Reinterpret a layout from a coarser to a finer coordinate space.

    Mirrors CuTe's downcast<N>(layout).  The inverse of upcast: converts
    element coordinates back to bit coordinates.  For the stride-1 mode
    the shape grows by n, and all other strides are multiplied by n.

    Examples:
        # Element layout → bit coordinates (×16)
        downcast(Layout((32, 2), (2, 1)), 16)
        # => Layout((32, 32), (32, 1))
    """
    if n == 1:
        return layout

    def _downcast_leaf(s, d):
        if abs(d) == 1:
            return (s * n, d)
        return (s, d * n)

    def _apply(shape, stride):
        if is_tuple(shape):
            if not is_tuple(stride) or len(shape) != len(stride):
                raise ValueError(f"Shape/stride structure mismatch: {shape} vs {stride}")
            pairs = [_apply(s, d) for s, d in zip(shape, stride)]
            new_s = tuple(p[0] for p in pairs)
            new_d = tuple(p[1] for p in pairs)
            return (new_s, new_d)
        return _downcast_leaf(shape, stride)

    new_shape, new_stride = _apply(layout.shape, layout.stride)
    return Layout(new_shape, new_stride)


def _composition_1d(layout_a: "Layout", b_shape: int, b_stride: int) -> "Layout":
    """Compose layout A with a 1D layout (scalar shape and stride).

    This is the core composition algorithm. It answers: "if B selects
    b_shape elements from A with stride b_stride, what layout results?"

    Algorithm:
      1. Coalesce A to merge contiguous modes into a flat list.
      2. Fold over A's modes (except the last):
         - Compute how many of B's elements fit in this mode of A.
         - Emit a result mode with shape = elements consumed,
           stride = b_stride * a_mode_stride.
         - Carry remaining B shape/stride to the next mode.
      3. Last mode absorbs all remaining shape (CuTe's extensible-
         last-mode convention: the outermost mode is implicitly
         infinite).

    Example:
        A = Layout((4, 8), (1, 8))  # two modes with a stride gap
        B has shape=8, stride=1     # select 8 contiguous elements
        Result: Layout((4, 2), (1, 8))  # 4 from first mode, 2 from second
    """
    if b_stride == 0:
        return Layout(b_shape, 0)

    flat_a = coalesce(layout_a)
    flat_shapes = [flat_a.shape] if is_int(flat_a.shape) else list(flat_a.shape)
    flat_strides = [flat_a.stride] if is_int(flat_a.stride) else list(flat_a.stride)

    result_shape = []
    result_stride = []
    remaining_shape = b_shape
    remaining_stride = b_stride

    # Process all modes except the last
    for curr_shape, curr_stride in zip(flat_shapes[:-1], flat_strides[:-1]):
        if curr_shape % remaining_stride != 0 and remaining_stride % curr_shape != 0:
            raise ValueError(
                f"complement: shape {curr_shape} and stride {remaining_stride} "
                f"are not divisible"
            )
        new_shape = min(max(1, curr_shape // remaining_stride), remaining_shape)
        if new_shape != 1:
            result_shape.append(new_shape)
            result_stride.append(remaining_stride * curr_stride)
        remaining_shape = remaining_shape // new_shape
        remaining_stride = -(-remaining_stride // curr_shape)  # ceil division

    # Last mode absorbs all remaining shape
    if remaining_shape != 1 or not result_shape:
        result_shape.append(remaining_shape)
        result_stride.append(remaining_stride * flat_strides[-1])

    return Layout(as_shape(result_shape), as_shape(result_stride))


def _compose_layouts(layout_a: Layout, layout_b: Layout) -> Layout:
    """Compose two Layout objects."""
    if is_tuple(layout_a.shape) and len(layout_a.shape) == 0:
        return Layout()
    if is_tuple(layout_b.shape) and len(layout_b.shape) == 0:
        return Layout()

    def compose_element(b_shape, b_stride):
        """Recursively compose A with one element of B's shape/stride."""
        if is_tuple(b_shape):
            results = [compose_element(b_shape[i], b_stride[i])
                       for i in range(len(b_shape))]
            return Layout(tuple(r.shape for r in results),
                          tuple(r.stride for r in results))
        return _composition_1d(layout_a, b_shape, b_stride)

    if is_tuple(layout_b.shape):
        results = [compose_element(layout_b.shape[i], layout_b.stride[i])
                   for i in range(len(layout_b.shape))]
        return Layout(tuple(r.shape for r in results),
                      tuple(r.stride for r in results))

    return _composition_1d(layout_a, layout_b.shape, layout_b.stride)


def _compose_with_tiler(layout_a: Layout, tiler) -> Layout:
    """Compose a layout mode-by-mode with a tiler (Tile or tuple)."""
    result_shapes = []
    result_strides = []

    for i, elem in enumerate(tiler):
        mode_layout = mode(layout_a, i)
        composed = compose(mode_layout, elem)
        result_shapes.append(unwrap(composed.shape))
        result_strides.append(unwrap(composed.stride))

    # Append remaining modes unchanged
    for i in range(len(tiler), rank(layout_a)):
        mode_layout = mode(layout_a, i)
        result_shapes.append(unwrap(mode_layout.shape))
        result_strides.append(unwrap(mode_layout.stride))

    return Layout(tuple(result_shapes), tuple(result_strides))


def compose(layout_a: Any, layout_b: Any) -> Any:
    """The fundamental operation of CuTe layout algebra: function composition.

    compose(A, B) produces a layout C where C(i) = A(B(i)). B selects which
    elements of A to visit, and in what order.
    The resulting layout has B's shape, and maps indices through B then A:
        compose(A, B)(i) = A(B(i))

    When A is a Swizzle, the result is a Layout with an embedded swizzle that
    applies the underlying layout B first, then applies the swizzle function:
        compose(Swizzle, Layout)(i) = Swizzle(Layout(i))

    This is the fundamental composition operation in CuTe that allows
    building complex memory access patterns from simpler ones.

    A Tiler (the second argument) can be one of:
    1. A Layout - composition between two functions from integers to integers
    2. A tuple of Tilers - mode-by-mode composition until case (1) is found
    3. A Shape (tuple of ints) - interpreted as tuple of Layout(n, 1)

    When B is a tuple of Tilers, composition is done mode-by-mode:
        compose(A, (B0, B1, ...)) = Layout(compose(mode(A,0), B0),
                                           compose(mode(A,1), B1), ...)

    This recursive definition allows:
    - By-mode tiling: "Give me the 3x5x8 subblock of this MxNxL tensor"
    - 1-D reshaping: "Reorder this 8x16 block using this element order"

    Args:
        layout_a: The outer layout (the one being indexed into)
        layout_b: A Tiler - Layout, tuple of Tilers, or Shape

    Returns:
        A Layout (possibly with nested shape/stride, possibly with embedded swizzle)

    Examples:
        compose(Layout(8, 2), Layout(4, 1)) -> Layout(4, 2)
        compose(Layout((6,2), (8,2)), Layout((4,3), (3,1))) -> Layout(((2,2),3), ((24,2),8))

        # Mode-by-mode with explicit Tile:
        a = Layout((12, (4, 8)), (59, (13, 1)))
        tiler = Tile(Layout(3, 4), Layout(8, 2))
        compose(a, tiler) -> Layout((3, (2, 4)), (236, (26, 1)))

        # Shape as tiler (interpreted as tuple-of-layouts with stride 1):
        a = Layout((12, (4, 8)), (59, (13, 1)))
        tiler = (3, 8)  # Equivalent to Tile(Layout(3,1), Layout(8,1))
        compose(a, tiler) -> Layout((3, (4, 2)), (59, (13, 1)))

        # Mixed tuple of tilers:
        tiler = (Layout(3, 4), 8)  # Layout for mode 0, Shape for mode 1
        compose(a, tiler) -> Layout((3, (4, 2)), (236, (13, 1)))

        # Swizzle composition (returns Layout with embedded swizzle):
        compose(Swizzle(3, 0, 3), Layout((8, 8), (8, 1))) -> Layout with swizzle
    """
    # Swizzle composition
    if isinstance(layout_a, Swizzle):
        if not isinstance(layout_b, Layout):
            raise TypeError(
                f"When composing with Swizzle, second argument must be Layout, got {type(layout_b).__name__}"
            )
        return Layout(layout_b.shape, layout_b.stride, swizzle=layout_a)

    # Layout-with-Layout composition
    if isinstance(layout_b, Layout):
        if layout_b._swizzle is not None:
            # CuTe C++: compose(A, Swizzle o B) = NewSwizzle o compose(A, B)
            # where NewSwizzle = make_swizzle(A(yyy_msk), A(zzz_msk))
            # See layout_composed.hpp:379 and swizzle_layout.hpp:327
            swz = layout_b._swizzle
            active_Y = layout_a(swz.yyy_msk)
            active_Z = layout_a(swz.zzz_msk)
            new_swizzle = make_swizzle(active_Y, active_Z)
            inner_b = Layout(layout_b.shape, layout_b.stride)  # strip swizzle
            composed = _compose_layouts(layout_a, inner_b)
            return Layout(composed.shape, composed.stride, swizzle=new_swizzle)
        return _compose_layouts(layout_a, layout_b)

    # Tiler composition (Tile or tuple)
    if isinstance(layout_b, Tile):
        if len(layout_b) > rank(layout_a):
            raise ValueError(
                f"Tiler has {len(layout_b)} elements but layout has only {rank(layout_a)} modes"
            )
        return _compose_with_tiler(layout_a, layout_b)

    # Tuple tiler - convert elements and recurse
    if is_tuple(layout_b):
        def to_layout(elem):
            if isinstance(elem, Layout):
                return elem
            if isinstance(elem, int):
                return Layout(elem, 1)
            if is_tuple(elem):
                if is_pure_shape(elem):
                    return Layout(elem, 1)
                return elem  # Mixed tuple, keep for recursive processing
            raise TypeError(f"Invalid tiler element: {type(elem)}")

        tiler = [to_layout(e) for e in layout_b]
        if all(isinstance(e, Layout) for e in tiler):
            tiler = Tile(*tiler)

        if len(tiler) > rank(layout_a):
            raise ValueError(
                f"Tiler has {len(tiler)} elements but layout has only {rank(layout_a)} modes"
            )
        return _compose_with_tiler(layout_a, tiler)

    raise TypeError(f"Invalid tiler type: {type(layout_b)}")


# =============================================================================
# Division operations
# =============================================================================
#
# Division factors a layout into (tile, rest):
#   logical_divide(A, B) = compose(A, Layout(B, complement(B, size(A))))
#
# The tile part tells you where within a tile, the rest part tells you
# which tile. Division answers: "how do I iterate in tiles of size T?"
#
# The zipped/tiled/flat variants control how the result modes are organized:
#   zipped_divide  -> ((tiles), (rests))          -- two modes
#   tiled_divide   -> ((tiles), rest0, rest1, ..) -- tiles grouped, rests flat
#   flat_divide    -> (tile0, tile1, rest0, ..)   -- everything flat
#


def logical_divide(layout: Layout, tiler: Any) -> Layout:
    """Divide a layout into (tile, rest) --- the core tiling operation.

    Division answers: "if I want to process this layout in tiles of size T,
    how do I organize the iteration?" The result has two parts:
    - Tile: coordinates *within* a tile (the inner loop)
    - Rest: coordinates *across* tiles (the outer loop)

    Formally: logical_divide(A, B) = compose(A, Layout(B, complement(B, size(A))))

    Intuition: to tile A by B, we need two coordinates:
    - "which element within a tile?" -> B (the tiler itself)
    - "which tile?" -> complement(B) (fills the gaps between tiles)
    Layout(B, complement(B)) bundles these into (within-tile, across-tiles).
    Composing with A maps this coordinate space through A's pattern.

    The result has the structure: (Tile, Rest)

    For multi-mode tilers (tuples), each mode is divided independently:
        ((TileM, RestM), (TileN, RestN), L, ...)

    The tiler can be:
    - An integer: simple 1D tile size (uses mode-by-mode division)
    - A tuple/Tuple of integers: tile sizes for each mode (mode-by-mode division)
    - A Layout: uses the CuTe formula with composition

    Args:
        layout: The layout to tile
        tiler: The tile specification (int, tuple, or Layout)

    Returns:
        A Layout with hierarchical (tile, rest) structure

    Examples:
        logical_divide(Layout(16), 4) -> Layout((4, 4), (1, 4))
        logical_divide(Layout((4,2,3), (2,1,8)), Layout(4, 2)) -> ((2,2),(2,3)):((4,1),(2,8))
    """
    if isinstance(tiler, Layout):
        # Layout tiler: use CuTe formula
        # logical_divide(A, B) = compose(A, Layout(B, complement(B, size(A))))
        layout_size = size(layout)
        tiler_complement = complement(tiler, layout_size)

        # Create bundled layout: (tiler, complement)
        combined = Layout(tiler, tiler_complement)

        # Compose layout with the combined tiler
        result = compose(layout, combined)
        return result
    elif isinstance(tiler, int):
        # Integer tiler: mode-by-mode division of first mode
        # If the tile doesn't evenly divide the first mode, use Layout path
        if is_int(layout.shape):
            # Scalar layout
            first_mode_size = layout.shape
        else:
            first_mode_size = size(layout.shape[0])
        if first_mode_size % tiler != 0:
            return logical_divide(layout, Layout(tiler, 1))
        return _logical_divide_by_shape(layout, (tiler,))
    elif is_tuple(tiler):
        # Tuple tiler: mode-by-mode division
        return _logical_divide_by_shape(layout, tiler)
    else:
        raise TypeError(f"Tiler must be int, tuple, or Layout, got {type(tiler)}")


def _logical_divide_by_shape(layout: Layout, tiler_shape: Any) -> Layout:
    """Divide a layout mode-by-mode using a shape tuple.

    This is used when the tiler is a simple shape tuple (not a Layout).
    Each mode of the layout is divided by the corresponding tiler element.

    Result structure: ((TileM, RestM), (TileN, RestN), L, ...)
    """
    tiler_sizes = [tiler_shape] if isinstance(tiler_shape, int) else list(tiler_shape)

    # Convert scalar shapes to tuple for uniform processing
    layout_shapes = as_tuple(layout.shape)
    layout_strides = as_tuple(layout.stride)

    result_shapes = []
    result_strides = []

    for i, (s, d) in enumerate(zip(layout_shapes, layout_strides)):
        if i >= len(tiler_sizes):
            result_shapes.append(s)
            result_strides.append(d)
            continue

        tile_size = tiler_sizes[i]
        mode_size = size(s)

        # Hierarchical strides can't be handled by the simple shortcut.
        # Fall through to the formal compose/complement path, matching
        # CuTe C++ which always uses that path (layout.hpp:1576).
        if is_tuple(d):
            mode_layout = Layout(s, d)
            divided = logical_divide(mode_layout, Layout(tile_size, 1))
            result_shapes.append(divided.shape)
            result_strides.append(divided.stride)
        elif tile_size == 1:
            result_shapes.append((1, s))
            result_strides.append((d, d))
        elif tile_size <= mode_size:
            rest_size = mode_size // tile_size
            result_shapes.append((tile_size, rest_size))
            result_strides.append((d, elem_scale(d, tile_size)))
        else:
            tile_part = compose(Layout(s, d), Layout(tile_size, 1))
            tile_s = unwrap(tile_part.shape) if is_tuple(tile_part.shape) else tile_part.shape
            tile_d = unwrap(tile_part.stride) if is_tuple(tile_part.stride) else tile_part.stride
            result_shapes.append((tile_s, 1))
            result_strides.append((tile_d, elem_scale(d, mode_size)))

    # Use as_shape to unwrap single-element results back to scalar form
    return Layout(as_shape(result_shapes), as_shape(result_strides))


def _split_divided_modes(layout: Layout, tiler: Any):
    """Divide a layout by a tiler and split the result into tile and rest parts.

    Performs logical_divide, then separates each divided mode into its tile
    portion and its remainder portion. Undivided modes go into the rest lists.

    Returns:
        (tile_shapes, tile_strides, rest_shapes, rest_strides) - four lists
    """
    # Normalize tiler to a shape tuple
    if isinstance(tiler, Layout):
        tiler_shape = tiler.shape
    elif isinstance(tiler, int):
        tiler_shape = (tiler,)
    elif is_tuple(tiler):
        tiler_shape = tiler
    else:
        raise TypeError(f"Tiler must be int, tuple, or Layout, got {type(tiler)}")

    divided = logical_divide(layout, tiler_shape)

    tile_shapes = []
    tile_strides = []
    rest_shapes = []
    rest_strides = []

    num_tiled = len(tiler_shape) if is_tuple(tiler_shape) else 1

    for i, (s, d) in enumerate(zip(divided.shape, divided.stride)):
        if i < num_tiled:
            if is_tuple(s) and len(s) == 2:
                tile_s, rest_s = s
                tile_d, rest_d = d
                tile_shapes.append(tile_s)
                tile_strides.append(tile_d)
                rest_shapes.append(rest_s)
                rest_strides.append(rest_d)
            else:
                tile_shapes.append(s)
                tile_strides.append(d)
        else:
            rest_shapes.append(s)
            rest_strides.append(d)

    return tile_shapes, tile_strides, rest_shapes, rest_strides


def zipped_divide(layout: Layout, tiler: Any) -> Layout:
    """Divide a layout and zip the tile/rest modes together.

    Result structure: ((TileM, TileN), (RestM, RestN, L, ...))
    - Mode 0: all tile shapes zipped together
    - Mode 1: all rest shapes and undivided modes zipped together

    This is useful when you want to iterate over all tile coordinates together,
    then all rest/tile-index coordinates together.

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A rank-2 Layout with ((tiles), (rests)) structure

    Examples:
        zipped_divide(Layout((4,8)), (2,4)) -> Layout(((2,4),(2,2)), ((1,4),(2,16)))
    """
    tile_shapes, tile_strides, rest_shapes, rest_strides = _split_divided_modes(layout, tiler)

    tiles_shape = as_shape(tile_shapes)
    tiles_stride = as_shape(tile_strides)

    if len(rest_shapes) == 0:
        rests_shape = 1
        rests_stride = 0
    else:
        rests_shape = as_shape(rest_shapes)
        rests_stride = as_shape(rest_strides)

    return Layout((tiles_shape, rests_shape), (tiles_stride, rests_stride))


def tiled_divide(layout: Layout, tiler: Any) -> Layout:
    """Divide a layout into tiles and tile indices.

    Result structure: ((TileM, TileN), RestM, RestN, L, ...)
    - Mode 0: all tile shapes grouped together
    - Modes 1+: individual rest shapes and undivided modes (flat)

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A Layout with ((tiles), rest0, rest1, ...) structure

    Examples:
        tiled_divide(Layout((8,8)), (2,2)) -> Layout(((2,2), 4, 4), ...)
    """
    tile_shapes, tile_strides, rest_shapes, rest_strides = _split_divided_modes(layout, tiler)

    tiles_shape = as_shape(tile_shapes)
    tiles_stride = as_shape(tile_strides)

    all_shapes = [tiles_shape] + rest_shapes
    all_strides = [tiles_stride] + rest_strides

    return Layout(tuple(all_shapes), tuple(all_strides))


def flat_divide(layout: Layout, tiler: Any) -> Layout:
    """Divide a layout and flatten all modes.

    Result structure: (TileM, TileN, RestM, RestN, L, ...)
    - All tile shapes come first (flat)
    - Then all rest shapes (flat)
    - Then any undivided modes (flat)

    Args:
        layout: The layout to tile
        tiler: The tile shape (int, tuple, or Layout)

    Returns:
        A flat Layout with (tile0, tile1, ..., rest0, rest1, ..., L, ...) structure

    Examples:
        flat_divide(Layout((8,8)), (2,2)) -> Layout((2, 2, 4, 4), ...)
    """
    tile_shapes, tile_strides, rest_shapes, rest_strides = _split_divided_modes(layout, tiler)

    all_shapes = tile_shapes + rest_shapes
    all_strides = tile_strides + rest_strides

    return Layout(tuple(all_shapes), tuple(all_strides))


# =============================================================================
# Product operations
# =============================================================================
#
# Product reproduces a layout across copies:
#   logical_product(A, B) = Layout(A, compose(complement(A), B))
#
# The result has A's pattern repeated at each position B describes.
# Product answers: "how do I replicate this pattern across B positions?"
#
# The zipped/tiled variants mirror their division counterparts:
#   zipped_product -> ((A-modes), (product-modes))
#   tiled_product  -> ((A-modes), rest0, rest1, ..)
#
# blocked_product interleaves modes: ((A0,B0), (A1,B1), ...) with B's strides
# scaled by cosize(A), placing each copy at a non-overlapping block.
#


def zipped_product(layout_a: Layout, layout_b) -> Layout:
    """Apply logical_product hierarchically and gather split modes into two modes.

    Like zipped_divide but uses logical_product instead of logical_divide.

    Args:
        layout_a: The layout to reproduce
        layout_b: The reproduction specification

    Returns:
        A rank-2 Layout with ((A-modes), (product-modes)) structure
    """
    return hier_unzip(logical_product, layout_a, layout_b)


def tiled_product(layout_a: Layout, layout_b) -> Layout:
    """Apply logical_product hierarchically and flatten the second mode.

    Like tiled_divide but uses logical_product instead of logical_divide.

    Args:
        layout_a: The layout to reproduce
        layout_b: The reproduction specification

    Returns:
        A Layout with ((A-modes), rest0, rest1, ...) structure
    """
    result = zipped_product(layout_a, layout_b)
    second = mode(result, 1)
    all_modes = [mode(result, 0)]
    if is_tuple(second.shape) and not is_scalar(second.shape):
        for i in range(rank(second)):
            all_modes.append(mode(second, i))
    else:
        all_modes.append(second)

    shapes = tuple(m.shape for m in all_modes)
    strides = tuple(m.stride for m in all_modes)
    return Layout(shapes, strides)


def hier_unzip(splitter, layout_a: Layout, layout_b) -> Layout:
    """Apply a splitter hierarchically and gather the split modes into two modes.

    This is the generic helper behind zipped_divide, zipped_product, etc.
    The splitter function (e.g., logical_divide or logical_product) is applied
    recursively through the modes of layout_b. The results are then gathered
    into a rank-2 layout: ((gathered-A-modes), (gathered-rest-modes)).

    Args:
        splitter: A function (layoutA, layoutB) -> rank-2 Layout
        layout_a: The layout to split
        layout_b: The splitting specification (Layout, tuple, int, or None)

    Returns:
        A rank-2 Layout with split modes gathered

    Examples:
        hier_unzip(logical_divide, Layout((4,8)), (2, 4))
        -> ((2,4),(2,2)):((1,4),(2,16))
    """
    if layout_b is None:
        return Layout(
            (1, layout_a.shape),
            (0, layout_a.stride),
        )

    if is_tuple(layout_b) and not isinstance(layout_b, Layout):
        if rank(layout_a) < len(layout_b):
            raise ValueError(
                f"layout_a rank ({rank(layout_a)}) < tiler length ({len(layout_b)})"
            )

        splits = [hier_unzip(splitter, mode(layout_a, i), layout_b[i])
                  for i in range(len(layout_b))]

        first_shapes = [mode(s, 0).shape for s in splits]
        first_strides = [mode(s, 0).stride for s in splits]
        second_shapes = [mode(s, 1).shape for s in splits]
        second_strides = [mode(s, 1).stride for s in splits]

        for i in range(len(layout_b), rank(layout_a)):
            m = mode(layout_a, i)
            second_shapes.append(m.shape)
            second_strides.append(m.stride)

        return Layout(
            (as_shape(first_shapes), as_shape(second_shapes)),
            (as_shape(first_strides), as_shape(second_strides)),
        )

    if isinstance(layout_b, int):
        layout_b = Layout(layout_b)
    return splitter(layout_a, layout_b)


def logical_product(layout_a: Layout, layout_b: Layout) -> Layout:
    """Reproduce layout A's pattern at each position B describes.

    Product is the reverse of division. If division splits A into tiles,
    product replicates A across B copies. The result has A's pattern repeated
    at non-overlapping memory locations determined by B.

    Formally: logical_product(A, B) = Layout(A, compose(complement(A, size(A)*cosize(B)), B))

    For multi-mode tilers (tuples), the operation is applied mode-by-mode.

    Args:
        layout_a: First layout
        layout_b: Second layout (or int or tuple of tilers)

    Returns:
        Layout combining both inputs

    Examples:
        logical_product(Layout(4,1), Layout(3,1)) -> Layout((4,3), (1,4))
    """
    if layout_b is None:
        return layout_a
    if isinstance(layout_b, int):
        return logical_product(layout_a, Layout(layout_b))

    # For tuple tilers, apply mode-by-mode
    if is_tuple(layout_b) and not isinstance(layout_b, Layout):
        if rank(layout_a) < len(layout_b):
            raise ValueError(
                f"layout_a rank ({rank(layout_a)}) < tiler length ({len(layout_b)})"
            )
        result_modes = []
        for i in range(len(layout_b)):
            result_modes.append(logical_product(mode(layout_a, i), layout_b[i]))
        # Append remaining modes unchanged
        for i in range(len(layout_b), rank(layout_a)):
            result_modes.append(mode(layout_a, i))
        shapes = tuple(r.shape for r in result_modes)
        strides = tuple(r.stride for r in result_modes)
        return Layout(shapes, strides)

    # CuTe definition:
    # logical_product(A, B) = Layout(A, compose(complement(A, size(A)*cosize(B)), B))
    comp = complement(layout_a, size(layout_a) * cosize(layout_b))
    composed = compose(comp, layout_b)

    # make_layout(A, composed)
    return Layout(
        (layout_a.shape, composed.shape),
        (layout_a.stride, composed.stride),
    )


def _product_interleave(layout_a: Layout, layout_b: Layout) -> Layout:
    """Interleave modes of two layouts, scaling B's strides by cosize(A).

    For each mode i: shape = (A_shape[i], B_shape[i]),
                      stride = (A_stride[i], B_stride[i] * cosize(A))

    Used by both logical_product (for rank > 1) and blocked_product.
    """
    a_cosize_val = cosize(layout_a)
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)
    max_rank = max(a_rank, b_rank)

    result_shapes = []
    result_strides = []

    for i in range(max_rank):
        if i < a_rank and i < b_rank:
            a_s_val = mode(layout_a.shape, i)
            a_st_val = mode(layout_a.stride, i)
            b_s_val = mode(layout_b.shape, i)
            b_st_val = mode(layout_b.stride, i)
            b_st_scaled = transform_tuple(b_st_val, lambda s: s * a_cosize_val)
            result_shapes.append((a_s_val, b_s_val))
            result_strides.append((a_st_val, b_st_scaled))
        elif i < a_rank:
            a_s_val = mode(layout_a.shape, i)
            a_st_val = mode(layout_a.stride, i)
            result_shapes.append(a_s_val)
            result_strides.append(a_st_val)
        else:
            b_s_val = mode(layout_b.shape, i)
            b_st_val = mode(layout_b.stride, i)
            b_st_scaled = transform_tuple(b_st_val, lambda s: s * a_cosize_val)
            result_shapes.append(b_s_val)
            result_strides.append(b_st_scaled)

    return Layout(tuple(result_shapes), tuple(result_strides))


def blocked_product(layout_a: Layout, layout_b: Layout) -> Layout:
    """Compute a blocked product of two layouts.

    Unlike logical_product which concatenates (A, B) for 1D, blocked_product
    always interleaves corresponding modes: ((A0, B0), (A1, B1), ...).

    A varies fastest (block-first): each block is contiguous, with blocks
    laid out according to B.  Think of A as the "block pattern" and B as
    the "grid of blocks."

    Compare with raked_product: both interleave, but raked has B vary
    fastest (rake-first), while blocked has A vary fastest (block-first).

    For each mode i:
        result_shape[i] = (A_shape[i], B_shape[i])
        result_stride[i] = (A_stride[i], B_stride[i] * cosize(A))

    Args:
        layout_a: First layout (the "inner" or "block" pattern)
        layout_b: Second layout (the "outer" or "tile count" pattern)

    Returns:
        Layout with blocked structure where modes are interleaved

    Examples:
        blocked_product((2,2):(1,2), (2,2):(1,2)) -> ((2,2),(2,2)):((1,4),(2,8))
    """
    a_cosize_val = cosize(layout_a)
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)

    # Handle scalar layouts (rank 0)
    if a_rank == 0 and b_rank == 0:
        # Both scalar: create 2D result
        # Result should be (a_size, b_size) with proper strides
        new_shape = (size(layout_a), size(layout_b))
        new_stride = (layout_a.stride, layout_b.stride * a_cosize_val)
        return Layout(new_shape, new_stride)
    if a_rank == 0:
        # Scalar a with non-scalar b:
        # Pair scalar a with first mode of b, append remaining modes
        b_shapes = list(as_tuple(layout_b.shape))
        b_strides = list(as_tuple(layout_b.stride))

        # First mode: pair (a_size, b[0])
        result_shapes = [(size(layout_a), b_shapes[0])]
        result_strides = [(layout_a.stride, b_strides[0] * a_cosize_val)]

        # Remaining modes: scale strides by a_cosize
        for i in range(1, len(b_shapes)):
            result_shapes.append(b_shapes[i])
            result_strides.append(b_strides[i] * a_cosize_val)

        return Layout(tuple(result_shapes), tuple(result_strides))
    if b_rank == 0:
        # Non-scalar a with scalar b:
        # Pair b with first mode of a, keep remaining modes
        a_shapes = list(as_tuple(layout_a.shape))
        a_strides = list(as_tuple(layout_a.stride))

        # First mode: pair (a[0], b_size)
        result_shapes = [(a_shapes[0], size(layout_b))]
        result_strides = [(a_strides[0], layout_b.stride * a_cosize_val)]

        # Remaining modes: unchanged
        for i in range(1, len(a_shapes)):
            result_shapes.append(a_shapes[i])
            result_strides.append(a_strides[i])

        return Layout(tuple(result_shapes), tuple(result_strides))

    return _product_interleave(layout_a, layout_b)


def _pad_to_rank(layout: Layout, target_rank: int) -> Layout:
    """Pad a layout to a target rank by appending (1, 0) modes.

    Matches C++ CuTe's append<R>(layout) which pads with Layout<_1,_0>{}.
    """
    current_rank = rank(layout)
    if current_rank >= target_rank:
        return layout
    shapes = list(as_tuple(layout.shape))
    strides = list(as_tuple(layout.stride))
    for _ in range(target_rank - current_rank):
        shapes.append(1)
        strides.append(0)
    return Layout(tuple(shapes), tuple(strides))


def _zip_layouts(layout_a: Layout, layout_b: Layout) -> Layout:
    """Zip two layouts mode-by-mode: ((a0,b0), (a1,b1), ...).

    Matches C++ CuTe's zip(layoutA, layoutB) which interleaves corresponding
    modes into paired tuples.

    Both layouts must have the same rank.
    """
    a_rank = rank(layout_a)
    b_rank = rank(layout_b)

    # Handle scalar layouts by treating them as rank-1
    if a_rank == 0 and b_rank == 0:
        # Both scalar: create a single mode with paired shapes/strides
        return Layout((layout_a.shape, layout_b.shape), (layout_a.stride, layout_b.stride))

    if a_rank != b_rank:
        raise ValueError(f"Rank mismatch in zip: {a_rank} vs {b_rank}")
    r = a_rank
    result_shapes = []
    result_strides = []
    for i in range(r):
        a_s = mode(layout_a.shape, i)
        a_d = mode(layout_a.stride, i)
        b_s = mode(layout_b.shape, i)
        b_d = mode(layout_b.stride, i)
        result_shapes.append((a_s, b_s))
        result_strides.append((a_d, b_d))
    return Layout(tuple(result_shapes), tuple(result_strides))


def flat_product(block: Layout, tiler) -> Layout:
    """Compute a flat product: zipped_product with both modes unpacked.

    Like zipped_product, but flattens both the block modes and the product
    modes into a single flat layout: (BLK_0, BLK_1, ..., tiler_0, tiler_1, ...).

    Args:
        block: The block layout to reproduce
        tiler: The reproduction specification

    Returns:
        A flat Layout with all block modes followed by all product modes

    Examples:
        flat_product(Layout((2,4), (1,2)), Layout(3,1))
            -> Layout with shape (2, 4, 3, ...) and appropriate strides
    """
    result = zipped_product(block, tiler)

    # Unpack both modes: result(repeat<R0>(_), repeat<R1>(_))
    # which is equivalent to flattening both the block mode and product mode
    m0 = mode(result, 0)
    m1 = mode(result, 1)

    shapes = []
    strides = []

    # Unpack mode 0 (block modes)
    if is_tuple(m0.shape):
        for i in range(rank(m0)):
            shapes.append(mode(m0.shape, i))
            strides.append(mode(m0.stride, i))
    else:
        shapes.append(m0.shape)
        strides.append(m0.stride)

    # Unpack mode 1 (product modes)
    if is_tuple(m1.shape):
        for i in range(rank(m1)):
            shapes.append(mode(m1.shape, i))
            strides.append(mode(m1.stride, i))
    else:
        shapes.append(m1.shape)
        strides.append(m1.stride)

    return Layout(tuple(shapes), tuple(strides))


def raked_product(block: Layout, tiler: Layout) -> Layout:
    """Compute a raked product: block-interleaved reproduction.

    Like blocked_product, but with the tiler varying fastest within each mode.
    Where blocked_product zips as ((block, tiler), ...), raked_product zips as
    ((tiler, block), ...) — the tiler's elements are interleaved *within* each
    block, rather than the block appearing contiguously.

    This is useful for distributing work across threads where you want each
    thread to access interleaved (raked) elements rather than contiguous blocks.

    Algorithm: pad both layouts to the same rank, compute logical_product,
    then zip with reversed order: zip(product_modes, block_modes).

    Args:
        block: The block layout
        tiler: The tiler layout

    Returns:
        A Layout with interleaved (tiler, block) structure per mode

    Examples:
        raked_product(Layout((2,2), (1,2)), Layout((2,2), (1,2)))
            -> ((2,2),(2,2)):((4,1),(8,2))
        # Compare with blocked_product which gives:
            -> ((2,2),(2,2)):((1,4),(2,8))
    """
    r = max(rank(block), rank(tiler))
    padded_block = _pad_to_rank(block, r)
    padded_tiler = _pad_to_rank(tiler, r)

    result = logical_product(padded_block, padded_tiler)

    # result is rank-2: (block_modes, product_modes)
    # For raked: zip(product_modes, block_modes) — reversed from blocked
    m0 = mode(result, 0)  # block modes
    m1 = mode(result, 1)  # product modes

    return _zip_layouts(m1, m0)


# =============================================================================
# Swizzle
# =============================================================================
#
# Swizzling is used in GPU shared memory to avoid bank conflicts. Shared memory
# is divided into banks (typically 32), and threads that access the same bank
# in the same cycle must serialize. By XORing row bits into column bits,
# adjacent rows access different banks, enabling full memory bandwidth.
#
# A Swizzle is a nonlinear function (it uses XOR, not multiply-add), so it
# cannot be represented as strides. Instead, compose(Swizzle, Layout) produces
# a Layout with an embedded swizzle that applies the layout first, then swizzles.
#


class Swizzle:
    """A nonlinear index transformation that XORs two bit fields to avoid bank conflicts.

    GPU shared memory is divided into banks (typically 32). When multiple threads
    access the same bank simultaneously, they serialize. Swizzling avoids this by
    XORing row bits into column bits so that adjacent rows map to different banks,
    enabling full memory bandwidth.

    Given an index with bit pattern: 0bxxxYYYxxxxZZZxxxx
    - base: number of least-significant bits to keep constant (rightmost xxxx)
    - bits: number of bits in each mask (ZZZ and YYY width)
    - shift: distance between the two bit fields

    The operation replaces ZZZ with (ZZZ XOR YYY), leaving everything else unchanged.

    Args:
        bits: Number of bits in each mask
        base: Number of least-significant bits to keep constant
        shift: Distance between the two masks (positive: YYY is above ZZZ)

    Examples:
        Swizzle(3, 0, 3)  -- XOR bits [0,3) with bits [3,6)
        Swizzle(2, 1, 3)  -- XOR bits [1,3) with bits [4,6)

    Visual example for Swizzle(3, 0, 3):
        Input index:  0b___YYY___ZZZ   (Y=row bits [3,6), Z=col bits [0,3))
        Output index: 0b___YYY___(ZZZ XOR YYY)

        Concrete: index 19 = 0b010_011  (Y=010=2, Z=011=3)
            -> 0b010_(011 XOR 010) = 0b010_001 = 17

        This causes adjacent rows to access different memory banks,
        avoiding shared memory bank conflicts.
    """

    def __init__(self, bits: int, base: int, shift: int):
        self.bits = bits
        self.base = base
        self.shift = shift

    def __repr__(self) -> str:
        return f"Swizzle({self.bits}, {self.base}, {self.shift})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Swizzle):
            return False
        return self.bits == other.bits and self.base == other.base and self.shift == other.shift

    @property
    def yyy_msk(self) -> int:
        """Bit mask for the Y (source) bits of the swizzle."""
        return ((1 << self.bits) - 1) << (self.base + max(0, self.shift))

    @property
    def zzz_msk(self) -> int:
        """Bit mask for the Z (destination) bits of the swizzle."""
        return ((1 << self.bits) - 1) << (self.base + max(0, -self.shift))

    def __call__(self, idx: int) -> int:
        """Apply the swizzle to an index."""
        # Create mask for 'bits' number of bits at position 'base'
        mask = ((1 << self.bits) - 1) << self.base

        if self.shift >= 0:
            # Positive shift: XOR higher bits into lower bits
            # Extract bits from [base+shift, base+shift+bits), XOR into [base, base+bits)
            return idx ^ ((idx >> self.shift) & mask)
        else:
            # Negative shift: XOR lower bits into higher bits
            # Extract bits from [base, base+bits), shift left, XOR into higher position
            return idx ^ ((idx & mask) << (-self.shift))


def make_swizzle(Y: int, Z: int):
    """Create a Swizzle from Y and Z bit positions.

    Given bit masks Y and Z indicating which bits interact, construct
    the Swizzle(bits, base, shift) that performs the corresponding XOR.

    Matches CuTe C++ make_swizzle<Y,Z>() in swizzle.hpp.

    Args:
        Y: Bit mask for the Y (source) bits
        Z: Bit mask for the Z (destination) bits

    Returns:
        A Swizzle, or None if both masks are zero (identity).
    """
    num_bits = bin(Y).count("1")
    if num_bits != bin(Z).count("1"):
        raise ValueError(
            f"make_swizzle: bit count mismatch: popcount({Y:#b})={num_bits} "
            f"vs popcount({Z:#b})={bin(Z).count('1')}"
        )
    if num_bits == 0:
        return None  # Identity swizzle
    tz_y = (Y & -Y).bit_length() - 1  # countr_zero(Y)
    tz_z = (Z & -Z).bit_length() - 1  # countr_zero(Z)
    base = min(tz_y, tz_z)
    shift = tz_y - tz_z
    return Swizzle(num_bits, base, shift)
