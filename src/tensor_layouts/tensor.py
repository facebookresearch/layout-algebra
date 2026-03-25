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

# ruff: noqa: F403,F405

"""Tensor class: combines a Layout with a base offset (pointer equivalent).

In CuTe, a Tensor is (Engine/Pointer, Layout). Here we represent the pointer
as an integer offset. The Tensor holds the offset and provides slicing operations
that accumulate offset contributions while reducing the layout.

IMPORTANT: For swizzled layouts, the base offset is a *linear* offset that
gets added to the linear coordinate-to-offset result BEFORE swizzling.
This is critical for correct slicing behavior: slicing accumulates linear
offset, and the swizzle is applied at the final call.
"""

from .layouts import *


class Tensor:
    """A Tensor combines a base offset with a Layout.

    In CuTe C++, a Tensor is (Pointer, Layout) where the pointer holds the
    base address into memory. Here we represent the pointer as an integer
    offset, which is sufficient for the algebraic operations.

    Tensor supports slicing: fixing coordinates accumulates their offset
    contribution and returns a new Tensor with a reduced Layout.

    IMPORTANT: For swizzled layouts, the offset is a *linear* offset.
    The swizzle is applied to (linear_offset + linear_layout_result) at
    the final __call__. This ensures slicing works correctly:

        tensor(i, j) = swizzle(offset + crd2offset((i, j), shape, stride))
        tensor[i, :](j) = swizzle(offset + i*stride_i + j*stride_j)

    NOT: offset + swizzle(crd2offset(...)) which would be incorrect.

    Args:
        layout: The Layout describing the coordinate-to-offset mapping
        offset: The base offset in linear (pre-swizzle) space (default 0)

    Examples:
        t = Tensor(Layout((8, 8), (8, 1)))
        t(3, 5)  -> offset for coordinate (3, 5)
        t[3, :]  -> Tensor with row 3's offset and column layout
    """

    def __init__(self, layout: Layout, offset: int = 0):
        self._layout = layout
        self._offset = offset

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def shape(self):
        return self._layout.shape

    @property
    def stride(self):
        return self._layout.stride

    def __repr__(self) -> str:
        if self._offset:
            return f"Tensor({self._layout}, offset={self._offset})"
        return f"Tensor({self._layout})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return False
        return self._layout == other._layout and self._offset == other._offset

    def __hash__(self):
        return hash((self._layout, self._offset))

    def __call__(self, *args) -> int:
        """Map coordinates to a memory offset.

        For non-swizzled layouts: offset + layout(coords)
        For swizzled layouts: swizzle(offset + linear_layout(coords))

        The key insight is that the swizzle applies to the TOTAL linear
        offset (base + coordinate contribution), not to each separately.
        """
        # Get linear offset from coordinates (without swizzle)
        # Handle single arg (rank-1) vs multiple args
        coords = args[0] if len(args) == 1 else args
        linear_result = crd2offset(coords, self._layout.shape, self._layout.stride)
        total_linear = self._offset + linear_result

        # Apply swizzle to total if present
        if self._layout.swizzle is not None:
            return self._layout.swizzle(total_linear)
        return total_linear

    def __getitem__(self, key):
        """Slice the tensor by fixing some coordinates.

        Fixed coordinates contribute to the offset; free coordinates (:)
        remain in the resulting tensor's layout.

        Examples:
            tensor[3, :]  -- fix mode 0 to 3, keep mode 1
            tensor[:, 5]  -- keep mode 0, fix mode 1 to 5
            tensor[2, 3]  -- fix both modes, returns integer offset
        """
        if isinstance(key, tuple):
            return self._slice_multi(key)
        else:
            return self._slice_single(key, 0)

    def _get_linear_mode_offset(self, mode_idx: int, coord) -> int:
        """Get the linear offset contribution from a mode (without swizzle)."""
        m = mode(self._layout, mode_idx)
        # Compute linear offset using shape and stride directly
        return crd2offset(coord, m.shape, m.stride)

    def _slice_single(self, key, mode_idx: int) -> "Tensor | int":
        """Slice a single mode of the tensor."""
        if isinstance(key, slice) and key == slice(None):
            # Slice with : (all elements) - return tensor for this mode
            mode_layout = mode(self._layout, mode_idx)
            return Tensor(
                Layout(mode_layout.shape, mode_layout.stride, swizzle=self._layout.swizzle),
                self._offset,
            )
        elif isinstance(key, (int, tuple)):
            # Fixed coordinate - compute the linear offset contribution
            return self._fix_mode(mode_idx, key)
        else:
            raise TypeError(f"Invalid slice key: {key}")

    def _slice_multi(self, keys: tuple) -> "Tensor | int":
        """Handle multi-dimensional slicing like tensor[i, :]."""
        if len(keys) != rank(self._layout):
            raise IndexError(f"Expected {rank(self._layout)} indices, got {len(keys)}")

        fixed_modes = []
        sliced_modes = []
        for i, key in enumerate(keys):
            if isinstance(key, (int, tuple)):
                fixed_modes.append((i, key))
            elif isinstance(key, slice) and key == slice(None):
                sliced_modes.append(i)
            else:
                raise TypeError(f"Invalid slice key at position {i}: {key}")

        if not sliced_modes:
            # All modes fixed - return the computed offset
            return self(*keys)

        # Compute LINEAR offset contribution from fixed modes (no swizzle)
        fixed_offset = sum(self._get_linear_mode_offset(i, coord) for i, coord in fixed_modes)

        new_layout = self._build_remaining_layout(sliced_modes)
        return Tensor(new_layout, self._offset + fixed_offset)

    def _build_remaining_layout(self, mode_indices) -> Layout:
        """Build a Layout from selected modes."""
        remaining_shapes = []
        remaining_strides = []
        for idx in mode_indices:
            m = mode(self._layout, idx)
            remaining_shapes.append(unwrap(m.shape))
            remaining_strides.append(unwrap(m.stride))
        return Layout(
            as_shape(remaining_shapes),
            as_shape(remaining_strides),
            swizzle=self._layout.swizzle,
        )

    def _fix_mode(self, mode_idx: int, coord) -> "Tensor | int":
        """Fix one mode to a specific coordinate value."""
        # Get LINEAR offset contribution (no swizzle)
        offset_contrib = self._get_linear_mode_offset(mode_idx, coord)
        remaining = [i for i in range(rank(self._layout)) if i != mode_idx]
        if not remaining:
            # No modes left - return the computed offset
            total_linear = self._offset + offset_contrib
            if self._layout.swizzle is not None:
                return self._layout.swizzle(total_linear)
            return total_linear
        new_layout = self._build_remaining_layout(remaining)
        return Tensor(new_layout, self._offset + offset_contrib)
