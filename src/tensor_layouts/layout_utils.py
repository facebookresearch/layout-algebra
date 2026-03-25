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

"""Convenience utilities for working with CuTe layouts.

These functions provide higher-level operations built on top of the core
layout algebra. They are useful for common tasks but not essential for
understanding the fundamental operations.
"""

from .layouts import *


def round_up(a: int, b: int) -> int:
    """Round a up to the nearest multiple of b."""
    return ((a + b - 1) // b) * b


def make_ordered_layout(shape, order: tuple = None) -> Layout:
    """Create a layout with modes ordered by the given permutation.

    Creates a layout with the given shape where the strides are determined
    by the iteration order. The order tuple specifies which modes vary fastest.

    Args:
        shape: The shape of the layout (int or tuple)
        order: Tuple specifying mode iteration order.
               order[i] = j means mode j is the i-th fastest varying.
               Default None means column-major (0, 1, 2, ...).

    Returns:
        A Layout with strides determined by the order.

    Examples:
        make_ordered_layout((4, 8))          -> Layout((4, 8), (1, 4))   # column-major
        make_ordered_layout((4, 8), (0, 1))  -> Layout((4, 8), (1, 4))   # column-major
        make_ordered_layout((4, 8), (1, 0))  -> Layout((4, 8), (8, 1))   # row-major
        make_ordered_layout((2, 3, 4), (2, 0, 1))  # mode 2 fastest, then 0, then 1
    """
    if is_int(shape):
        return Layout(shape, 1)

    shape_tuple = as_tuple(shape)
    n = len(shape_tuple)

    if order is None:
        order = tuple(range(n))

    strides = [0] * n
    current_stride = 1

    for mode_idx in order:
        strides[mode_idx] = current_stride
        current_stride *= shape_tuple[mode_idx]

    return Layout(shape_tuple, tuple(strides))


def tile_to_shape(layout: Layout, target_shape, order: tuple = None) -> Layout:
    """Tile a layout to achieve a target shape by replicating the block pattern.

    This function replicates the layout's block pattern to fill the target shape.
    It computes how many times to replicate in each dimension using ceil_div,
    then creates an ordered layout for the replication pattern.

    C++ equivalent: blocked_product(block, make_ordered_layout(product_shape, order))
    where product_shape = ceil_div(target_shape, block_shape)

    Args:
        layout: The layout to tile (the "block" pattern)
        target_shape: The desired shape to fill
        order: Tuple specifying mode iteration order for replication.
               order[i] = j means mode j is the i-th fastest varying.
               Default None means column-major (0, 1, 2, ...).

    Returns:
        A tiled layout with blocked structure that covers the target shape

    Examples:
        tile_to_shape(Layout((4, 8), (1, 4)), (16, 32))
            -> blocked_product of (4,8):(1,4) with (4,4):(1,4)
            -> replicates the 4x8 block to cover 16x32
    """
    if is_int(target_shape):
        target_shape = (target_shape,)
    target_shape = as_tuple(target_shape)

    block_shape = product_each(layout.shape)

    product_shape = tuple((t + b - 1) // b for t, b in zip(target_shape, block_shape))

    replication = make_ordered_layout(product_shape, order)

    return blocked_product(layout, replication)


def make_layout_like(layout: Layout, tiler) -> Layout:
    """Create a layout with the same shape as tiler but with layout's strides.

    This is used to extract a tile-shaped sublayout from a larger layout.

    Args:
        layout: The layout to extract from
        tiler: The desired shape (int, tuple, or Layout)

    Returns:
        A Layout with tiler's shape and appropriate strides from layout
    """
    if isinstance(tiler, Layout):
        tiler_shape = tiler.shape
    else:
        tiler_shape = normalize(tiler)

    # Get the strides for the tile shape from the layout
    flat_layout = flatten(layout)

    # Convert flat_layout strides to list for uniform processing
    if is_int(flat_layout.shape):
        flat_strides_list = [flat_layout.stride]
    else:
        flat_strides_list = list(flat_layout.stride)

    def get_strides_for_shape(shape, offset=0):
        """Recursively get strides for a shape from the flattened layout."""
        if isinstance(shape, int):
            if offset < len(flat_strides_list):
                return flat_strides_list[offset]
            return 0

        result = []
        current_offset = offset
        for s in shape:
            if is_tuple(s):
                result.append(get_strides_for_shape(s, current_offset))
                current_offset += rank(flatten(s))
            else:
                if current_offset < len(flat_strides_list):
                    result.append(flat_strides_list[current_offset])
                else:
                    result.append(0)
                current_offset += 1
        return tuple(result)

    result_strides = get_strides_for_shape(tiler_shape)
    return Layout(tiler_shape, result_strides)


def tile_mma_grid(atom, atom_layout, matrix="C", tile_mnk=None):
    """Compute the tiled MMA grid by replicating an atom across quadpairs.

    Mirrors the C++ make_tiled_mma(atom, atom_layout, Tile<M,N,K>) function.

    Two levels of tiling:
    1. Thread tiling (atom_layout): distributes atoms across thread groups
       (quadpairs). Each atom position uses different physical threads.
    2. Value tiling (tile_mnk): if the tile is larger than the atom
       arrangement's natural size, replicates the pattern with new value
       indices. Same threads, more values per thread.

    Args:
        atom: MMAAtom with a_layout, b_layout, c_layout, thr_id, shape_mnk
        atom_layout: Layout mapping (am, an) → atom_index.
        matrix: Which matrix to tile: 'A', 'B', or 'C'
        tile_mnk: Optional (M, N, K) final tile dimensions. If larger than
                  the natural tile, replicates via value tiling.

    Returns:
        grid: dict (row, col) → (physical_thread_id, value_id, logical_thread_id)
        tile_shape: (M_tiled, N_tiled, K) tuple
    """

    M_atom, N_atom, K_atom = atom.shape_mnk
    thr_id = atom.thr_id

    # Number of atoms in each dimension
    atom_shape = atom_layout.shape
    if is_int(atom_shape):
        n_atoms_m, n_atoms_n = atom_shape, 1
    else:
        n_atoms_m, n_atoms_n = atom_shape

    n_thr_per_atom = size(mode(atom.c_layout, 0))

    # QP offset: how many lanes between adjacent quadpairs
    # For SM70 QuadPair (4,2):(1,16), QP size in low half = 4
    qp_offset = n_thr_per_atom // 2 if thr_id is not None else n_thr_per_atom

    # Select atom layout and tile dimensions based on matrix
    if matrix == "C":
        atom_lyt = atom.c_layout
        row_atoms = n_atoms_m
        col_atoms = n_atoms_n
        atom_rows, atom_cols = M_atom, N_atom
    elif matrix == "A":
        atom_lyt = atom.a_layout
        row_atoms = n_atoms_m
        col_atoms = 1
        atom_rows, atom_cols = M_atom, K_atom
    elif matrix == "B":
        atom_lyt = atom.b_layout
        row_atoms = n_atoms_n
        col_atoms = 1
        atom_rows, atom_cols = N_atom, K_atom
    else:
        raise ValueError(f"matrix must be 'A', 'B', or 'C', got {matrix!r}")

    t_shape = mode(atom_lyt.shape, 0)
    v_shape = mode(atom_lyt.shape, 1)
    num_t = size(t_shape)
    num_v = size(v_shape)

    grid = {}
    for am in range(row_atoms):
        for an in range(col_atoms):
            # Determine atom index from the atom_layout
            if matrix == "C":
                atom_idx = atom_layout((am, an)) if not is_int(atom_shape) else am
            elif matrix == "A":
                # A tiles along M only; use first N-column atom
                atom_idx = atom_layout((am, 0)) if not is_int(atom_shape) else am
            else:  # B
                # B tiles along N only; am iterates over N-atoms
                atom_idx = atom_layout((0, am)) if not is_int(atom_shape) else am

            for flat_t in range(num_t):
                for flat_v in range(num_v):
                    t_coord = idx2crd(flat_t, t_shape)
                    v_coord = idx2crd(flat_v, v_shape)
                    offset = atom_lyt(t_coord, v_coord)

                    local_r = offset % atom_rows
                    local_c = offset // atom_rows

                    global_r = local_r + am * atom_rows
                    global_c = local_c + an * atom_cols

                    phys_t = thr_id(flat_t) if thr_id is not None else flat_t
                    phys_t += atom_idx * qp_offset
                    logical_t = flat_t + atom_idx * num_t

                    grid[(global_r, global_c)] = (phys_t, flat_v, logical_t)

    # Natural tile from atom arrangement (thread tiling only)
    nat_M = M_atom * n_atoms_m
    nat_N = N_atom * n_atoms_n

    # Value tiling: if tile_mnk is larger than natural, replicate with new values
    if tile_mnk is not None:
        tile_M, tile_N, tile_K = tile_mnk

        if matrix == "C":
            rep_m = tile_M // nat_M
            rep_n = tile_N // nat_N
        elif matrix == "A":
            rep_m = tile_M // nat_M
            rep_n = 1
        else:  # B
            rep_m = tile_N // nat_N
            rep_n = 1

        if rep_m * rep_n > 1:
            base_grid = dict(grid)
            base_rows = row_atoms * atom_rows
            base_cols = col_atoms * atom_cols
            grid = {}
            val_offset = 0
            for vm in range(rep_m):
                for vn in range(rep_n):
                    for (r, c), (pt, v, lt) in base_grid.items():
                        new_r = r + vm * base_rows
                        new_c = c + vn * base_cols
                        new_v = v + val_offset * num_v
                        grid[(new_r, new_c)] = (pt, new_v, lt)
                    val_offset += 1

    tile_shape = (M_atom * n_atoms_m, N_atom * n_atoms_n, K_atom)
    if tile_mnk is not None:
        tile_shape = tile_mnk
    return grid, tile_shape
