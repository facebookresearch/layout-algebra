# MIT License
#
# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
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

import tempfile

import pytest

from layout_algebra import Layout, Swizzle
from layout_algebra.atoms_amd import (
    CDNA3P_16x16x32_F32F16F16_MFMA,
    CDNA3_32x32x16_F32F8F8_MFMA,
)
from layout_algebra.atoms_nv import (
    SM80_16x8x16_F16F16F16F16_TN,
    SM90_16x8x4_F64F64F64F64_TN,
    SM120_16x8x32_F32E4M3E4M3F32_TN,
)
from layout_algebra.layout_utils import tile_mma_grid

try:
    import matplotlib.figure
    import matplotlib.pyplot as plt
    from layout_algebra.viz import (
        _format_hierarchical_cell_lines,
        _format_nested_coord,
        _coord_levels,
        _level_block_sizes,
        _level_spans,
        _get_hierarchical_cell_coords_2d,
        _get_indices_2d,
        _get_color_indices_2d,
        draw_composite,
        draw_layout,
        draw_mma_layout,
        draw_slice,
        draw_swizzle,
        draw_tiled_grid,
        draw_tv_layout,
        show_layout,
        show_swizzle,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


requires_viz = pytest.mark.skipif(
    not HAS_VIZ,
    reason="layout_algebra.viz not available (needs matplotlib)"
)

MIXED_VIZ_ATOMS = [
    # Representative cross-section for visualization smoke tests:
    # - NVIDIA: Ampere (SM80), Hopper-era scalar/legacy-style atom (SM90),
    #   and Blackwell (SM120)
    # - AMD: CDNA3 fp8 and CDNA3+ fp16
    # Keep this list small so viz tests stay lightweight while still covering
    # multiple layout families and thread/value mapping styles.
    SM80_16x8x16_F16F16F16F16_TN,
    SM90_16x8x4_F64F64F64F64_TN,
    SM120_16x8x32_F32E4M3E4M3F32_TN,
    CDNA3_32x32x16_F32F8F8_MFMA,
    CDNA3P_16x16x32_F32F16F16_MFMA,
]


@requires_viz
def test_show_layout_returns_figure_without_raising():
    """Smoke test for show_layout helper."""
    fig = show_layout(Layout((8, 8), (8, 1)))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


@requires_viz
def test_show_swizzle_returns_figure_without_raising():
    """Regression test for show_swizzle helper."""
    fig = show_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3))
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


@requires_viz
def test_draw_layout_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_layout(Layout((8, 8), (8, 1)), filename=f.name)


@requires_viz
def test_draw_swizzle_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), filename=f.name)


@requires_viz
def test_draw_slice_smoke():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_slice(Layout((4, 8), (8, 1)), (2, None), filename=f.name)


@requires_viz
@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
def test_draw_tv_layout_smoke(atom):
    m, n, _ = atom.shape_mnk
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_tv_layout(
            atom.c_layout,
            filename=f.name,
            grid_shape=(m, n),
            colorize=True,
            thr_id_layout=atom.thr_id,
        )


@requires_viz
@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
def test_draw_mma_layout_smoke(atom):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_mma_layout(
            atom.a_layout,
            atom.b_layout,
            atom.c_layout,
            filename=f.name,
            tile_mnk=atom.shape_mnk,
            colorize=True,
            thr_id_layout=atom.thr_id,
        )


@requires_viz
@pytest.mark.parametrize("atom", MIXED_VIZ_ATOMS, ids=lambda a: a.name)
def test_draw_tiled_grid_smoke(atom):
    atom_layout = Layout((2, 2), (1, 2))
    grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix="C")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_tiled_grid(
            grid,
            tile_shape[0],
            tile_shape[1],
            filename=f.name,
            title="tiled grid smoke",
        )


@requires_viz
def test_draw_composite_smoke():
    panels = [Layout((4, 4), (4, 1)), Layout((4, 4), (1, 4))]
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        draw_composite(
            panels,
            filename=f.name,
            titles=["Row-Major", "Column-Major"],
            main_title="Composite Smoke",
            colorize=True,
        )


@requires_viz
def test_get_indices_2d_row_major_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    indices = _get_indices_2d(layout)
    assert indices.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]


@requires_viz
def test_get_indices_2d_column_major_matches_logical_coordinates():
    layout = Layout((4, 3), (1, 4))
    indices = _get_indices_2d(layout)
    assert indices.tolist() == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]


@requires_viz
def test_get_color_indices_2d_by_row_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout((4, 3), (1, 0))
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ]


@requires_viz
def test_get_color_indices_2d_by_column_matches_logical_coordinates():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout((4, 3), (0, 1))
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]


@requires_viz
def test_get_color_indices_2d_uniform_layout_is_uniform():
    layout = Layout((4, 3), (3, 1))
    color_layout = Layout(1, 0)
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]


def test_get_color_indices_2d_1d_layout_is_not_treated_as_uniform():
    layout = Layout(4, 1)
    color_layout = Layout(4, 1)
    color_indices = _get_color_indices_2d(layout, color_layout)
    assert color_indices.tolist() == [[0, 1, 2, 3]]


@requires_viz
def test_get_hierarchical_cell_coords_2d_preserves_nested_coordinates():
    layout = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    coords = _get_hierarchical_cell_coords_2d(layout)
    assert coords[0, 0] == ((0, 0), (0, 0))
    assert coords[1, 0] == ((1, 0), (0, 0))
    assert coords[2, 0] == ((0, 1), (0, 0))
    assert coords[0, 1] == ((0, 0), (1, 0))
    assert coords[0, 2] == ((0, 0), (0, 1))


@requires_viz
def test_format_nested_coord_formats_hierarchical_labels():
    assert _format_nested_coord(3) == "3"
    assert _format_nested_coord((1, 2)) == "(1,2)"
    assert _format_nested_coord(((1, 2), 3)) == "((1,2),3)"


@requires_viz
def test_format_hierarchical_cell_lines_is_explicit_and_pedagogical():
    assert _format_hierarchical_cell_lines((1, 2), (3, 4), 17) == (
        "row=(1,2)",
        "col=(3,4)",
        "offset=17",
    )


@requires_viz
def test_coord_levels_flattens_nested_coordinates_for_axis_labels():
    assert _coord_levels(3) == (3,)
    assert _coord_levels((1, 2)) == (1, 2)
    assert _coord_levels(((1, 2), 3)) == (1, 2, 3)


@requires_viz
def test_level_spans_supports_three_level_hierarchy():
    assert _level_spans((2, 3, 4)) == (2, 6, 24)


@requires_viz
def test_level_block_sizes_supports_three_level_hierarchy():
    assert _level_block_sizes((2, 3, 4)) == (1, 2, 6)
