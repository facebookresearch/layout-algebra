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

#!/usr/bin/env python3
"""Regenerate all PNG figures used in the documentation.

Run from the repo root after installing visualization dependencies:
    pip install -e ".[viz]"
    python3 docs/generate_figures.py
Or, after the same install step:
    make docs
"""

import shutil
from pathlib import Path

from tensor_layouts import Layout, Swizzle
from tensor_layouts.atoms_nv import SM80_16x8x16_F16F16F16F16_TN
from tensor_layouts.layout_utils import tile_mma_grid
from tensor_layouts.viz import (
    draw_composite,
    draw_layout,
    draw_slice,
    draw_swizzle,
    draw_tiled_grid,
    draw_tv_layout,
    draw_mma_layout,
)

IMAGES = Path(__file__).resolve().parent / "images"


def main():
    IMAGES.mkdir(exist_ok=True)

    layout_8x8 = Layout((8, 8), (8, 1))

    # -- draw_layout (README + viz_api) --
    draw_layout(
        layout_8x8,
        IMAGES / "draw_layout.png",
        title="Row-Major 8x8",
        colorize=True,
    )
    # README uses a separate file with identical content
    shutil.copy(IMAGES / "draw_layout.png", IMAGES / "row_major_8x8.png")

    # -- draw_swizzle (README + viz_api) --
    draw_swizzle(
        layout_8x8,
        Swizzle(3, 0, 3),
        IMAGES / "draw_swizzle.png",
        colorize=True,
    )
    shutil.copy(IMAGES / "draw_swizzle.png", IMAGES / "swizzle_8x8.png")

    # -- color_by presets --
    draw_layout(
        layout_8x8,
        IMAGES / "color_by_row.png",
        color_layout=Layout((8, 8), (1, 0)),
        colorize=True,
    )
    draw_layout(
        layout_8x8,
        IMAGES / "color_by_col.png",
        color_layout=Layout((8, 8), (0, 1)),
        colorize=True,
    )
    draw_layout(
        layout_8x8,
        IMAGES / "color_uniform.png",
        color_layout=Layout(1, 0),
    )

    # -- hierarchical --
    hier = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
    draw_layout(
        hier,
        IMAGES / "hierarchical.png",
        flatten_hierarchical=False,
        title="With explicit nested coordinates",
    )

    # -- draw_tv_layout --
    atom = SM80_16x8x16_F16F16F16F16_TN
    draw_tv_layout(
        atom.c_layout,
        IMAGES / "draw_tv_layout.png",
        title="SM80 16x8x16 C (Thread-Value)",
        colorize=True,
    )

    # -- draw_mma_layout --
    draw_mma_layout(
        atom.a_layout,
        atom.b_layout,
        atom.c_layout,
        IMAGES / "draw_mma_layout.png",
        tile_mnk=atom.shape_mnk,
        main_title=atom.name,
    )

    # -- draw_slice --
    slice_layout = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    draw_slice(
        slice_layout,
        ((1, None), ((None, 0), None)),
        IMAGES / "draw_slice.png",
        title="((1,:),((:,0),:))",
    )

    # -- draw_composite --
    panels = [Layout((4, 4), (4, 1)), Layout((4, 4), (1, 4))]
    draw_composite(
        panels,
        IMAGES / "draw_composite.png",
        titles=["Row-Major", "Column-Major"],
        main_title="Layout Comparison",
        colorize=True,
    )

    # -- draw_tiled_grid --
    atom_layout = Layout((2, 2), (1, 2))
    grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix="C")
    draw_tiled_grid(
        grid,
        tile_shape[0],
        tile_shape[1],
        IMAGES / "draw_tiled_grid.png",
        title="SM80 16x8x16 C \u2014 2x2 atoms",
    )

    # -- show_layout (no title) --
    draw_layout(
        layout_8x8,
        IMAGES / "show_layout.png",
        colorize=True,
    )

    # -- show_swizzle (no colorize) --
    draw_swizzle(
        layout_8x8,
        Swizzle(3, 0, 3),
        IMAGES / "show_swizzle.png",
    )

    print(f"Generated {len(list(IMAGES.glob('*.png')))} figures in {IMAGES}")


if __name__ == "__main__":
    main()
