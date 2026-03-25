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

# ruff: noqa: F403, F405

"""Layout visualization with PNG/SVG/PDF output.

Visualize layouts, swizzled layouts, and tensor slices in cute-viz style.
Supports multiple output formats: SVG, PNG, PDF.

Usage:
    from tensor_layouts.viz import draw_layout, draw_swizzle, draw_slice
    from tensor_layouts import Layout, Swizzle, compose

    # Basic layout - format inferred from extension
    draw_layout(Layout((8, 8), (8, 1)), "row_major.svg")
    draw_layout(Layout((8, 8), (8, 1)), "row_major.png")
    draw_layout(Layout((8, 8), (8, 1)), "row_major.pdf")
    draw_layout(Layout((8, 8), (8, 1))) --- Inline display in Jupyter

    # Swizzled layout comparison
    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), "swizzle.png")

    # Layout with slice highlighted
    draw_slice(Layout((4, 8), (8, 1)), (2, None), "slice.svg")

    # Jupyter notebook display
    from tensor_layouts.viz import show_layout
    show_layout(Layout((8, 8), (8, 1)))

Requirements:
    pip install matplotlib numpy
"""

import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

# Use Agg (non-interactive) backend unless already set (e.g. by Jupyter)
if matplotlib.get_backend() == "agg" or not hasattr(matplotlib, "_called_from_jupyter"):
    try:
        # Check if we're in an IPython/Jupyter environment
        get_ipython()  # noqa: F821
    except NameError:
        matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from .layouts import *
from .tensor import Tensor

__all__ = [
    # draw_* (save to file or display inline)
    "draw_layout",
    "draw_swizzle",
    "draw_slice",
    "draw_tv_layout",
    "draw_mma_layout",
    "draw_tiled_grid",
    "draw_combined_mma_grid",
    "draw_copy_layout",
    "draw_composite",
    "draw_copy_atom",
    # show_* (return matplotlib Figure)
    "show_layout",
    "show_swizzle",
    "show_slice",
    "show_tv_layout",
    "show_mma_layout",
    "show_tiled_grid",
    "show_combined_mma_grid",
    "show_copy_layout",
    "show_composite",
    "show_copy_atom",
    # Demo
    "demo",
]


# =============================================================================
# Color palettes and utilities
# =============================================================================


def _max_contrast_order(n: int) -> list:
    """Return index permutation for maximum contrast between adjacent entries.

    For *n* = 8 this produces ``[0, 4, 2, 6, 1, 5, 3, 7]``, matching the
    CUTLASS ``TikzColor_BWx8`` bit-reversal ordering so that consecutive
    offsets get maximally different brightness.
    """
    if n <= 2:
        return list(range(n))
    bits = (n - 1).bit_length()

    def _bit_reverse(x: int, nbits: int) -> int:
        result = 0
        for _ in range(nbits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result

    if n & (n - 1) == 0:  # power of 2
        return [_bit_reverse(i, bits) for i in range(n)]
    # Non-power-of-2: generate for next power of 2, keep only indices < n
    n2 = 1 << bits
    return [x for x in (_bit_reverse(i, bits) for i in range(n2)) if x < n]


def _make_grayscale_palette(n: int) -> list:
    """Generate *n* grayscale colors from white to dark gray.

    Colors are reordered via a bit-reversal permutation so that consecutive
    palette indices have maximally different brightness.  This matches the
    CUTLASS ``TikzColor_BWx8`` convention and avoids the jarring
    white→dark→white sawtooth that a monotonic palette creates when it wraps
    around mid-tile.
    """
    grays = [int(255 - i * 175 / max(n - 1, 1)) for i in range(n)]
    order = _max_contrast_order(n)
    return [f"#{grays[k]:02X}{grays[k]:02X}{grays[k]:02X}" for k in order]


def _make_rainbow_palette(n: int) -> list:
    """Generate n distinct rainbow colors.

    Uses 8 base colors to match cute-viz palette for n <= 8.
    For n > 8, generates HSV colors with saturation that increases with *n*
    (small n → pastel, large n → more vivid so adjacent hues stay
    distinguishable), then applies a bit-reversal reorder so consecutive
    palette indices are maximally different in hue.
    """
    # 8 base colors matching cute-viz pastel palette
    base_colors = [
        "#AFAFFF",  # Light blue
        "#AFFFAF",  # Light green
        "#FFFFAF",  # Light yellow
        "#FFAFAF",  # Light red
        "#D2D2FF",  # Lighter blue
        "#D2FFD2",  # Lighter green
        "#FFFFD2",  # Lighter yellow
        "#FFD2D2",  # Lighter red
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    # Ramp saturation from pastel (0.35 near n=8) to vivid (0.60 at n>=32)
    # so that the smaller hue steps at large n remain distinguishable.
    import colorsys

    t = min(1.0, (n - 8) / 24)  # 0 at n=8, 1 at n>=32
    sat = 0.35 + t * 0.25  # 0.35 → 0.60
    val = 0.95 - t * 0.03  # 0.95 → 0.92
    monotonic = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        monotonic.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    order = _max_contrast_order(n)
    return [monotonic[k] for k in order]


# Default palettes (8 shades/colors)
GRAYSCALE_COLORS = _make_grayscale_palette(8)
RAINBOW_COLORS = _make_rainbow_palette(8)

# Dark colors that need white text (by hex color)
# With cute-viz pastel palette, only the darker grayscale colors need white text
DARK_COLORS = {"#505050", "#3D3D3D", "#2B2B2B", "#696969", "#504B4B"}

HIGHLIGHT_COLOR = "#FFDC96"  # Orange for highlighted cells
HIGHLIGHT_EDGE = "#FF0000"  # Red border
HIERARCHY_LEVEL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]

# Light variants for use on dark cell backgrounds
HIERARCHY_LEVEL_COLORS_LIGHT = [
    "#6baed6",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
]


# =============================================================================
# Grid data model
# =============================================================================


@dataclass
class OffsetGrid:
    """Pre-computed data for an offset-based grid visualization.

    Captures everything needed to render a flat or hierarchical layout grid,
    separating data extraction (what to draw) from rendering (how to draw it).

    Attributes:
        indices: 2D array of offset values, shape (rows, cols).
        color_indices: 2D array of palette indices, shape (rows, cols).
        highlight_mask: Optional boolean mask for highlighted cells.
        cell_coords: For hierarchical layouts, 2D object array of
            (row_coord, col_coord) tuples.  None for flat layouts.
        row_shape: Hierarchical row shape, or None for flat layouts.
        col_shape: Hierarchical col shape, or None for flat layouts.
    """

    indices: np.ndarray
    color_indices: np.ndarray
    highlight_mask: Optional[np.ndarray] = None
    cell_coords: Optional[np.ndarray] = None
    row_shape: object = None
    col_shape: object = None

    @property
    def rows(self) -> int:
        return self.indices.shape[0]

    @property
    def cols(self) -> int:
        return self.indices.shape[1]

    @property
    def is_hierarchical(self) -> bool:
        return self.cell_coords is not None


def _prepare_offset_grid(
    layout, color_layout=None, slice_spec=None, hierarchical: bool = False, eval_fn=None
) -> OffsetGrid:
    """Extract all visualization data from a layout into an OffsetGrid.

    Args:
        layout: Layout or Layout-like object (e.g. pycute Layout) to visualize.
        color_layout: Optional layout controlling cell coloring.
        slice_spec: Optional slice specification for highlight mask.
        hierarchical: If True, extract hierarchical cell coordinates.
        eval_fn: Callable mapping coordinates to offset values. Defaults to
            layout.__call__. Pass tensor.__call__ for Tensor visualization.
    """
    layout = as_layout(layout)
    cell_coords = None
    row_shape = None
    col_shape = None

    if hierarchical:
        try:
            indices, rows, cols, _, _ = _get_hierarchical_indices_2d(layout)
            cell_coords = _get_hierarchical_cell_coords_2d(layout)
            row_shape = mode(layout.shape, 0)
            col_shape = mode(layout.shape, 1)
        except (ValueError, TypeError):
            # Hierarchical index extraction can fail for layouts whose shape
            # structure doesn't decompose cleanly into 2D row/col modes
            # (e.g., rank-1 layouts, deeply nested shapes).  Fall back to
            # flat offset display which works for any layout.
            indices = _get_indices_2d(layout, eval_fn=eval_fn)
            hierarchical = False
    else:
        indices = _get_indices_2d(layout, eval_fn=eval_fn)

    color_indices = _get_color_indices_2d(layout, color_layout)

    highlight_mask = None
    if slice_spec is not None:
        highlight_mask = _get_slice_highlight_mask_2d(layout, slice_spec)

    return OffsetGrid(
        indices=indices,
        color_indices=color_indices,
        highlight_mask=highlight_mask,
        cell_coords=cell_coords,
        row_shape=row_shape,
        col_shape=col_shape,
    )


def _is_dark(hex_color: str) -> bool:
    """Return True if a hex color is dark enough to need white text."""
    # Fast path for known dark colors
    if hex_color in DARK_COLORS:
        return True
    # Parse hex and compute relative luminance
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 0.5


# =============================================================================
# Index extraction
# =============================================================================


def _get_indices_2d(layout, eval_fn=None) -> np.ndarray:
    """Extract offset indices from layout as a displayed 2D grid.

    For ordinary rank-2 layouts, displayed cell (row, col) corresponds to the
    logical coordinate (row, col), with hierarchical sub-modes flattened within
    each top-level mode. This keeps visualization semantics aligned with the
    usual matrix interpretation used throughout the docs and examples.

    Args:
        layout: Layout or Layout-like object (e.g. pycute Layout) whose shape
            determines the grid dimensions.
        eval_fn: Callable mapping coordinates to offset values. Defaults to
            layout.__call__. Pass tensor.__call__ for Tensor visualization.
    """
    layout = as_layout(layout)
    if eval_fn is None:
        eval_fn = layout
    r = rank(layout)
    total = size(layout)

    if r == 0 or r == 1:
        rows, cols = 1, total
    elif r == 2:
        row_shape = mode(layout.shape, 0)
        col_shape = mode(layout.shape, 1)
        rows = size(row_shape)
        cols = size(col_shape)
    else:
        rows, cols = 1, total

    indices = np.zeros((rows, cols), dtype=np.int32)

    if r == 2:
        for i in range(rows):
            row_coord = idx2crd(i, row_shape)
            for j in range(cols):
                col_coord = idx2crd(j, col_shape)
                indices[i, j] = eval_fn(row_coord, col_coord)
    else:
        for i in range(total):
            coord = idx2crd(i, layout.shape)
            indices[0, i] = eval_fn(coord)

    return indices


def _color_result_to_index(result) -> int:
    """Normalize a color-layout evaluation result to a scalar color index."""
    if isinstance(result, tuple):
        return 0 if len(result) == 0 else int(result[0])
    return int(result)


def _get_color_indices_2d(layout, color_layout) -> Optional[np.ndarray]:
    """Extract per-cell color indices aligned with _get_indices_2d() display semantics.

    For rank-2 color layouts, displayed cell (row, col) is colored by evaluating
    color_layout at the corresponding logical coordinate of `layout`.

    Scalar color layouts are treated as uniform color. Rank-1 color layouts are
    evaluated over the flattened displayed grid.
    """
    if color_layout is None:
        return None

    indices = _get_indices_2d(layout)
    rows, cols = indices.shape
    color_indices = np.zeros((rows, cols), dtype=np.int32)

    layout_rank = rank(layout)
    color_rank = rank(color_layout)

    if size(color_layout) == 1:
        color_idx = _color_result_to_index(color_layout(0))
        color_indices.fill(color_idx)
        return color_indices

    if color_rank <= 1:
        for i in range(rows):
            for j in range(cols):
                flat_idx = i * cols + j
                color_indices[i, j] = _color_result_to_index(color_layout(flat_idx))
        return color_indices

    if layout_rank == 2 and color_rank == 2:
        row_shape = mode(layout.shape, 0)
        col_shape = mode(layout.shape, 1)
        for i in range(rows):
            row_coord = idx2crd(i, row_shape)
            for j in range(cols):
                col_coord = idx2crd(j, col_shape)
                color_indices[i, j] = _color_result_to_index(color_layout(row_coord, col_coord))
        return color_indices

    raise ValueError(f"Unsupported color_layout rank {color_rank} for layout rank {layout_rank}")


# =============================================================================
# Core drawing functions
# =============================================================================


def _setup_axes(
    ax,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: Optional[str] = None,
    title_fontsize: float = 10,
):
    """Configure axes for a cell-grid visualization."""
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)


def _draw_grid(
    ax,
    indices: np.ndarray,
    highlight_mask: Optional[np.ndarray] = None,
    hierarchy_shapes: Optional[Tuple[object, object]] = None,
    cell_size: float = 1.0,
    show_labels: bool = True,
    title: Optional[str] = None,
    colorize: bool = False,
    color_indices: Optional[np.ndarray] = None,
    num_colors: int = 8,
    label_color: str = "blue",
    label_fontsize: float = 8,
    cell_fontsize: Optional[float] = None,
):
    """Draw a grid of cells with indices on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        indices: 2D array of index values
        highlight_mask: Boolean mask aligned with `indices`
        hierarchy_shapes: Optional `(row_shape, col_shape)` pair used to draw
            hierarchy boundary overlays on top of a flattened displayed grid
        cell_size: Size of each cell
        show_labels: Whether to show row/column labels
        title: Optional title for the plot
        colorize: If True, use rainbow colors; if False, use grayscale
        color_indices: 2D array of per-cell color indices aligned with `indices`.
        num_colors: Number of colors in palette (default 8)
        label_color: Color for row/column margin labels
        label_fontsize: Font size for margin labels
        cell_fontsize: Font size for cell value text. If None, auto-scaled
            based on the figure width and number of columns so text fits
            within cells.
    """
    rows, cols = indices.shape

    # Auto-scale cell font size: estimate inches per cell from the figure,
    # then pick a font size that fits the widest value inside cells.
    if cell_fontsize is None:
        fig = ax.get_figure()
        if fig is not None:
            fig_w, _ = fig.get_size_inches()
            n_axes = max(len(fig.axes), 1)
            panel_w = fig_w / n_axes * 0.85
            inches_per_cell = panel_w / max(cols, 1)
            max_val = int(np.max(np.abs(indices))) if indices.size > 0 else 0
            n_digits = max(len(str(max_val)), 1)
            max_fs = inches_per_cell * 72 / (n_digits * 0.6)
            cell_fontsize = max(4, min(8, max_fs))
            label_fontsize = min(label_fontsize, cell_fontsize)
        else:
            cell_fontsize = 8

    # Build the appropriate palette
    if colorize:
        colors = _make_rainbow_palette(num_colors)
    else:
        colors = _make_grayscale_palette(num_colors)

    _setup_axes(ax, (-0.5, cols + 0.5), (-0.5, rows + 0.5), title=title)

    final_facecolors = np.empty((rows, cols), dtype=object)
    highlighted_cells = []

    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])

            # Determine highlight state
            is_hl = bool(highlight_mask[i, j]) if highlight_mask is not None else False

            # Determine base face color
            if color_indices is not None:
                color_idx = int(color_indices[i, j]) % len(colors)
            else:
                # Default: color by cell value
                color_idx = idx % len(colors)

            base_facecolor = colors[color_idx]
            final_facecolors[i, j] = HIGHLIGHT_COLOR if is_hl else base_facecolor

            # Draw base cell first; highlights are overlaid later so their
            # borders are not covered by neighboring cells.
            rect = patches.Rectangle(
                (j, i),
                cell_size,
                cell_size,
                facecolor=base_facecolor,
                edgecolor="black",
                linewidth=1,
                zorder=1,
            )
            ax.add_patch(rect)

            if is_hl:
                highlighted_cells.append((i, j))

    if hierarchy_shapes is not None:
        row_shape, col_shape = hierarchy_shapes
        _draw_hierarchy_boundary_lines(
            ax,
            rows,
            cols,
            _level_block_sizes(row_shape),
            _level_block_sizes(col_shape),
            zorder_base=4,
        )

    # Two-pass rendering for correct z-ordering:
    #   Pass 1 (below): draw all base cells at zorder=1
    #   Pass 2 (line ~469): overlay highlights at zorder=6 with thicker borders
    # This ensures highlight borders aren't obscured by adjacent base cells.
    for i, j in highlighted_cells:
        rect = patches.Rectangle(
            (j, i),
            cell_size,
            cell_size,
            facecolor=HIGHLIGHT_COLOR,
            edgecolor=HIGHLIGHT_EDGE,
            linewidth=2,
            zorder=6,
        )
        ax.add_patch(rect)

    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])
            facecolor = final_facecolors[i, j]
            text_color = "white" if _is_dark(facecolor) else "black"
            ax.text(
                j + 0.5,
                i + 0.5,
                str(idx),
                ha="center",
                va="center",
                fontsize=cell_fontsize,
                color=text_color,
                zorder=7,
            )

    if show_labels:
        # Row labels (left)
        for i in range(rows):
            ax.text(
                -0.3,
                i + 0.5,
                str(i),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_color,
                zorder=8,
            )

        # Column labels (top)
        for j in range(cols):
            ax.text(
                j + 0.5,
                -0.3,
                str(j),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_color,
                zorder=8,
            )


def _save_figure(fig, filename, dpi: int = 150):
    """Save figure to file, or display inline in Jupyter if filename is None."""
    if filename is None:
        # Inline display in Jupyter notebook
        try:
            import io

            from IPython.display import display, Image

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
        except ImportError:
            plt.show()
        return

    path = Path(filename)
    fmt = path.suffix.lower().lstrip(".")

    if fmt == "svg":
        fig.savefig(filename, format="svg", bbox_inches="tight")
    elif fmt == "pdf":
        fig.savefig(filename, format="pdf", bbox_inches="tight")
    elif fmt in ("png", "jpg", "jpeg"):
        fig.savefig(filename, format=fmt, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(filename, format="png", dpi=dpi, bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# Composite Figure API
# =============================================================================


def _build_composite_figure(
    panels: list,
    arrangement: str = "horizontal",
    titles: Optional[list] = None,
    main_title: Optional[str] = None,
    panel_size: Tuple[float, float] = (4, 4),
    colorize: bool = False,
    tv_mode: bool = False,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Build the composite figure used by draw_composite/show_composite."""
    n = len(panels)
    if n == 0:
        raise ValueError("panels list cannot be empty")

    # Parse arrangement
    if arrangement == "horizontal":
        nrows, ncols = 1, n
    elif arrangement == "vertical":
        nrows, ncols = n, 1
    elif arrangement.startswith("grid:"):
        dims = arrangement[5:].split("x")
        nrows, ncols = int(dims[0]), int(dims[1])
    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")

    # Create figure
    fig_width = ncols * panel_size[0]
    fig_height = nrows * panel_size[1]

    fig, axes_array = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    # Normalize axes to list
    if n == 1:
        axes = [axes_array]
    elif nrows == 1 or ncols == 1:
        axes = list(axes_array)
    else:
        axes = [axes_array[i, j] for i in range(nrows) for j in range(ncols)]

    # Process each panel
    if len(panels) > nrows * ncols:
        import warnings

        warnings.warn(
            f"{len(panels)} panels provided but grid has only "
            f"{nrows * ncols} cells ({nrows}x{ncols}); "
            f"extra panels will be dropped",
            stacklevel=3,
        )
    for idx, panel in enumerate(panels):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Extract layout and options
        if isinstance(panel, tuple):
            layout, opts = panel
        else:
            layout = panel
            opts = {}

        # Unwrap Tensor for offset-grid rendering
        eval_fn = None
        if isinstance(layout, Tensor):
            eval_fn = layout.__call__
            layout = layout.layout

        # Merge with defaults
        panel_colorize = opts.get("colorize", colorize)
        panel_tv_mode = opts.get("tv_mode", tv_mode)
        color_layout = opts.get("color_layout", None)
        num_colors = opts.get("num_colors", 8)
        panel_flatten = opts.get("flatten_hierarchical", flatten_hierarchical)
        panel_label_levels = opts.get("label_hierarchy_levels", label_hierarchy_levels)

        # Get title
        title = titles[idx] if titles and idx < len(titles) else None

        # Draw the panel
        if panel_tv_mode:
            _draw_tv_grid(
                ax,
                layout,
                title=title,
                colorize=panel_colorize,
                num_colors=opts.get("num_colors", num_colors),
                grid_rows=opts.get("grid_rows"),
                grid_cols=opts.get("grid_cols"),
                thr_id_layout=opts.get("thr_id_layout"),
                col_major=opts.get("col_major", True),
            )
        else:
            # Check if this panel should use hierarchical rendering
            r = rank(layout)
            is_hier = (
                r == 2
                and not panel_flatten
                and (
                    isinstance(mode(layout.shape, 0), tuple)
                    or isinstance(mode(layout.shape, 1), tuple)
                )
            )
            grid = _prepare_offset_grid(
                layout,
                color_layout=color_layout,
                eval_fn=eval_fn,
                hierarchical=is_hier,
            )
            if grid.is_hierarchical:
                _draw_hierarchical_grid(
                    ax,
                    grid.indices,
                    grid.rows,
                    grid.cols,
                    cell_coords=grid.cell_coords,
                    row_shape=grid.row_shape,
                    col_shape=grid.col_shape,
                    title=title,
                    colorize=panel_colorize,
                    color_indices=grid.color_indices,
                    flatten_hierarchical=False,
                    label_hierarchy_levels=panel_label_levels,
                    num_colors=num_colors,
                )
            else:
                _draw_grid(
                    ax,
                    grid.indices,
                    title=title,
                    colorize=panel_colorize,
                    color_indices=grid.color_indices,
                    num_colors=num_colors,
                )

    # Hide unused axes
    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def draw_composite(
    panels: list,
    filename: str,
    arrangement: str = "horizontal",
    titles: Optional[list] = None,
    main_title: Optional[str] = None,
    dpi: int = 150,
    panel_size: Tuple[float, float] = (4, 4),
    colorize: bool = False,
    tv_mode: bool = False,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Draw multiple layouts in a single composite figure.

    This allows composing sub-images into bigger figures, useful for:
    - Copy atom visualizations (source and destination side by side)
    - Before/after comparisons
    - Multiple related layouts

    For MMA visualizations, use the dedicated draw_mma_layout() function instead.

    Args:
        panels: List of Layout/Tensor objects or (Layout/Tensor, options_dict)
                tuples. Per-panel options override the top-level defaults:
                  colorize, color_layout, num_colors -- offset-grid options
                  tv_mode -- if True, render this panel as a TV grid
                  grid_rows, grid_cols, thr_id_layout, col_major -- TV options
                  flatten_hierarchical, label_hierarchy_levels -- hierarchy options
        filename: Output path (.svg, .png, or .pdf)
        arrangement: How to arrange panels:
            - "horizontal": side by side (1 row)
            - "vertical": stacked (1 column)
            - "grid:RxC": R rows by C columns (e.g., "grid:2x2")
        titles: Optional list of titles for each panel
        main_title: Optional title for the entire figure
        dpi: Resolution for raster formats
        panel_size: Size of each panel in inches (width, height)
        colorize: Default colorize setting for all panels
        tv_mode: If True, render panels as TV layouts with T/V labels
        flatten_hierarchical: Default for all panels. If False, show explicit
            nested coordinate labels for hierarchical layouts
        label_hierarchy_levels: Default for all panels. If True, annotate axes
            with hierarchy level labels

    Example:
        # Side-by-side comparison
        draw_composite(
            [linear_layout, swizzled_layout],
            "comparison.svg",
            arrangement="horizontal",
            titles=["Linear", "Swizzled"]
        )
    """
    fig = _build_composite_figure(
        panels,
        arrangement=arrangement,
        titles=titles,
        main_title=main_title,
        panel_size=panel_size,
        colorize=colorize,
        tv_mode=tv_mode,
        flatten_hierarchical=flatten_hierarchical,
        label_hierarchy_levels=label_hierarchy_levels,
    )
    _save_figure(fig, filename, dpi)


# =============================================================================
# Public API
# =============================================================================


def _get_hierarchical_indices_2d(layout) -> Tuple[np.ndarray, int, int, tuple, tuple]:
    """Extract indices and dimensions for hierarchical layout visualization.

    For a layout with shape ((inner_rows, outer_rows), (inner_cols, outer_cols)),
    returns the flattened indices and the inner/outer dimensions.

    Returns:
        (indices, total_rows, total_cols, row_structure, col_structure)
    """
    r = rank(layout)
    if r != 2:
        raise ValueError(f"Hierarchical layout must be rank 2, got rank {r}")

    row_shape = mode(layout.shape, 0)
    col_shape = mode(layout.shape, 1)

    total_rows = size(row_shape)
    total_cols = size(col_shape)
    row_structure = row_shape if isinstance(row_shape, tuple) else (row_shape,)
    col_structure = col_shape if isinstance(col_shape, tuple) else (col_shape,)

    indices = np.zeros((total_rows, total_cols), dtype=np.int32)

    for i in range(total_rows):
        for j in range(total_cols):
            row_coord = idx2crd(i, row_shape)
            col_coord = idx2crd(j, col_shape)
            indices[i, j] = layout(row_coord, col_coord)

    return indices, total_rows, total_cols, row_structure, col_structure


def _format_nested_coord(coord) -> str:
    """Format a scalar or nested tuple coordinate compactly."""
    if isinstance(coord, tuple):
        return "(" + ",".join(_format_nested_coord(c) for c in coord) + ")"
    return str(coord)


def _coord_levels(coord) -> tuple[int, ...]:
    """Flatten a scalar or nested coordinate into level-ordered scalar components."""
    flat = flatten(coord)
    return tuple(flat) if isinstance(flat, tuple) else (int(flat),)


def _level_spans(shape) -> tuple[int, ...]:
    """Return the span at each flattened coordinate level.

    For flattened shape (s0, s1, s2), returns (s0, s0*s1, s0*s1*s2).
    Level 0 is the fastest-varying coordinate.
    """
    flat_shape = flatten(shape)
    flat_shape = tuple(flat_shape) if isinstance(flat_shape, tuple) else (flat_shape,)
    spans = []
    prod = 1
    for dim in flat_shape:
        prod *= int(dim)
        spans.append(prod)
    return tuple(spans)


def _level_block_sizes(shape) -> tuple[int, ...]:
    """Return the run length for each flattened coordinate level.

    For flattened shape (s0, s1, s2), returns (1, s0, s0*s1). These are the
    sizes of contiguous displayed runs for which level-k stays constant.
    """
    spans = _level_spans(shape)
    if not spans:
        return ()
    return (1,) + spans[:-1]


def _hierarchy_level_color(level: int, for_dark_bg: bool = False) -> str:
    """Color for a hierarchy level.

    Level 0 is the fastest-varying coordinate and has no corresponding tile
    boundary, so keep it neutral. Coarser levels map to the hierarchy palette.

    When ``for_dark_bg`` is True, return a lighter variant so the text remains
    readable against dark cell backgrounds.
    """
    if level == 0:
        return "#cccccc" if for_dark_bg else "#333333"
    palette = HIERARCHY_LEVEL_COLORS_LIGHT if for_dark_bg else HIERARCHY_LEVEL_COLORS
    return palette[(level - 1) % len(palette)]


def _draw_hierarchy_boundary_lines(
    ax,
    rows: int,
    cols: int,
    row_block_sizes: tuple[int, ...],
    col_block_sizes: tuple[int, ...],
    zorder_base: float = 4,
):
    """Draw hierarchy boundary lines over a displayed grid.

    Boundary lines are drawn from finest to coarsest so coarser levels sit
    above finer ones at intersections. Perimeter strokes are included for
    each hierarchy level beyond level 0, which gives edge tiles full colored
    boxes rather than only internal separators.
    """
    n_row_levels = len(row_block_sizes)
    n_col_levels = len(col_block_sizes)

    def _is_shadowed_by_coarser(level: int, pos: int, block_sizes: tuple[int, ...]) -> bool:
        """Return True if a same-orientation coarser hierarchy line also sits at pos."""
        for coarser_level in range(level + 1, len(block_sizes)):
            coarser_block = block_sizes[coarser_level]
            if coarser_block > 0 and pos % coarser_block == 0:
                return True
        return False

    def _draw_boundary_line(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: str,
        linewidth: float,
        zorder: float,
    ):
        """Draw a hierarchy boundary segment with consistent stroke caps."""
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            solid_capstyle="butt",
        )

    max_levels = max(n_row_levels, n_col_levels)

    for level in range(1, max_levels):
        color = HIERARCHY_LEVEL_COLORS[(level - 1) % len(HIERARCHY_LEVEL_COLORS)]
        linewidth = 2.0 + 1.2 * (level - 1)
        zorder = zorder_base + level

        # Always draw a full perimeter for levels present on either axis so
        # asymmetric hierarchies still produce closed colored boxes.
        _draw_boundary_line(0, 0, cols, 0, color, linewidth, zorder)
        _draw_boundary_line(0, rows, cols, rows, color, linewidth, zorder)
        _draw_boundary_line(0, 0, 0, rows, color, linewidth, zorder)
        _draw_boundary_line(cols, 0, cols, rows, color, linewidth, zorder)

        if level < n_row_levels:
            block_size = row_block_sizes[level]
            for i in range(block_size, rows, block_size):
                if _is_shadowed_by_coarser(level, i, row_block_sizes):
                    continue
                _draw_boundary_line(0, i, cols, i, color, linewidth, zorder)

        if level < n_col_levels:
            block_size = col_block_sizes[level]
            for j in range(block_size, cols, block_size):
                if _is_shadowed_by_coarser(level, j, col_block_sizes):
                    continue
                _draw_boundary_line(j, 0, j, rows, color, linewidth, zorder)


def _format_hierarchical_cell_lines(row_coord, col_coord, offset: int) -> tuple[str, str, str]:
    """Format pedagogical hierarchical cell labels.

    Returns three explicit lines:
      row=<row-coord>
      col=<col-coord>
      offset=<offset>
    """
    return (
        f"row={_format_nested_coord(row_coord)}",
        f"col={_format_nested_coord(col_coord)}",
        f"offset={offset}",
    )


@lru_cache(maxsize=None)
def _measure_text_width_pts(
    text: str, fontsize: float, family: str = "sans-serif", weight: str = "normal"
) -> float:
    """Return rendered text width in points using advance widths."""
    if not text:
        return 0.0
    fontprops = FontProperties(family=family, weight=weight, size=fontsize)
    w, _, _ = TextToPath().get_text_width_height_descent(text, fontprops, ismath=False)
    return w


def _hierarchical_label_margins(
    n_row_levels: int,
    n_col_levels: int,
    label_hierarchy_levels: bool,
    row_label_band_spacing: float = 1.2,
    col_label_band_spacing: float = 0.8,
) -> tuple[float, float, float, float]:
    """Return margins and corner gaps for hierarchical axis labels."""
    corner_gap_x = 0.0
    corner_gap_y = 0.0
    if not label_hierarchy_levels:
        return 0.9, 0.9, corner_gap_x, corner_gap_y

    # Reserve a corner gap so the row[...] and col[...] label bands do not
    # crowd each other at the top-left corner for asymmetric hierarchies.
    corner_gap_x = 0.45 + 0.10 * max(n_col_levels - 1, 0)
    corner_gap_y = 0.45 + 0.10 * max(n_row_levels - 1, 0)
    left_margin = 0.9 + corner_gap_x + row_label_band_spacing * max(n_row_levels, 0)
    top_margin = 0.9 + corner_gap_y + col_label_band_spacing * max(n_col_levels, 0)
    return left_margin, top_margin, corner_gap_x, corner_gap_y


def _auto_hierarchical_figsize(
    layout, indices: np.ndarray, rows: int, cols: int, label_hierarchy_levels: bool
) -> Tuple[float, float]:
    """Estimate a readable default figure size for nested hierarchical views."""
    row_shape = mode(layout.shape, 0)
    col_shape = mode(layout.shape, 1)
    n_row_levels = len(_level_block_sizes(row_shape))
    n_col_levels = len(_level_block_sizes(col_shape))
    left_margin, top_margin, _, _ = _hierarchical_label_margins(
        n_row_levels, n_col_levels, label_hierarchy_levels
    )

    max_levels = max(n_row_levels, n_col_levels, 1)
    coord_fontsize = max(3.2, 5.5 - 0.45 * (max_levels - 1))
    offset_fontsize = coord_fontsize

    max_row_width_pts = max(
        (
            _measure_text_width_pts(
                f"row={_format_nested_coord(idx2crd(i, row_shape))}",
                coord_fontsize,
                family="monospace",
            )
            for i in range(rows)
        ),
        default=0.0,
    )
    max_col_width_pts = max(
        (
            _measure_text_width_pts(
                f"col={_format_nested_coord(idx2crd(j, col_shape))}",
                coord_fontsize,
                family="monospace",
            )
            for j in range(cols)
        ),
        default=0.0,
    )
    offset_width_pts = _measure_text_width_pts(
        "offset=", offset_fontsize, family="monospace"
    ) + max(
        (
            _measure_text_width_pts(
                str(int(idx)),
                offset_fontsize,
                family="monospace",
                weight="bold",
            )
            for idx in indices.flat
        ),
        default=0.0,
    )

    widest_content_pts = max(max_row_width_pts, max_col_width_pts, offset_width_pts)

    # Cell text starts at x=0.12 within the cell. Size the figure so that the
    # widest in-cell line plus a small right margin fits within the remaining
    # 88% of the cell width, after accounting for Matplotlib's default subplot
    # margins and the extra hierarchical label bands.
    right_padding_pts = 2.0
    required_cell_width_pts = (widest_content_pts + right_padding_pts) / 0.88

    line_height_pts = max(coord_fontsize, offset_fontsize) * 1.15
    required_cell_height_pts = max(
        (line_height_pts + 0.5) / 0.28,
        (line_height_pts + 0.5) / 0.22,
    )

    scale_pts = max(required_cell_width_pts, required_cell_height_pts)
    subplot_width_frac = (
        matplotlib.rcParams["figure.subplot.right"] - matplotlib.rcParams["figure.subplot.left"]
    )
    subplot_height_frac = (
        matplotlib.rcParams["figure.subplot.top"] - matplotlib.rcParams["figure.subplot.bottom"]
    )
    total_x_range = cols + 0.5 + left_margin
    total_y_range = rows + 0.5 + top_margin

    fig_width = scale_pts * total_x_range / (72.0 * subplot_width_frac)
    fig_height = scale_pts * total_y_range / (72.0 * subplot_height_frac)
    fig_width = max(fig_width, cols * 0.8 + 1.0)
    fig_height = max(fig_height, rows * 0.8 + 1.0)
    return fig_width, fig_height


def _draw_colored_coord_line(
    ax,
    x: float,
    y: float,
    prefix: str,
    coord,
    base_color: str,
    fontsize: float,
    use_level_colors: bool,
    for_dark_bg: bool = False,
):
    """Draw row=/col= line with per-coordinate level colors.

    When enabled, the scalar coordinate components use the same colors as the
    corresponding hierarchy-level margin labels/boundaries.

    This helper lays out colored coordinate components using measured
    monospace text widths in point offsets so the mixed-color pieces remain
    aligned across raster and SVG outputs.
    """
    levels = _coord_levels(coord)
    pieces = [(f"{prefix}=(", base_color)]
    for level, value in enumerate(levels):
        pieces.append(
            (
                str(value),
                (_hierarchy_level_color(level, for_dark_bg) if use_level_colors else base_color),
            )
        )
        if level != len(levels) - 1:
            pieces.append((",", base_color))
    pieces.append((")", base_color))

    offset_pts = 0
    for text, color in pieces:
        trans = mtransforms.offset_copy(
            ax.transData, fig=ax.figure, x=offset_pts, y=0, units="points"
        )
        ax.text(
            x,
            y,
            text,
            transform=trans,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            family="monospace",
        )
        offset_pts += _measure_text_width_pts(text, fontsize, family="monospace")


def _get_hierarchical_cell_coords_2d(layout) -> np.ndarray:
    """Return per-cell hierarchical coordinates aligned with the displayed grid.

    Each entry is a `(row_coord, col_coord)` tuple, where each coordinate may be
    scalar or nested depending on the corresponding top-level mode shape.
    """
    _, rows, cols, _, _ = _get_hierarchical_indices_2d(layout)
    row_shape = mode(layout.shape, 0)
    col_shape = mode(layout.shape, 1)

    coords = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        row_coord = idx2crd(i, row_shape)
        for j in range(cols):
            col_coord = idx2crd(j, col_shape)
            coords[i, j] = (row_coord, col_coord)
    return coords


def _draw_hierarchical_grid(
    ax,
    indices,
    rows,
    cols,
    cell_coords,
    row_shape,
    col_shape,
    cell_size: float = 1.0,
    show_labels: bool = True,
    title: Optional[str] = None,
    colorize: bool = False,
    color_indices: Optional[np.ndarray] = None,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
    num_colors: int = 8,
):
    """Draw a hierarchical layout grid.

    Args:
        indices: 2D array of offset values (rows × cols)
        rows: Number of grid rows
        cols: Number of grid columns
        cell_coords: 2D object array of (row_coord, col_coord) tuples
        row_shape: Hierarchical row shape (e.g. (4, 2) for inner=4, outer=2)
        col_shape: Hierarchical column shape
        flatten_hierarchical: If True, show flat grid with offset values.
                              If False, show explicit pedagogical labels in
                              each cell:
                                - row=... for the nested row coordinate
                                - col=... for the nested column coordinate
                                - offset=... for the resulting offset
                              plus blue tile boundary lines.
        label_hierarchy_levels: If True, annotate axes with each hierarchy
                              level at block/tile granularity using labels
                              such as row[1]=..., row[2]=..., col[1]=...,
                              col[2]=.... Label colors match the corresponding
                              hierarchy boundary lines. If False, keep axes
                              simple (R0, R1, ... / C0, C1, ...).
    """
    row_block_sizes = _level_block_sizes(row_shape)
    col_block_sizes = _level_block_sizes(col_shape)
    n_row_levels = len(row_block_sizes)
    n_col_levels = len(col_block_sizes)

    # Build palette
    if colorize:
        colors = _make_rainbow_palette(num_colors)
    else:
        colors = _make_grayscale_palette(num_colors)

    row_label_band_spacing = 1.2
    col_label_band_spacing = 0.8
    left_margin, top_margin, corner_gap_x, corner_gap_y = _hierarchical_label_margins(
        n_row_levels,
        n_col_levels,
        label_hierarchy_levels=(not flatten_hierarchical and label_hierarchy_levels),
        row_label_band_spacing=row_label_band_spacing,
        col_label_band_spacing=col_label_band_spacing,
    )
    _setup_axes(ax, (-left_margin, cols + 0.5), (-top_margin, rows + 0.5), title=title)

    max_levels = max(n_row_levels, n_col_levels, 1)
    coord_fontsize = max(3.2, 5.5 - 0.45 * (max_levels - 1))
    offset_fontsize = coord_fontsize
    row_axis_label_fontsize = max(5.0, min(6.5, 56.0 / max(rows, 1)))
    col_axis_label_fontsize = max(4.0, min(6.0, 48.0 / max(cols, 1)))

    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])
            if color_indices is not None:
                color_idx = int(color_indices[i, j]) % len(colors)
            else:
                color_idx = idx % len(colors)
            facecolor = colors[color_idx]
            if flatten_hierarchical:
                edgecolor = "black"
                linewidth = 0.5
            else:
                # In pedagogical hierarchical mode, keep per-cell borders light
                # so hierarchy boundary lines stand out clearly.
                edgecolor = "#444444"
                linewidth = 0.3

            rect = patches.Rectangle(
                (j, i),
                cell_size,
                cell_size,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=1,
            )
            ax.add_patch(rect)

            is_dark_bg = _is_dark(facecolor)
            text_color = "white" if is_dark_bg else "black"
            if flatten_hierarchical:
                # Draw flat offset value in cell
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    str(idx),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
            else:
                # Draw explicit pedagogical labels: row coordinate, column
                # coordinate, and resulting offset.
                row_coord, col_coord = cell_coords[i, j]
                row_line, col_line, off_line = _format_hierarchical_cell_lines(
                    row_coord, col_coord, idx
                )
                x_left = j + 0.12
                _draw_colored_coord_line(
                    ax,
                    x_left,
                    i + 0.22,
                    "row",
                    row_coord,
                    text_color,
                    coord_fontsize,
                    use_level_colors=label_hierarchy_levels,
                    for_dark_bg=is_dark_bg,
                )
                _draw_colored_coord_line(
                    ax,
                    x_left,
                    i + 0.50,
                    "col",
                    col_coord,
                    text_color,
                    coord_fontsize,
                    use_level_colors=label_hierarchy_levels,
                    for_dark_bg=is_dark_bg,
                )
                offset_label, offset_value = off_line.split("=", 1)
                offset_label_text = f"{offset_label}="
                ax.text(
                    x_left,
                    i + 0.78,
                    offset_label_text,
                    ha="left",
                    va="center",
                    fontsize=offset_fontsize,
                    color=text_color,
                    family="monospace",
                )
                offset_label_pts = _measure_text_width_pts(
                    offset_label_text, offset_fontsize, family="monospace"
                )
                trans = mtransforms.offset_copy(
                    ax.transData, fig=ax.figure, x=offset_label_pts, y=0, units="points"
                )
                ax.text(
                    x_left,
                    i + 0.78,
                    offset_value,
                    transform=trans,
                    ha="left",
                    va="center",
                    fontsize=offset_fontsize,
                    color=text_color,
                    fontweight="bold",
                    family="monospace",
                )

    # Draw blue tile boundary lines for nested view
    if not flatten_hierarchical:
        _draw_hierarchy_boundary_lines(
            ax, rows, cols, row_block_sizes, col_block_sizes, zorder_base=4
        )

    if show_labels:
        if not flatten_hierarchical and label_hierarchy_levels:
            # Annotate each hierarchy level at the block/tile granularity where
            # that level is constant, rather than labeling every displayed row/col.
            for level in range(n_row_levels):
                block_size = row_block_sizes[level]
                color = _hierarchy_level_color(level)
                x = -0.55 - corner_gap_x - row_label_band_spacing * level
                for start in range(0, rows, block_size):
                    center = start + block_size / 2
                    value = _coord_levels(idx2crd(start, row_shape))[level]
                    ax.text(
                        x,
                        center,
                        f"row[{level}]={value}",
                        ha="center",
                        va="center",
                        fontsize=row_axis_label_fontsize,
                        color=color,
                        fontweight="bold",
                    )
            for level in range(n_col_levels):
                block_size = col_block_sizes[level]
                color = _hierarchy_level_color(level)
                y = -0.55 - corner_gap_y - col_label_band_spacing * level
                for start in range(0, cols, block_size):
                    center = start + block_size / 2
                    value = _coord_levels(idx2crd(start, col_shape))[level]
                    ax.text(
                        center,
                        y,
                        f"col[{level}]={value}",
                        ha="center",
                        va="center",
                        fontsize=col_axis_label_fontsize,
                        color=color,
                        fontweight="bold",
                    )
        else:
            # Single-level labels
            for i in range(rows):
                ax.text(
                    -0.3,
                    i + 0.5,
                    f"R{i}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue",
                )
            for j in range(cols):
                ax.text(
                    j + 0.5,
                    -0.3,
                    f"C{j}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="blue",
                )


def _build_layout_figure(
    layout,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    color_layout: Optional[Layout] = None,
    color_by: Optional[str] = None,
    num_colors: int = 8,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Build the layout figure used by draw_layout/show_layout.

    Accepts Layout or Tensor. When given a Tensor, cells display offset-adjusted
    values and the default title includes the base offset.
    """
    # Unwrap Tensor: use its layout for shape/structure, its __call__ for values
    eval_fn = None
    if isinstance(layout, Tensor):
        tensor = layout
        eval_fn = tensor.__call__
        layout = tensor.layout
        if title is None:
            title = repr(tensor)

    # Resolve color_by shorthand to a color_layout
    if color_by is not None:
        if color_layout is not None:
            raise ValueError("color_by and color_layout are mutually exclusive")
        shape = layout.shape
        if color_by == "row":
            color_layout = Layout(shape, (1, 0)) if rank(layout) == 2 else None
        elif color_by == "column":
            color_layout = Layout(shape, (0, 1)) if rank(layout) == 2 else None
        elif color_by == "offset":
            color_layout = None  # default behavior
        else:
            raise ValueError(
                f"Unknown color_by value: {color_by!r} (expected 'row', 'column', or 'offset')"
            )
        colorize = True

    # Rank >= 3: decompose into multiple 2D panels along the leading mode(s)
    r = rank(layout)
    if r >= 3:
        # Group all leading modes (0..r-3) into a flat slice index,
        # keeping the last two modes as the 2D grid per panel.
        outer_sizes = [size(mode(layout.shape, i)) for i in range(r - 2)]
        n_panels = 1
        for s in outer_sizes:
            n_panels *= s
        inner_shape = as_shape(tuple(mode(layout.shape, i) for i in range(r - 2, r)))
        inner_stride = as_shape(tuple(mode(layout.stride, i) for i in range(r - 2, r)))

        # Build per-panel sub-layouts and eval functions
        sub_layouts = []
        sub_evals = []
        panel_titles = []
        for flat_idx in range(n_panels):
            outer_coord = idx2crd(flat_idx, as_shape(outer_sizes))
            if not isinstance(outer_coord, tuple):
                outer_coord = (outer_coord,)

            # Offset contribution from fixed outer modes
            offset = 0
            for dim_idx, coord_val in enumerate(outer_coord):
                m = mode(layout, dim_idx)
                offset += crd2offset(coord_val, m.shape, m.stride)

            sub = Layout(inner_shape, inner_stride, swizzle=layout.swizzle)
            sub_layouts.append(sub)

            # Eval function: offset + sub_layout(coord), optionally with swizzle
            _off = offset
            _sub = sub
            if layout.swizzle is not None:
                _sw = layout.swizzle

                def _make_eval(o, s, sw):
                    def fn(*args):
                        coord = args[0] if len(args) == 1 else args
                        linear = o + crd2offset(coord, s.shape, s.stride)
                        return sw(linear)

                    return fn

                sub_evals.append(_make_eval(_off, _sub, _sw))
            else:

                def _make_eval_plain(o, s):
                    def fn(*args):
                        coord = args[0] if len(args) == 1 else args
                        return o + crd2offset(coord, s.shape, s.stride)

                    return fn

                sub_evals.append(_make_eval_plain(_off, _sub))

            if len(outer_sizes) == 1:
                panel_titles.append(f"mode[0]={outer_coord[0]}")
            else:
                coord_str = ", ".join(str(c) for c in outer_coord)
                panel_titles.append(f"outer=({coord_str})")

        # Render panels using subplots
        ncols = min(n_panels, 4)
        nrows = (n_panels + ncols - 1) // ncols
        panel_size = (3.5, 3.5)
        if figsize is None:
            figsize = (ncols * panel_size[0], nrows * panel_size[1])

        fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize)
        if n_panels == 1:
            axes = [axes_array]
        elif nrows == 1 or ncols == 1:
            axes = list(axes_array)
        else:
            axes = [axes_array[i, j] for i in range(nrows) for j in range(ncols)]

        for idx in range(n_panels):
            grid = _prepare_offset_grid(
                sub_layouts[idx], color_layout=color_layout, eval_fn=sub_evals[idx]
            )
            _draw_grid(
                axes[idx],
                grid.indices,
                title=panel_titles[idx],
                colorize=colorize,
                color_indices=grid.color_indices,
                num_colors=num_colors,
            )

        for idx in range(n_panels, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(title or str(layout), fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    # Check if this is a hierarchical layout (has nested tuple shapes)
    is_hierarchical = r == 2 and (
        isinstance(mode(layout.shape, 0), tuple) or isinstance(mode(layout.shape, 1), tuple)
    )

    want_hierarchical = is_hierarchical and not flatten_hierarchical
    grid = _prepare_offset_grid(
        layout,
        color_layout=color_layout,
        hierarchical=want_hierarchical,
        eval_fn=eval_fn,
    )

    if figsize is None:
        if grid.is_hierarchical:
            figsize = _auto_hierarchical_figsize(
                layout, grid.indices, grid.rows, grid.cols, label_hierarchy_levels
            )
        else:
            cell_scale = 0.5
            figsize = (grid.cols * cell_scale + 1, grid.rows * cell_scale + 1)

    fig, ax = plt.subplots(figsize=figsize)

    if grid.is_hierarchical:
        _draw_hierarchical_grid(
            ax,
            grid.indices,
            grid.rows,
            grid.cols,
            cell_coords=grid.cell_coords,
            row_shape=grid.row_shape,
            col_shape=grid.col_shape,
            title=title or str(layout),
            colorize=colorize,
            color_indices=grid.color_indices,
            flatten_hierarchical=False,
            label_hierarchy_levels=label_hierarchy_levels,
            num_colors=num_colors,
        )
    else:
        _draw_grid(
            ax,
            grid.indices,
            title=title or str(layout),
            colorize=colorize,
            color_indices=grid.color_indices,
            num_colors=num_colors,
        )

    return fig


def draw_layout(
    layout,
    filename=None,
    title: Optional[str] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    color_layout: Optional[Layout] = None,
    color_by: Optional[str] = None,
    num_colors: int = 8,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Draw a layout or tensor and save to file.

    Args:
        layout: Layout or Tensor to visualize. When given a Tensor, cells
            display offset-adjusted values and the title includes the base offset.
        filename: Output path (.svg, .png, or .pdf)
        title: Optional title (defaults to layout repr)
        dpi: Resolution for raster formats
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors; if False, use grayscale
        color_layout: Optional layout controlling cell coloring. For displayed
            rank-2 grids, this is evaluated in the same logical coordinate
            space as the layout being drawn, so displayed cell (row, col) is
            colored by color_layout(row_coord, col_coord). Examples:
            - Layout((8,8), (1, 0)): color by row
            - Layout((8,8), (0, 1)): color by column
            - Layout(1, 0): uniform color
            - None: color by cell value (default)
        color_by: Shorthand for common color_layout patterns. Mutually exclusive
            with color_layout. One of:
            - "row": color by logical row (implies colorize=True)
            - "column": color by logical column (implies colorize=True)
            - "offset": color by cell value (default)
        num_colors: Number of colors in palette (default 8)
        flatten_hierarchical: For hierarchical layouts, if True show flat grid with
            offset values. If False, show explicit cell labels:
              - row=... nested row coordinate
              - col=... nested column coordinate
              - offset=... resulting offset
        label_hierarchy_levels: For hierarchical nested views, if True annotate
            axes with each hierarchy level at block/tile granularity. Label
            colors match the corresponding hierarchy boundary lines.
    """
    fig = _build_layout_figure(
        layout,
        title=title,
        figsize=figsize,
        colorize=colorize,
        color_layout=color_layout,
        color_by=color_by,
        num_colors=num_colors,
        flatten_hierarchical=flatten_hierarchical,
        label_hierarchy_levels=label_hierarchy_levels,
    )
    _save_figure(fig, filename, dpi)


def _infer_tv_grid_shape(layout, grid_shape=None, grid_rows=None, grid_cols=None):
    """Infer (rows, cols) for a TV grid from layout cosize.

    Accepts either a (rows, cols) tuple via grid_shape, or separate
    grid_rows/grid_cols ints.  When dimensions are not fully specified,
    falls back to a sqrt-based factorisation of cosize(layout).
    """
    if grid_shape is not None:
        return grid_shape
    if grid_rows is not None and grid_cols is not None:
        return (grid_rows, grid_cols)
    cosize_val = cosize(layout)
    cols = int(np.sqrt(cosize_val))
    while cols > 0 and cosize_val % cols != 0:
        cols -= 1
    rows = cosize_val // cols if cols > 0 else cosize_val
    return (rows, cols)


def _compute_tv_mapping(
    layout,
    grid_cols: Optional[int] = None,
    grid_rows: Optional[int] = None,
    thr_id_layout=None,
    col_major: bool = True,
):
    """Compute the inverse TV mapping: for each (row, col), return (thread, value).

    The offset is decomposed to (row, col) using either:
      col_major=True  (default, CuTe A/C convention): row = offset % rows, col = offset // rows
      col_major=False  (CuTe B display convention):   row = offset // cols, col = offset % cols

    Args:
        layout: The TV layout
        grid_cols: Number of columns in the output grid.
        grid_rows: Number of rows in the output grid.
        thr_id_layout: Optional ThrID layout mapping logical→physical thread IDs.
        col_major: If True, use column-major decomposition (A/C matrices).
                   If False, use row-major (B matrix displayed as K×N).

    Returns:
        dict mapping (row, col) or output_index -> (thread_id, value_id)

    If multiple (thread, value) pairs land on the same output cell, the first
    one encountered wins. This matches CuTe's print_latex* helpers.
    """
    t_shape = mode(layout.shape, 0)
    v_shape = mode(layout.shape, 1)

    num_t = size(t_shape)
    num_v = size(v_shape)

    inv_map = {}
    for flat_t in range(num_t):
        for flat_v in range(num_v):
            t_coord = idx2crd(flat_t, t_shape)
            v_coord = idx2crd(flat_v, v_shape)
            out_idx = layout(t_coord, v_coord)

            phys_t = thr_id_layout(flat_t) if thr_id_layout is not None else flat_t

            if grid_rows is not None and grid_cols is not None:
                if col_major:
                    row = out_idx % grid_rows
                    col = out_idx // grid_rows
                else:
                    row = out_idx // grid_cols
                    col = out_idx % grid_cols
                if not (0 <= row < grid_rows and 0 <= col < grid_cols):
                    raise ValueError(
                        f"TV layout output cell {(row, col)} is out of bounds for "
                        f"grid_shape=({grid_rows}, {grid_cols}); offset={out_idx}"
                    )
                # Store (physical_thread_id, value_id, logical_thread_id)
                key = (row, col)
                if key not in inv_map:
                    inv_map[key] = (phys_t, flat_v, flat_t)
            else:
                if out_idx not in inv_map:
                    inv_map[out_idx] = (phys_t, flat_v, flat_t)

    return inv_map


def _draw_tv_cells(
    ax,
    tv_map: dict,
    rows: int,
    cols: int,
    colors,
    cell_size: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    fontsize: float = 7,
    linewidth: float = 1.0,
):
    """Draw TV cells from a precomputed mapping.

    Shared cell-drawing loop used by _draw_tv_grid, draw_mma_layout, and
    draw_tiled_grid.  Each cell displays "Tx" / "Vy" labels colored by
    logical thread index.
    """
    for i in range(rows):
        for j in range(cols):
            key = (i, j)
            if key in tv_map:
                phys_t, v_idx, logical_t = tv_map[key]
                t_idx = phys_t
            else:
                t_idx, v_idx, logical_t = -1, -1, -1

            color_idx = logical_t % len(colors) if logical_t >= 0 else 0
            facecolor = colors[color_idx]

            x = offset_x + j
            y = offset_y + i

            rect = patches.Rectangle(
                (x, y),
                cell_size,
                cell_size,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=linewidth,
            )
            ax.add_patch(rect)

            text_color = "white" if _is_dark(facecolor) else "black"
            if t_idx >= 0:
                ax.text(
                    x + 0.5,
                    y + 0.3,
                    f"T{t_idx}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                )
                ax.text(
                    x + 0.5,
                    y + 0.7,
                    f"V{v_idx}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                )
            else:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    "?",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                )


def _draw_tv_grid(
    ax,
    layout,
    cell_size: float = 1.0,
    title: Optional[str] = None,
    colorize: bool = False,
    num_colors: Optional[int] = None,
    label_margin: float = 0.5,
    title_position: str = "top",
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    thr_id_layout=None,
    col_major: bool = True,
):
    """Draw a Thread-Value grid showing T and V indices in each cell.

    For a TV layout, this shows the physical arrangement of threads and values.
    Each cell displays "Tx\\nVy" where x is the thread and y is the value that
    owns that memory location.

    Colors are assigned by thread index so all values owned by the same
    thread have the same color.
    """
    r = rank(layout)
    if r != 2:
        raise ValueError(f"TV layout must be rank 2, got rank {r}")

    num_t = size(mode(layout, 0))  # Number of threads

    # Determine grid dimensions
    rows, cols = _infer_tv_grid_shape(layout, grid_rows=grid_rows, grid_cols=grid_cols)

    # Build the inverse mapping with grid dimensions
    tv_map = _compute_tv_mapping(
        layout,
        grid_cols=cols,
        grid_rows=rows,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )

    # Build color palette based on number of threads (cycle through 8 colors like cute-viz)
    n_colors = num_colors if num_colors else min(num_t, 8)
    if colorize:
        colors = _make_rainbow_palette(n_colors)
    else:
        colors = _make_grayscale_palette(n_colors)

    # Set up axes with label margin
    _setup_axes(
        ax,
        (-label_margin, cols + label_margin),
        (-label_margin, rows + label_margin),
        title=title if title_position == "top" else None,
    )

    if title and title_position != "top":
        ax.text(
            cols / 2,
            rows + 0.5,
            title,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Draw each cell
    _draw_tv_cells(ax, tv_map, rows, cols, colors, cell_size)


def _build_tv_figure(
    layout,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    thr_id_layout=None,
    col_major: bool = True,
):
    """Build the TV figure used by draw_tv_layout/show_tv_layout."""
    r = rank(layout)
    if r != 2:
        raise ValueError(f"TV layout must be rank 2, got rank {r}")

    rows, cols = _infer_tv_grid_shape(layout, grid_shape=grid_shape)

    if figsize is None:
        figsize = (cols * 0.6 + 1.5, rows * 0.5 + 1)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_tv_grid(
        ax,
        layout,
        title=title or f"TV: {layout}",
        colorize=colorize,
        num_colors=num_colors,
        grid_rows=rows,
        grid_cols=cols,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )
    return fig


def draw_tv_layout(
    layout,
    filename=None,
    title: Optional[str] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    thr_id_layout=None,
    col_major: bool = True,
):
    """Draw a Thread-Value layout showing T and V indices in each cell.

    For TV layouts where shape is (Threads, Values), this shows each cell
    labeled as "Tx\\nVy" indicating which thread owns which value.

    Args:
        layout: Layout object with shape (T, V) for Thread-Value
        filename: Output path (.svg, .png, or .pdf)
        title: Optional title (defaults to "TV: {layout}")
        dpi: Resolution for raster formats
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors; if False, use grayscale
        num_colors: Override number of colors (defaults to T dimension)
        grid_shape: Optional (rows, cols) for the output grid. If None,
                    inferred from cosize. Required for proper visualization
                    of MMA-style layouts.

    Example:
        tv_layout = Layout((4, 2), (2, 1))  # 4 threads, 2 values each
        draw_tv_layout(tv_layout, "tv_4x2.svg")

        # MMA-style layout with explicit grid shape
        mma_a = Layout(((4,2,2,2), (2,2)), ((32,1,8,128), (16,4)))
        draw_tv_layout(mma_a, "mma_a.svg", grid_shape=(16, 8))
    """
    fig = _build_tv_figure(
        layout,
        title=title,
        figsize=figsize,
        colorize=colorize,
        num_colors=num_colors,
        grid_shape=grid_shape,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )
    _save_figure(fig, filename, dpi)


def _build_mma_figure(
    layout_a,
    layout_b,
    layout_c,
    tile_mnk=None,
    main_title=None,
    colorize=True,
    thr_id_layout=None,
):
    """Build the MMA figure used by draw_mma_layout/show_mma_layout."""
    # Infer M, N, K from tile_mnk or layout dimensions
    if tile_mnk:
        M, N, K = tile_mnk
    else:
        # Infer M, K, N from layout cosizes when tile_mnk is not provided.
        # A is M×K, C is M×N.  Start with M = sqrt(cosize_a) and search
        # downward for a divisor (heuristic: assumes roughly square tiles).
        cosize_a = cosize(layout_a)
        cosize_c = cosize(layout_c)
        M = int(np.sqrt(cosize_a))
        while M > 0 and cosize_a % M != 0:
            M -= 1
        K = cosize_a // M if M > 0 else cosize_a
        N = cosize_c // M if M > 0 else cosize_c

    if colorize:
        mma_colors = _make_rainbow_palette(8)
    else:
        mma_colors = _make_grayscale_palette(8)

    gap = 2.0
    label_margin = 1.5

    total_width = K + gap + N + label_margin
    total_height = K + gap + M + label_margin

    scale = 0.35
    fig_width = total_width * scale + 1.5
    fig_height = total_height * scale + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def draw_tv_matrix(
        layout,
        offset_x,
        offset_y,
        matrix_rows,
        matrix_cols,
        title,
        title_above=True,
        col_major=True,
    ):
        try:
            tv_map = _compute_tv_mapping(
                layout,
                grid_cols=matrix_cols,
                grid_rows=matrix_rows,
                thr_id_layout=thr_id_layout,
                col_major=col_major,
            )
        except ValueError as exc:
            raise ValueError(
                f"{title} layout does not fit within panel shape "
                f"({matrix_rows}, {matrix_cols}): {exc}"
            ) from exc

        if title:
            if title_above:
                ax.text(
                    offset_x + matrix_cols / 2,
                    offset_y - 1.2,
                    title,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.text(
                    offset_x + matrix_cols / 2,
                    offset_y + matrix_rows + 0.6,
                    title,
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

        _draw_tv_cells(
            ax,
            tv_map,
            matrix_rows,
            matrix_cols,
            mma_colors,
            offset_x=offset_x,
            offset_y=offset_y,
            fontsize=6,
            linewidth=0.5,
        )

    b_offset_x = K + gap
    b_offset_y = 0
    a_offset_x = 0
    a_offset_y = K + gap
    c_offset_x = K + gap
    c_offset_y = K + gap

    draw_tv_matrix(
        layout_b,
        b_offset_x,
        b_offset_y,
        K,
        N,
        f"B ({K}×{N})",
        title_above=True,
        col_major=False,
    )
    draw_tv_matrix(layout_a, a_offset_x, a_offset_y, M, K, f"A ({M}×{K})", title_above=False)
    draw_tv_matrix(layout_c, c_offset_x, c_offset_y, M, N, f"C ({M}×{N})", title_above=False)

    for k in range(K):
        ax.text(
            b_offset_x - 0.4,
            k + 0.5,
            str(k),
            ha="right",
            va="center",
            fontsize=7,
            color="dimgray",
        )
    for n in range(N):
        ax.text(
            b_offset_x + n + 0.5,
            -0.4,
            str(n),
            ha="center",
            va="bottom",
            fontsize=7,
            color="dimgray",
        )
    for m in range(M):
        ax.text(
            a_offset_x - 0.4,
            a_offset_y + m + 0.5,
            str(m),
            ha="right",
            va="center",
            fontsize=7,
            color="dimgray",
        )
    for k in range(K):
        ax.text(
            a_offset_x + k + 0.5,
            a_offset_y - 0.4,
            str(k),
            ha="center",
            va="bottom",
            fontsize=7,
            color="dimgray",
        )

    _setup_axes(ax, (-label_margin, total_width), (-label_margin, total_height))

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig


def draw_mma_layout(
    layout_a,
    layout_b,
    layout_c,
    filename=None,
    tile_mnk: Optional[Tuple[int, int, int]] = None,
    main_title: Optional[str] = None,
    dpi: int = 150,
    colorize: bool = True,
    thr_id_layout=None,
):
    """Draw MMA (Matrix Multiply-Accumulate) layout visualization.

    Renders A, B, and C matrices in the standard MMA arrangement:

              B (K×N)
        A (M×K)    C (M×N)

    Where C = A × B for matrix multiplication. B and C share the same
    column alignment (N dimension), while A and C share the same row
    alignment (M dimension).

    This matches the cute-viz MMA visualization style.

    Args:
        layout_a: TV layout for matrix A (M×K)
        layout_b: TV layout for matrix B (K×N, rendered as N×K for visual alignment)
        layout_c: TV layout for matrix C (M×N)
        filename: Output path (.svg, .png, or .pdf)
        tile_mnk: Optional (M, N, K) dimensions. If None, inferred from cosize.
                  If an operand layout does not fit within those logical panel
                  dimensions, a ValueError is raised instead of silently
                  dropping cells.
        main_title: Optional title for the entire figure
        dpi: Resolution for raster formats
        colorize: If True, use rainbow colors by thread ID

    Example:
        # MMA 16×8×8 layouts
        mma_a = Layout(((4, 2, 2), (2, 2)), ((32, 1, 8), (16, 4)))
        mma_b = Layout(((2, 2, 2), (2, 2)), ((1, 16, 4), (8, 2)))
        mma_c = Layout(((4, 2, 2), (2, 2)), ((32, 1, 8), (16, 4)))
        draw_mma_layout(mma_a, mma_b, mma_c, "mma_16x8x8.svg",
                        tile_mnk=(16, 8, 8), main_title="MMA 16×8×8")
    """
    fig = _build_mma_figure(
        layout_a,
        layout_b,
        layout_c,
        tile_mnk=tile_mnk,
        main_title=main_title,
        colorize=colorize,
        thr_id_layout=thr_id_layout,
    )
    _save_figure(fig, filename, dpi)


def _build_tiled_grid_figure(grid: dict, rows: int, cols: int, title: Optional[str] = None):
    """Build the tiled-grid figure used by draw_tiled_grid/show_tiled_grid."""
    colors = _make_rainbow_palette(8)
    font = max(4, min(7, int(60 / max(rows, cols))))
    fig, ax = plt.subplots(figsize=(cols * 0.45 + 1.5, rows * 0.4 + 1.0))
    _setup_axes(ax, (-0.5, cols + 0.5), (-0.5, rows + 0.5), title=title, title_fontsize=9)
    _draw_tv_cells(ax, grid, rows, cols, colors, fontsize=font, linewidth=0.5)
    plt.tight_layout()
    return fig


def draw_tiled_grid(
    grid: dict,
    rows: int,
    cols: int,
    filename=None,
    dpi: int = 150,
    title: Optional[str] = None,
):
    """Draw a tiled MMA grid produced by tile_mma_grid().

    Each cell shows thread (T) and value (V) labels, colored by logical
    thread group.

    Args:
        grid:  dict mapping (row, col) → (phys_thread, value, logical_thread)
        rows:  number of rows in the grid
        cols:  number of columns in the grid
        filename: output path (SVG/PNG/PDF) or None for inline display
        dpi:   output resolution
        title: plot title
    """
    fig = _build_tiled_grid_figure(grid, rows, cols, title=title)
    _save_figure(fig, filename, dpi)


def _build_combined_grid_figure(a_grid, b_grid, c_grid, M, N, K, title=None):
    """Build a combined A/B/C figure from pre-computed grid dicts.

    Arranges three grid-dict panels in the standard MMA layout:

              B (K×N)
        A (M×K)    C (M×N)
    """
    colors = _make_rainbow_palette(8)
    font = max(4, min(7, int(60 / max(M, N, K))))
    gap = 2.0
    label_margin = 1.5

    total_w = K + gap + N + label_margin
    total_h = K + gap + M + label_margin

    scale = 0.35
    fig, ax = plt.subplots(figsize=(total_w * scale + 1.5, total_h * scale + 1.0))

    b_ox, b_oy = K + gap, 0
    a_ox, a_oy = 0, K + gap
    c_ox, c_oy = K + gap, K + gap

    _draw_tv_cells(
        ax,
        b_grid,
        K,
        N,
        colors,
        offset_x=b_ox,
        offset_y=b_oy,
        fontsize=font,
        linewidth=0.5,
    )
    _draw_tv_cells(
        ax,
        a_grid,
        M,
        K,
        colors,
        offset_x=a_ox,
        offset_y=a_oy,
        fontsize=font,
        linewidth=0.5,
    )
    _draw_tv_cells(
        ax,
        c_grid,
        M,
        N,
        colors,
        offset_x=c_ox,
        offset_y=c_oy,
        fontsize=font,
        linewidth=0.5,
    )

    _setup_axes(ax, (-label_margin, total_w), (-label_margin, total_h))

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig


def draw_combined_mma_grid(a_grid, b_grid, c_grid, M, N, K, filename=None, dpi=150, title=None):
    """Draw combined A/B/C grid-dict panels in the standard MMA arrangement.

    This is the grid-dict counterpart of draw_mma_layout.  Use it when
    you have pre-computed ``(row, col) → (phys_thread, value, logical_thread)``
    dicts (e.g. from ``tile_mma_grid``).

    Args:
        a_grid: dict for A panel (M×K)
        b_grid: dict for B panel (K×N)
        c_grid: dict for C panel (M×N)
        M, N, K: panel dimensions
        filename: output path (SVG/PNG/PDF) or None for inline display
        dpi: output resolution
        title: plot title
    """
    fig = _build_combined_grid_figure(a_grid, b_grid, c_grid, M, N, K, title=title)
    _save_figure(fig, filename, dpi)


def _build_copy_figure(
    src_layout,
    dst_layout,
    grid_shape=None,
    title=None,
    colorize=True,
    thr_id_layout=None,
    col_major=True,
):
    """Build a side-by-side src/dst copy layout figure.

    Matches CUTLASS print_latex_copy: two TV grids with the same coloring
    so thread data movement is visually traceable.

    Layout:
        Src (rows×cols)    Dst (rows×cols)
    """
    r_s, r_d = rank(src_layout), rank(dst_layout)
    if r_s != 2:
        raise ValueError(f"src_layout must be rank 2, got rank {r_s}")
    if r_d != 2:
        raise ValueError(f"dst_layout must be rank 2, got rank {r_d}")

    rows, cols = _infer_tv_grid_shape(src_layout, grid_shape=grid_shape)

    num_t = size(mode(src_layout, 0))
    n_colors = min(num_t, 8)
    if colorize:
        colors = _make_rainbow_palette(n_colors)
    else:
        colors = _make_grayscale_palette(n_colors)

    gap = 3.0
    label_margin = 0.5
    total_width = 2 * cols + gap + 2 * label_margin
    total_height = rows + 2 * label_margin

    scale = 0.5
    fig_width = total_width * scale + 1.5
    fig_height = total_height * scale + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Source grid (left)
    src_map = _compute_tv_mapping(
        src_layout,
        grid_cols=cols,
        grid_rows=rows,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )
    _draw_tv_cells(
        ax,
        src_map,
        rows,
        cols,
        colors,
        offset_x=0,
        offset_y=0,
        fontsize=6,
        linewidth=0.5,
    )
    ax.text(cols / 2, -0.6, "Src", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Destination grid (right)
    dst_ox = cols + gap
    dst_map = _compute_tv_mapping(
        dst_layout,
        grid_cols=cols,
        grid_rows=rows,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )
    _draw_tv_cells(
        ax,
        dst_map,
        rows,
        cols,
        colors,
        offset_x=dst_ox,
        offset_y=0,
        fontsize=6,
        linewidth=0.5,
    )
    ax.text(
        dst_ox + cols / 2,
        -0.6,
        "Dst",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    _setup_axes(
        ax,
        (-label_margin, total_width - label_margin),
        (-label_margin, total_height - label_margin),
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig


def draw_copy_layout(
    src_layout,
    dst_layout,
    filename=None,
    grid_shape=None,
    title=None,
    dpi=150,
    colorize=True,
    thr_id_layout=None,
    col_major=True,
):
    """Draw a copy layout showing src and dst TV grids side by side.

    Matches CUTLASS print_latex_copy: same thread coloring on both panels
    so data movement is visually traceable.

    Args:
        src_layout: TV layout for source (thread, value) -> offset
        dst_layout: TV layout for destination (thread, value) -> offset
        filename: Output path (.svg, .png, or .pdf)
        grid_shape: Optional (rows, cols) for the output grids. If None,
                    inferred from cosize of src_layout.
        title: Optional title for the entire figure
        dpi: Resolution for raster formats
        colorize: If True, use rainbow colors; if False, use grayscale
        thr_id_layout: Optional layout for thread ID mapping
        col_major: If True (default), use column-major decomposition
                    (CuTe A/C convention: row = offset % rows).
                    If False, use row-major (CuTe B convention:
                    row = offset // cols).

    Example:
        # LDMATRIX x4 non-transpose (fp16)
        src = Layout((32, 8), (8, 1))     # smem src
        dst = Layout((32, (2, 4)), (2, (1, 64)))  # register dst
        draw_copy_layout(src, dst, "ldsm_x4.svg",
                         grid_shape=(16, 16), title="SM75 LDMATRIX x4")
    """
    fig = _build_copy_figure(
        src_layout,
        dst_layout,
        grid_shape=grid_shape,
        title=title,
        colorize=colorize,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )
    _save_figure(fig, filename, dpi)


def _swizzle_figsize(linear_idx, swizzle_idx, rows, cols):
    """Compute figure size for a two-panel swizzle comparison.

    Uses the larger of the "comfortable" cell width (0.5in, matching the
    original 8×8 sizing) and the minimum width needed to render the widest
    cell value at 7pt without overlap.
    """
    max_val = max(int(np.max(linear_idx)), int(np.max(swizzle_idx)))
    n_digits = max(len(str(max_val)), 1)
    # Minimum cell width so 7pt text fits: digit ≈ 0.6em
    min_cell_w = n_digits * 0.6 * 7 / 72
    cell_w = max(min_cell_w, 0.5)
    return (cols * cell_w * 2 + 3, rows * 0.5 + 1.5)


def _build_swizzle_figure(
    base_layout,
    swizzle,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: int = 8,
):
    """Build the swizzle comparison figure used by draw_swizzle/show_swizzle."""
    sw_layout = compose(swizzle, base_layout)

    linear_idx = _get_indices_2d(base_layout)
    swizzle_idx = _get_indices_2d(sw_layout)

    rows, cols = linear_idx.shape

    if swizzle.base > 0 and cols > (1 << swizzle.base):
        blocks_per_row = cols // (1 << swizzle.base)
        effective_colors = blocks_per_row
        bit_shift = swizzle.base
        if figsize is None:
            figsize = _swizzle_figsize(linear_idx, swizzle_idx, rows, cols)
    elif swizzle.base == 0:
        effective_colors = num_colors
        bit_shift = 0
        if figsize is None:
            figsize = _swizzle_figsize(linear_idx, swizzle_idx, rows, cols)
    else:
        bit_shift = swizzle.base
        distinct_groups = len(set(int(v) >> bit_shift for v in linear_idx.flat))
        effective_colors = max(num_colors, distinct_groups)
        if figsize is None:
            figsize = _swizzle_figsize(linear_idx, swizzle_idx, rows, cols)

    def _swizzle_color_indices(idx_array):
        return np.vectorize(lambda v: (int(v) >> bit_shift) % effective_colors)(idx_array)

    linear_ci = _swizzle_color_indices(linear_idx)
    swizzle_ci = _swizzle_color_indices(swizzle_idx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _draw_grid(
        ax1,
        linear_idx,
        title=f"Linear: {base_layout}",
        colorize=colorize,
        color_indices=linear_ci,
        num_colors=effective_colors,
        label_color="gray",
        label_fontsize=7,
    )
    _draw_grid(
        ax2,
        swizzle_idx,
        title=f"Swizzled: {swizzle}",
        colorize=colorize,
        color_indices=swizzle_ci,
        num_colors=effective_colors,
        label_color="gray",
        label_fontsize=7,
    )

    plt.tight_layout()
    return fig


def draw_swizzle(
    base_layout,
    swizzle,
    filename=None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: int = 8,
):
    """Draw side-by-side comparison of linear vs swizzled layout.

    For swizzles with base=0 (affecting low bits), colors by value % num_colors
    to show how values within rows get permuted at element granularity.

    For swizzles with base>0 (affecting higher bits), shows a block-level view
    where each cell represents a block of 2^base consecutive elements.
    Blocks are colored by (block_value >> base) % blocks_per_row, showing
    how the swizzle permutes blocks within each row — the same visual pattern
    as the base=0 case but at block granularity.

    Args:
        base_layout: Base Layout object
        swizzle: Swizzle object to apply
        filename: Output path (.svg, .png, or .pdf)
        dpi: Resolution for raster formats
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors (makes swizzle movement clearer)
        num_colors: Number of colors in palette
    """
    fig = _build_swizzle_figure(
        base_layout, swizzle, figsize=figsize, colorize=colorize, num_colors=num_colors
    )
    _save_figure(fig, filename, dpi)


def _expand_hier_slice(spec, shape):
    """Expand a hierarchical slice spec into all matching flat coordinates.

    A slice spec mirrors the shape structure but may contain None at any level,
    meaning "all values in this sub-dimension." Concrete integers fix that
    position. Returns an iterator of fully-specified hierarchical coordinates.

    Examples (shape=(3,2), ((2,3),2)):
        spec=(2, None)         -> all coords with mode-0 fixed to 2
        spec=(None, 5)         -> all coords with mode-1 fixed to 5
        spec=(2, ((0,None),None)) -> mode-0=2, mode-1 inner-0-0=0, rest free
    """
    if spec is None:
        # Wildcard: enumerate all valid coordinates for this sub-shape
        if is_tuple(shape):
            sub_iters = [_expand_hier_slice(None, s) for s in shape]
            for combo in itertools.product(*sub_iters):
                yield combo
        else:
            # scalar shape: enumerate 0..shape-1
            for i in range(shape):
                yield i
    elif is_tuple(spec):
        if is_tuple(shape):
            if len(spec) != len(shape):
                raise ValueError(
                    f"Rank mismatch: spec has {len(spec)} elements, shape has {len(shape)}"
                )
            sub_iters = [_expand_hier_slice(s, sh) for s, sh in zip(spec, shape)]
            for combo in itertools.product(*sub_iters):
                yield combo
        else:
            raise TypeError(f"Tuple spec {spec} for scalar shape {shape}")
    else:
        # Concrete integer: single value
        yield spec


def _is_flat_slice_component(spec) -> bool:
    """Return True for top-level flat slice components."""
    return spec is None or isinstance(spec, (int, slice))


def _match_flat_slice_component(display_idx: int, spec, extent: int) -> bool:
    """Match a displayed row/column index against a flat slice component."""
    if spec is None:
        return True
    if isinstance(spec, int):
        return display_idx == spec
    return display_idx in range(*spec.indices(extent))


def _match_nested_slice_component(coord, spec, shape) -> bool:
    """Match a hierarchical coordinate against a nested slice spec."""
    if spec is None:
        return True
    if isinstance(spec, int):
        return coord == spec
    if isinstance(spec, slice):
        if is_tuple(coord):
            raise TypeError(f"Slice spec {spec} requires a scalar coordinate, got {coord!r}")
        return coord in range(*spec.indices(shape))
    if is_tuple(spec):
        if not is_tuple(coord) or not is_tuple(shape):
            raise TypeError(f"Tuple spec {spec!r} is incompatible with coordinate {coord!r}")
        if len(spec) != len(coord) or len(spec) != len(shape):
            raise ValueError(
                f"Tuple spec length {len(spec)} does not match coordinate/shape lengths "
                f"{len(coord)} / {len(shape)}"
            )
        return all(_match_nested_slice_component(c, s, sh) for c, s, sh in zip(coord, spec, shape))
    raise TypeError(f"Unsupported slice component {spec!r}")


def _get_slice_highlight_mask_2d(layout, slice_spec) -> np.ndarray:
    """Return a displayed-cell highlight mask for draw_slice()."""
    indices = _get_indices_2d(layout)
    rows, cols = indices.shape
    mask = np.zeros((rows, cols), dtype=bool)
    r = rank(layout)

    if r == 2:
        row_shape = mode(layout.shape, 0)
        col_shape = mode(layout.shape, 1)
    else:
        row_shape = None
        col_shape = None

    if isinstance(slice_spec, int):
        target_coord = idx2crd(slice_spec, layout.shape)
        if r == 2:
            target_row, target_col = target_coord
            for i in range(rows):
                row_coord = idx2crd(i, row_shape)
                if row_coord != target_row:
                    continue
                for j in range(cols):
                    col_coord = idx2crd(j, col_shape)
                    if col_coord == target_col:
                        mask[i, j] = True
        else:
            for j in range(cols):
                coord = idx2crd(j, layout.shape)
                mask[0, j] = coord == target_coord
        return mask

    if isinstance(slice_spec, tuple) and r == 2:
        row_spec, col_spec = slice_spec
        row_flat = _is_flat_slice_component(row_spec)
        col_flat = _is_flat_slice_component(col_spec)

        for i in range(rows):
            row_coord = idx2crd(i, row_shape)
            row_match = (
                _match_flat_slice_component(i, row_spec, rows)
                if row_flat
                else _match_nested_slice_component(row_coord, row_spec, row_shape)
            )
            if not row_match:
                continue

            for j in range(cols):
                col_coord = idx2crd(j, col_shape)
                col_match = (
                    _match_flat_slice_component(j, col_spec, cols)
                    if col_flat
                    else _match_nested_slice_component(col_coord, col_spec, col_shape)
                )
                if col_match:
                    mask[i, j] = True

    elif isinstance(slice_spec, tuple) and r < 2:
        if len(slice_spec) != 1:
            raise ValueError(
                f"Rank-{r} layout requires a 1-element tuple slice_spec, got {len(slice_spec)}"
            )
        (col_spec,) = slice_spec
        col_flat = _is_flat_slice_component(col_spec)
        for j in range(cols):
            col_coord = idx2crd(j, layout.shape)
            mask[0, j] = (
                _match_flat_slice_component(j, col_spec, cols)
                if col_flat
                else _match_nested_slice_component(col_coord, col_spec, layout.shape)
            )

    return mask


def _build_slice_figure(
    layout,
    slice_spec,
    title=None,
    figsize=None,
    colorize=False,
    color_layout=None,
    num_colors=8,
):
    """Build the slice figure used by draw_slice/show_slice."""
    grid = _prepare_offset_grid(layout, color_layout=color_layout, slice_spec=slice_spec)

    if figsize is None:
        figsize = (grid.cols * 0.5 + 1, grid.rows * 0.5 + 1)

    if title is None:
        title = f"{layout}[{slice_spec}]"

    fig, ax = plt.subplots(figsize=figsize)
    _draw_grid(
        ax,
        grid.indices,
        highlight_mask=grid.highlight_mask,
        title=title,
        colorize=colorize,
        color_indices=grid.color_indices,
        num_colors=num_colors,
    )
    return fig


def draw_slice(
    layout,
    slice_spec,
    filename=None,
    title: Optional[str] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    color_layout: Optional[Layout] = None,
    num_colors: int = 8,
):
    """Draw layout with sliced elements highlighted.

    Args:
        layout: Layout object
        slice_spec: Slice specification, e.g.:
            - (2, None) for row 2
            - (None, 3) for column 3
            - (2, 5) for single element
            - (slice(1,3), slice(2,6)) for rectangular region
        filename: Output path (.svg, .png, or .pdf)
        title: Optional title
        dpi: Resolution for raster formats
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors for background cells
        color_layout: Optional layout controlling background-cell coloring in
            the same logical coordinate space as `layout` (None = color by value)
        num_colors: Number of colors in palette
    """
    fig = _build_slice_figure(
        layout,
        slice_spec,
        title=title,
        figsize=figsize,
        colorize=colorize,
        color_layout=color_layout,
        num_colors=num_colors,
    )
    _save_figure(fig, filename, dpi)


def show_layout(
    layout,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    color_layout: Optional[Layout] = None,
    color_by: Optional[str] = None,
    num_colors: int = 8,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Display a layout or tensor inline (for Jupyter notebooks).

    Args:
        layout: Layout or Tensor to visualize. When given a Tensor, cells
            display offset-adjusted values and the title includes the base offset.
        title: Optional title
        figsize: Figure size in inches
        colorize: If True, use rainbow colors for distinct cells
        color_layout: Optional layout controlling cell coloring in the same
            logical coordinate space as `layout` (None = color by value)
        color_by: Shorthand for common color_layout patterns ("row", "column",
            or "offset"). Mutually exclusive with color_layout.
        num_colors: Number of colors in palette
        flatten_hierarchical: For hierarchical layouts, if True show flat grid with
            offset values. If False, show explicit cell labels.
        label_hierarchy_levels: For hierarchical nested views, if True annotate
            axes with each hierarchy level at block/tile granularity.

    Returns:
        matplotlib Figure
    """
    return _build_layout_figure(
        layout,
        title=title,
        figsize=figsize,
        colorize=colorize,
        color_layout=color_layout,
        color_by=color_by,
        num_colors=num_colors,
        flatten_hierarchical=flatten_hierarchical,
        label_hierarchy_levels=label_hierarchy_levels,
    )


def show_swizzle(
    base_layout,
    swizzle,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: int = 8,
):
    """Display swizzle comparison inline (for Jupyter notebooks).

    Colors by the bits that the swizzle affects: (value >> swizzle.base) % num_colors.

    Args:
        base_layout: Base Layout object
        swizzle: Swizzle object
        figsize: Figure size in inches
        colorize: If True, use rainbow colors (makes swizzle movement clearer)
        num_colors: Number of colors in palette

    Returns:
        matplotlib Figure
    """
    return _build_swizzle_figure(
        base_layout, swizzle, figsize=figsize, colorize=colorize, num_colors=num_colors
    )


def show_copy_layout(
    src_layout,
    dst_layout,
    grid_shape=None,
    title=None,
    colorize=True,
    thr_id_layout=None,
    col_major=True,
):
    """Display a copy layout inline (for Jupyter notebooks).

    Args:
        src_layout: TV layout for source
        dst_layout: TV layout for destination
        grid_shape: Optional (rows, cols) for the output grids
        title: Optional title
        colorize: If True, use rainbow colors
        thr_id_layout: Optional layout for thread ID mapping
        col_major: If True (default), use column-major decomposition
                    (CuTe A/C convention: row = offset % rows).
                    If False, use row-major (CuTe B convention:
                    row = offset // cols).

    Returns:
        matplotlib Figure
    """
    return _build_copy_figure(
        src_layout,
        dst_layout,
        grid_shape=grid_shape,
        title=title,
        colorize=colorize,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )


def draw_copy_atom(
    atom,
    element_bits: int = 16,
    filename=None,
    grid_shape=None,
    title=None,
    dpi: int = 150,
    colorize: bool = True,
    col_major: bool = True,
):
    """Draw a CopyAtom by converting bit layouts to element coordinates.

    Convenience wrapper around draw_copy_layout that handles the upcast
    from bit coordinates to element coordinates automatically.

    Args:
        atom: CopyAtom with src_layout_bits and dst_layout_bits in bit coords
        element_bits: Bit width of the data type (default 16 for fp16)
        filename: Output path (.svg, .png, or .pdf). None for Jupyter inline.
        grid_shape: Optional (rows, cols) for the output grids
        title: Optional title (defaults to atom.name)
        dpi: Resolution for raster formats
        colorize: If True, use rainbow colors
        col_major: If True (default), use column-major grid decomposition
    """
    src = upcast(atom.src_layout_bits, element_bits)
    dst = upcast(atom.dst_layout_bits, element_bits)
    draw_copy_layout(
        src,
        dst,
        filename=filename,
        grid_shape=grid_shape,
        title=title or f"{atom.name}  ({atom.ptx})",
        dpi=dpi,
        colorize=colorize,
        thr_id_layout=atom.thr_id,
        col_major=col_major,
    )


def show_copy_atom(
    atom,
    element_bits: int = 16,
    grid_shape=None,
    title=None,
    colorize: bool = True,
    col_major: bool = True,
):
    """Display a CopyAtom inline (for Jupyter notebooks).

    Convenience wrapper around show_copy_layout that handles the upcast
    from bit coordinates to element coordinates automatically.

    Args:
        atom: CopyAtom with src_layout_bits and dst_layout_bits in bit coords
        element_bits: Bit width of the data type (default 16 for fp16)
        grid_shape: Optional (rows, cols) for the output grids
        title: Optional title (defaults to atom.name)
        colorize: If True, use rainbow colors
        col_major: If True (default), use column-major grid decomposition

    Returns:
        matplotlib Figure
    """
    src = upcast(atom.src_layout_bits, element_bits)
    dst = upcast(atom.dst_layout_bits, element_bits)
    return show_copy_layout(
        src,
        dst,
        grid_shape=grid_shape,
        title=title or f"{atom.name}  ({atom.ptx})",
        colorize=colorize,
        thr_id_layout=atom.thr_id,
        col_major=col_major,
    )


def show_tv_layout(
    layout,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    num_colors: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    thr_id_layout=None,
    col_major: bool = True,
):
    """Display a TV layout inline (for Jupyter notebooks).

    Args:
        layout: Layout object with shape (T, V) for Thread-Value
        title: Optional title (defaults to "TV: {layout}")
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors; if False, use grayscale
        num_colors: Override number of colors (defaults to T dimension)
        grid_shape: Optional (rows, cols) for the output grid
        thr_id_layout: Optional layout for thread ID mapping
        col_major: If True (default), use column-major decomposition
                    (CuTe A/C convention: row = offset % rows).
                    If False, use row-major (CuTe B convention:
                    row = offset // cols).

    Returns:
        matplotlib Figure
    """
    return _build_tv_figure(
        layout,
        title=title,
        figsize=figsize,
        colorize=colorize,
        num_colors=num_colors,
        grid_shape=grid_shape,
        thr_id_layout=thr_id_layout,
        col_major=col_major,
    )


def show_mma_layout(
    layout_a,
    layout_b,
    layout_c,
    tile_mnk=None,
    main_title=None,
    colorize=True,
    thr_id_layout=None,
):
    """Display an MMA layout inline (for Jupyter notebooks).

    Args:
        layout_a: TV layout for matrix A (M×K)
        layout_b: TV layout for matrix B (K×N)
        layout_c: TV layout for matrix C (M×N)
        tile_mnk: Optional (M, N, K) dimensions
        main_title: Optional title for the entire figure
        colorize: If True, use rainbow colors by thread ID
        thr_id_layout: Optional layout for thread ID mapping

    Returns:
        matplotlib Figure
    """
    return _build_mma_figure(
        layout_a,
        layout_b,
        layout_c,
        tile_mnk=tile_mnk,
        main_title=main_title,
        colorize=colorize,
        thr_id_layout=thr_id_layout,
    )


def show_tiled_grid(grid: dict, rows: int, cols: int, title: Optional[str] = None):
    """Display a tiled MMA grid inline (for Jupyter notebooks).

    Args:
        grid:  dict mapping (row, col) → (phys_thread, value, logical_thread)
        rows:  number of rows in the grid
        cols:  number of columns in the grid
        title: plot title

    Returns:
        matplotlib Figure
    """
    return _build_tiled_grid_figure(grid, rows, cols, title=title)


def show_combined_mma_grid(a_grid, b_grid, c_grid, M, N, K, title=None):
    """Display combined A/B/C grid-dict panels inline (for Jupyter).

    Args:
        a_grid: dict for A panel (M×K)
        b_grid: dict for B panel (K×N)
        c_grid: dict for C panel (M×N)
        M, N, K: panel dimensions
        title: plot title

    Returns:
        matplotlib Figure
    """
    return _build_combined_grid_figure(a_grid, b_grid, c_grid, M, N, K, title=title)


def show_slice(
    layout,
    slice_spec,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colorize: bool = False,
    color_layout: Optional[Layout] = None,
    num_colors: int = 8,
):
    """Display a layout slice inline (for Jupyter notebooks).

    Args:
        layout: Layout object
        slice_spec: Slice specification (see draw_slice for details)
        title: Optional title
        figsize: Figure size in inches (auto-calculated if None)
        colorize: If True, use rainbow colors for background cells
        color_layout: Optional layout controlling background-cell coloring
        num_colors: Number of colors in palette

    Returns:
        matplotlib Figure
    """
    return _build_slice_figure(
        layout,
        slice_spec,
        title=title,
        figsize=figsize,
        colorize=colorize,
        color_layout=color_layout,
        num_colors=num_colors,
    )


def show_composite(
    panels: list,
    arrangement: str = "horizontal",
    titles: Optional[list] = None,
    main_title: Optional[str] = None,
    panel_size: Tuple[float, float] = (4, 4),
    colorize: bool = False,
    tv_mode: bool = False,
    flatten_hierarchical: bool = True,
    label_hierarchy_levels: bool = False,
):
    """Display a composite figure inline (for Jupyter notebooks).

    Args:
        panels: List of Layout objects or (Layout, options_dict) tuples.
        arrangement: "horizontal", "vertical", or "grid:RxC"
        titles: Optional list of titles for each panel
        main_title: Optional title for the entire figure
        panel_size: Size of each panel in inches (width, height)
        colorize: Default colorize setting for all panels
        tv_mode: If True, render panels as TV layouts with T/V labels
        flatten_hierarchical: Default for all panels. If False, show explicit
            nested coordinate labels for hierarchical layouts
        label_hierarchy_levels: Default for all panels. If True, annotate axes
            with hierarchy level labels

    Returns:
        matplotlib Figure
    """
    return _build_composite_figure(
        panels,
        arrangement=arrangement,
        titles=titles,
        main_title=main_title,
        panel_size=panel_size,
        colorize=colorize,
        tv_mode=tv_mode,
        flatten_hierarchical=flatten_hierarchical,
        label_hierarchy_levels=label_hierarchy_levels,
    )


# =============================================================================
# Demo
# =============================================================================


def demo(output_dir: str = "."):
    """Generate example visualizations in all formats."""
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Layout Visualization Demo")
    print("=" * 60)

    # Basic layouts - all formats
    layout_8x8 = Layout((8, 8), (8, 1))
    draw_layout(layout_8x8, output / "row_major.svg")
    draw_layout(layout_8x8, output / "row_major.png")
    draw_layout(layout_8x8, output / "row_major.pdf")
    print("✓ Row-major 8x8 (SVG, PNG, PDF)")

    layout_col = Layout((8, 8), (1, 8))
    draw_layout(layout_col, output / "col_major.svg")
    print("✓ Column-major 8x8")

    # Swizzle comparisons
    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), output / "swizzle_303.svg")
    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), output / "swizzle_303.png")
    print("✓ Swizzle(3,0,3) comparison (SVG, PNG)")

    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(2, 0, 3), output / "swizzle_203.svg")
    print("✓ Swizzle(2,0,3) comparison")

    draw_swizzle(Layout((4, 8), (8, 1)), Swizzle(2, 1, 3), output / "swizzle_213.svg")
    print("✓ Swizzle(2,1,3) on 4x8")

    # Slice highlights
    layout_4x8 = Layout((4, 8), (8, 1))
    draw_slice(layout_4x8, (2, None), output / "slice_row.svg")
    draw_slice(layout_4x8, (2, None), output / "slice_row.png")
    print("✓ Slice row 2 (SVG, PNG)")

    draw_slice(layout_4x8, (None, 5), output / "slice_col.svg")
    print("✓ Slice column 5")

    draw_slice(layout_4x8, (2, 5), output / "slice_single.svg")
    print("✓ Slice single element (2, 5)")

    draw_slice(layout_4x8, (slice(1, 3), slice(2, 6)), output / "slice_rect.svg")
    print("✓ Slice rectangular region")

    # Hierarchical layout
    hier = Layout(((2, 4), 8), ((1, 2), 8))
    draw_layout(hier, output / "hierarchical.svg")
    draw_layout(hier, output / "hierarchical.png")
    print("✓ Hierarchical layout (SVG, PNG)")

    # Logical divide
    divided = Layout((4, 4), (1, 4))  # Result of logical_divide
    draw_layout(divided, output / "divide_result.svg")
    print("✓ Logical divide result")

    # Different sizes
    draw_layout(Layout((4, 16), (16, 1)), output / "wide.svg")
    draw_layout(Layout((16, 4), (4, 1)), output / "tall.svg")
    print("✓ Wide and tall layouts")

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print("  Formats: .svg (vector), .png (raster), .pdf (print)")


if __name__ == "__main__":
    import sys

    demo(sys.argv[1] if len(sys.argv) > 1 else "./viz_output")
