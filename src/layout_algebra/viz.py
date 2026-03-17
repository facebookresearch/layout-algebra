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

"""Layout visualization with PNG/SVG/PDF output.

Visualize layouts, swizzled layouts, and tensor slices in cute-viz style.
Supports multiple output formats: SVG, PNG, PDF.

Usage:
    from layout_algebra.viz import draw_layout, draw_swizzle, draw_slice
    from layout_algebra import Layout, Swizzle, compose

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
    from layout_algebra.viz import show_layout
    show_layout(Layout((8, 8), (8, 1)))

Requirements:
    pip install matplotlib numpy
"""

from pathlib import Path
from typing import Optional, Set, Tuple

import matplotlib
# Use Agg (non-interactive) backend unless already set (e.g. by Jupyter)
if matplotlib.get_backend() == 'agg' or not hasattr(matplotlib, '_called_from_jupyter'):
    try:
        # Check if we're in an IPython/Jupyter environment
        get_ipython()  # noqa: F821
    except NameError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np

from .layouts import *


# =============================================================================
# Color palettes and utilities
# =============================================================================

def _make_grayscale_palette(n: int) -> list:
    """Generate n grayscale colors from white to dark gray."""
    # Range from white (255) to dark gray (~80)
    colors = []
    for i in range(n):
        # Interpolate from 255 (white) to 80 (dark gray)
        gray = int(255 - i * 175 / max(n-1, 1))
        colors.append(f'#{gray:02X}{gray:02X}{gray:02X}')
    return colors

def _make_rainbow_palette(n: int) -> list:
    """Generate n distinct rainbow colors.

    Uses 8 base colors to match cute-viz palette for n <= 8.
    For n > 8, generates distinct pastel colors via HSV interpolation.
    """
    # 8 base colors matching cute-viz pastel palette
    base_colors = [
        '#AFAFFF',  # Light blue
        '#AFFFAF',  # Light green
        '#FFFFAF',  # Light yellow
        '#FFAFAF',  # Light red
        '#D2D2FF',  # Lighter blue
        '#D2FFD2',  # Lighter green
        '#FFFFD2',  # Lighter yellow
        '#FFD2D2',  # Lighter red
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    # Generate truly distinct pastel colors via HSV
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.35, 0.95)
        colors.append(f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}')
    return colors

# Default palettes (8 shades/colors)
GRAYSCALE_COLORS = _make_grayscale_palette(8)
RAINBOW_COLORS = _make_rainbow_palette(8)

# Dark colors that need white text (by hex color)
# With cute-viz pastel palette, only the darker grayscale colors need white text
DARK_COLORS = {'#505050', '#3D3D3D', '#2B2B2B', '#696969', '#504B4B'}

HIGHLIGHT_COLOR = '#FFDC96'  # Orange for highlighted cells
HIGHLIGHT_EDGE = '#FF0000'   # Red border
HIERARCHY_LEVEL_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
]


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

def _get_indices_2d(layout) -> np.ndarray:
    """Extract offset indices from layout as a displayed 2D grid.

    For ordinary rank-2 layouts, displayed cell (row, col) corresponds to the
    logical coordinate (row, col), with hierarchical sub-modes flattened within
    each top-level mode. This keeps visualization semantics aligned with the
    usual matrix interpretation used throughout the docs and examples.
    """
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
                indices[i, j] = layout(row_coord, col_coord)
    else:
        for i in range(total):
            coord = idx2crd(i, layout.shape)
            indices[0, i] = layout(coord)

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

    raise ValueError(
        f"Unsupported color_layout rank {color_rank} for layout rank {layout_rank}"
    )


# =============================================================================
# Core drawing functions
# =============================================================================

def _draw_grid(ax, indices: np.ndarray,
               highlight: Optional[Set[int]] = None,
               cell_size: float = 1.0,
               show_labels: bool = True,
               title: Optional[str] = None,
               colorize: bool = False,
               color_layout: Optional[Layout] = None,
               color_indices: Optional[np.ndarray] = None,
               num_shades: int = 8):
    """Draw a grid of cells with indices on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        indices: 2D array of index values
        highlight: Set of indices to highlight
        cell_size: Size of each cell
        show_labels: Whether to show row/column labels
        title: Optional title for the plot
        colorize: If True, use rainbow colors; if False, use grayscale
        color_layout: Deprecated at this level; callers should usually pass
            precomputed `color_indices` aligned with the displayed grid.
        color_indices: 2D array of per-cell color indices aligned with `indices`.
        num_shades: Number of colors/shades in palette (default 8)
    """
    rows, cols = indices.shape

    # Build the appropriate palette
    if colorize:
        colors = _make_rainbow_palette(num_shades)
    else:
        colors = _make_grayscale_palette(num_shades)

    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])

            # Determine colors
            is_hl = highlight and idx in highlight
            if is_hl:
                facecolor = HIGHLIGHT_COLOR
                edgecolor = HIGHLIGHT_EDGE
                linewidth = 2
            else:
                # Determine color index
                if color_indices is not None:
                    color_idx = int(color_indices[i, j]) % len(colors)
                elif color_layout is None:
                    # Default: color by cell value
                    color_idx = idx % len(colors)
                else:
                    # Backward-compatible fallback for direct display-coordinate use
                    result = color_layout(i, j)
                    color_idx = _color_result_to_index(result) % len(colors)

                facecolor = colors[color_idx]
                edgecolor = 'black'
                linewidth = 1

            # Draw cell
            rect = patches.Rectangle(
                (j, i), cell_size, cell_size,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth
            )
            ax.add_patch(rect)

            # Draw index text (use white text on dark colors for readability)
            text_color = 'white' if _is_dark(facecolor) else 'black'
            ax.text(j + 0.5, i + 0.5, str(idx),
                    ha='center', va='center', fontsize=8, color=text_color)

    if show_labels:
        # Row labels (left)
        for i in range(rows):
            ax.text(-0.3, i + 0.5, str(i), ha='center', va='center',
                    fontsize=8, color='blue')

        # Column labels (top)
        for j in range(cols):
            ax.text(j + 0.5, -0.3, str(j), ha='center', va='center',
                    fontsize=8, color='blue')


def _save_figure(fig, filename, dpi: int = 150):
    """Save figure to file, or display inline in Jupyter if filename is None."""
    if filename is None:
        # Inline display in Jupyter notebook
        try:
            from IPython.display import display, Image
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            display(Image(data=buf.getvalue()))
        except ImportError:
            plt.show()
        return

    path = Path(filename)
    fmt = path.suffix.lower().lstrip('.')

    if fmt == 'svg':
        fig.savefig(filename, format='svg', bbox_inches='tight')
    elif fmt == 'pdf':
        fig.savefig(filename, format='pdf', bbox_inches='tight')
    elif fmt in ('png', 'jpg', 'jpeg'):
        fig.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(filename, format='png', dpi=dpi, bbox_inches='tight')

    plt.close(fig)


# =============================================================================
# Composite Figure API
# =============================================================================

def draw_composite(panels: list, filename: str,
                   arrangement: str = "horizontal",
                   titles: Optional[list] = None,
                   main_title: Optional[str] = None,
                   dpi: int = 150,
                   panel_size: Tuple[float, float] = (4, 4),
                   colorize: bool = False,
                   tv_mode: bool = False):
    """Draw multiple layouts in a single composite figure.

    This allows composing sub-images into bigger figures, useful for:
    - Copy atom visualizations (source and destination side by side)
    - Before/after comparisons
    - Multiple related layouts

    For MMA visualizations, use the dedicated draw_mma_layout() function instead.

    Args:
        panels: List of Layout objects or (Layout, options_dict) tuples.
                Options can include: colorize, color_layout, tv_mode, etc.
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

    Example:
        # Side-by-side comparison
        draw_composite(
            [linear_layout, swizzled_layout],
            "comparison.svg",
            arrangement="horizontal",
            titles=["Linear", "Swizzled"]
        )
    """
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

        # Merge with defaults
        panel_colorize = opts.get('colorize', colorize)
        panel_tv_mode = opts.get('tv_mode', tv_mode)
        color_layout = opts.get('color_layout', None)
        num_shades = opts.get('num_shades', 8)

        # Get title
        title = titles[idx] if titles and idx < len(titles) else None

        # Draw the panel
        if panel_tv_mode:
            _draw_tv_grid(ax, layout, title=title,
                          colorize=panel_colorize, num_threads=num_shades)
        else:
            indices = _get_indices_2d(layout)
            color_indices = _get_color_indices_2d(layout, color_layout)
            _draw_grid(ax, indices, title=title,
                       colorize=panel_colorize, color_layout=color_layout,
                       color_indices=color_indices,
                       num_shades=num_shades)

    # Hide unused axes
    for idx in range(len(panels), len(axes)):
        axes[idx].axis('off')

    if main_title:
        fig.suptitle(main_title, fontsize=12, fontweight='bold')

    plt.tight_layout()
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


def _hierarchy_level_color(level: int) -> str:
    """Color for a hierarchy level.

    Level 0 is the fastest-varying coordinate and has no corresponding tile
    boundary, so keep it neutral. Coarser levels map to the hierarchy palette.
    """
    if level == 0:
        return '#333333'
    return HIERARCHY_LEVEL_COLORS[(level - 1) % len(HIERARCHY_LEVEL_COLORS)]


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


def _draw_colored_coord_line(ax, x: float, y: float, prefix: str, coord,
                             base_color: str, fontsize: float,
                             use_level_colors: bool):
    """Draw row=/col= line with per-coordinate level colors.

    When enabled, the scalar coordinate components use the same colors as the
    corresponding hierarchy-level margin labels/boundaries.

    This helper lays out colored coordinate components using point offsets and
    an approximate monospace character width (`fontsize * 0.58`). That is a
    practical heuristic for pedagogical diagrams, but it is not exact text
    measurement and may vary slightly across renderers/platforms.
    """
    levels = _coord_levels(coord)
    pieces = [(f"{prefix}=(", base_color)]
    for level, value in enumerate(levels):
        pieces.append((str(value),
                       _hierarchy_level_color(level) if use_level_colors else base_color))
        if level != len(levels) - 1:
            pieces.append((",", base_color))
    pieces.append((")", base_color))

    # Left-align within the cell using monospace character widths in point offsets.
    char_width_pts = fontsize * 0.58
    offset_pts = 0
    for text, color in pieces:
        trans = mtransforms.offset_copy(ax.transData, fig=ax.figure,
                                        x=offset_pts, y=0, units='points')
        ax.text(x, y, text, transform=trans,
                ha='left', va='center', fontsize=fontsize,
                color=color, family='monospace')
        offset_pts += len(text) * char_width_pts


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


def _draw_hierarchical_grid(ax, layout,
                            cell_size: float = 1.0,
                            show_labels: bool = True,
                            title: Optional[str] = None,
                            colorize: bool = False,
                            flatten_hierarchical: bool = True,
                            label_hierarchy_levels: bool = False,
                            num_shades: int = 8):
    """Draw a hierarchical layout grid.

    Args:
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
    indices, rows, cols, row_struct, col_struct = _get_hierarchical_indices_2d(layout)
    cell_coords = _get_hierarchical_cell_coords_2d(layout)

    row_shape = mode(layout.shape, 0)
    col_shape = mode(layout.shape, 1)
    row_spans = _level_spans(row_shape)
    col_spans = _level_spans(col_shape)
    row_block_sizes = _level_block_sizes(row_shape)
    col_block_sizes = _level_block_sizes(col_shape)
    n_row_levels = len(row_block_sizes)
    n_col_levels = len(col_block_sizes)

    # Build palette
    if colorize:
        colors = _make_rainbow_palette(num_shades)
    else:
        colors = _make_grayscale_palette(num_shades)

    if flatten_hierarchical or not label_hierarchy_levels:
        left_margin = 0.9
        top_margin = 0.9
    else:
        # One label band per hierarchical level beyond the fastest-varying one.
        left_margin = 0.9 + 1.0 * max(n_row_levels, 0)
        top_margin = 0.9 + 0.7 * max(n_col_levels, 0)
    ax.set_xlim(-left_margin, cols + 0.5)
    ax.set_ylim(-top_margin, rows + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    max_levels = max(n_row_levels, n_col_levels, 1)
    coord_fontsize = max(3.2, 5.5 - 0.45 * (max_levels - 1))
    offset_fontsize = coord_fontsize

    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])
            color_idx = idx % len(colors)
            facecolor = colors[color_idx]
            if flatten_hierarchical:
                edgecolor = 'black'
                linewidth = 0.5
            else:
                # In pedagogical hierarchical mode, keep per-cell borders light
                # so hierarchy boundary lines stand out clearly.
                edgecolor = '#444444'
                linewidth = 0.3

            rect = patches.Rectangle(
                (j, i), cell_size, cell_size,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
                zorder=1
            )
            ax.add_patch(rect)

            text_color = 'white' if _is_dark(facecolor) else 'black'
            if flatten_hierarchical:
                # Draw flat offset value in cell
                ax.text(j + 0.5, i + 0.5, str(idx),
                        ha='center', va='center', fontsize=8, color=text_color)
            else:
                # Draw explicit pedagogical labels: row coordinate, column
                # coordinate, and resulting offset.
                row_coord, col_coord = cell_coords[i, j]
                row_line, col_line, off_line = _format_hierarchical_cell_lines(row_coord, col_coord, idx)
                x_left = j + 0.12
                _draw_colored_coord_line(
                    ax, x_left, i + 0.22, "row", row_coord,
                    text_color, coord_fontsize, use_level_colors=label_hierarchy_levels
                )
                _draw_colored_coord_line(
                    ax, x_left, i + 0.50, "col", col_coord,
                    text_color, coord_fontsize, use_level_colors=label_hierarchy_levels
                )
                offset_label, offset_value = off_line.split("=", 1)
                ax.text(x_left, i + 0.78, f"{offset_label}=",
                        ha='left', va='center', fontsize=offset_fontsize, color=text_color,
                        family='monospace')
                offset_label_pts = len(f"{offset_label}=") * offset_fontsize * 0.58
                trans = mtransforms.offset_copy(ax.transData, fig=ax.figure,
                                                x=offset_label_pts, y=0, units='points')
                ax.text(x_left, i + 0.78, offset_value, transform=trans,
                        ha='left', va='center', fontsize=offset_fontsize,
                        color=text_color, fontweight='bold', family='monospace')

    # Draw blue tile boundary lines for nested view
    if not flatten_hierarchical:
        def _is_shadowed_by_coarser(level: int, pos: int, block_sizes: tuple[int, ...]) -> bool:
            """Return True if a same-orientation coarser hierarchy line also sits at pos."""
            for coarser_level in range(level + 1, len(block_sizes)):
                coarser_block = block_sizes[coarser_level]
                if coarser_block > 0 and pos % coarser_block == 0:
                    return True
            return False

        # Horizontal boundaries for each hierarchy level beyond the
        # fastest-varying one. Draw from coarsest to finest so intersections
        # are colored by the finest granularity present there, while the
        # thicker coarser lines remain underneath.
        for level in reversed(range(1, n_row_levels)):
            block_size = row_block_sizes[level]
            color = HIERARCHY_LEVEL_COLORS[(level - 1) % len(HIERARCHY_LEVEL_COLORS)]
            linewidth = 2.0 + 1.2 * (level - 1)
            for i in range(block_size, rows, block_size):
                if _is_shadowed_by_coarser(level, i, row_block_sizes):
                    continue
                ax.plot([0, cols], [i, i], color=color, linewidth=linewidth,
                        zorder=4 + (n_row_levels - level))
        # Vertical boundaries for each hierarchy level beyond the
        # fastest-varying one. Draw from coarsest to finest for the same
        # reason as horizontal boundaries.
        for level in reversed(range(1, n_col_levels)):
            block_size = col_block_sizes[level]
            color = HIERARCHY_LEVEL_COLORS[(level - 1) % len(HIERARCHY_LEVEL_COLORS)]
            linewidth = 2.0 + 1.2 * (level - 1)
            for j in range(block_size, cols, block_size):
                if _is_shadowed_by_coarser(level, j, col_block_sizes):
                    continue
                ax.plot([j, j], [0, rows], color=color, linewidth=linewidth,
                        zorder=4 + (n_col_levels - level))

    if show_labels:
        if not flatten_hierarchical and label_hierarchy_levels:
            # Annotate each hierarchy level at the block/tile granularity where
            # that level is constant, rather than labeling every displayed row/col.
            for level in range(n_row_levels):
                block_size = row_block_sizes[level]
                color = _hierarchy_level_color(level)
                x = -0.55 - 1.0 * level
                for start in range(0, rows, block_size):
                    center = start + block_size / 2
                    value = _coord_levels(idx2crd(start, row_shape))[level]
                    ax.text(x, center, f"row[{level}]={value}",
                            ha='center', va='center', fontsize=7,
                            color=color, fontweight='bold')
            for level in range(n_col_levels):
                block_size = col_block_sizes[level]
                color = _hierarchy_level_color(level)
                y = -0.55 - 0.7 * level
                for start in range(0, cols, block_size):
                    center = start + block_size / 2
                    value = _coord_levels(idx2crd(start, col_shape))[level]
                    ax.text(center, y, f"col[{level}]={value}",
                            ha='center', va='center', fontsize=7,
                            color=color, fontweight='bold')
        else:
            # Single-level labels
            for i in range(rows):
                ax.text(-0.3, i + 0.5, f"R{i}", ha='center', va='center',
                        fontsize=8, color='blue')
            for j in range(cols):
                ax.text(j + 0.5, -0.3, f"C{j}", ha='center', va='center',
                        fontsize=8, color='blue')


def draw_layout(layout, filename=None,
                title: Optional[str] = None,
                dpi: int = 150,
                figsize: Optional[Tuple[float, float]] = None,
                colorize: bool = False,
                color_layout: Optional[Layout] = None,
                num_shades: int = 8,
                flatten_hierarchical: bool = True,
                label_hierarchy_levels: bool = False):
    """Draw a layout and save to file.

    Args:
        layout: Layout object to visualize
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
        num_shades: Number of colors/shades in palette (default 8)
        flatten_hierarchical: For hierarchical layouts, if True show flat grid with
            offset values. If False, show explicit cell labels:
              - row=... nested row coordinate
              - col=... nested column coordinate
              - offset=... resulting offset
        label_hierarchy_levels: For hierarchical nested views, if True annotate
            axes with each hierarchy level at block/tile granularity. Label
            colors match the corresponding hierarchy boundary lines.
    """
    # Check if this is a hierarchical layout (has nested tuple shapes)
    r = rank(layout)
    is_hierarchical = (r == 2 and
                       (isinstance(mode(layout.shape, 0), tuple) or
                        isinstance(mode(layout.shape, 1), tuple)))

    if is_hierarchical and not flatten_hierarchical:
        # Use hierarchical grid rendering
        try:
            indices, rows, cols, _, _ = _get_hierarchical_indices_2d(layout)
        except (ValueError, TypeError):
            # Fall back to regular rendering if hierarchical extraction fails
            indices = _get_indices_2d(layout)
            rows, cols = indices.shape
            is_hierarchical = False
    else:
        indices = _get_indices_2d(layout)
        rows, cols = indices.shape

    color_indices = _get_color_indices_2d(layout, color_layout)

    if figsize is None:
        # Larger cells for nested coordinate display
        cell_scale = 0.8 if (is_hierarchical and not flatten_hierarchical) else 0.5
        figsize = (cols * cell_scale + 1, rows * cell_scale + 1)

    fig, ax = plt.subplots(figsize=figsize)

    if is_hierarchical and not flatten_hierarchical:
        _draw_hierarchical_grid(ax, layout, title=title or str(layout),
                                colorize=colorize, flatten_hierarchical=False,
                                label_hierarchy_levels=label_hierarchy_levels,
                                num_shades=num_shades)
    else:
        _draw_grid(ax, indices, title=title or str(layout),
                   colorize=colorize, color_layout=color_layout,
                   color_indices=color_indices, num_shades=num_shades)

    _save_figure(fig, filename, dpi)


def _compute_tv_mapping(layout, grid_cols: Optional[int] = None,
                        grid_rows: Optional[int] = None,
                        thr_id_layout=None,
                        col_major: bool = True):
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
                # Store (physical_thread_id, value_id, logical_thread_id)
                inv_map[(row, col)] = (phys_t, flat_v, flat_t)
            else:
                inv_map[out_idx] = (phys_t, flat_v, flat_t)

    return inv_map


def _draw_tv_grid(ax, layout,
                  cell_size: float = 1.0,
                  title: Optional[str] = None,
                  colorize: bool = False,
                  num_threads: Optional[int] = None,
                  label_margin: float = 0.5,
                  title_position: str = "top",
                  grid_rows: Optional[int] = None,
                  grid_cols: Optional[int] = None,
                  thr_id_layout=None,
                  col_major: bool = True):
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
    if grid_rows is not None and grid_cols is not None:
        rows, cols = grid_rows, grid_cols
    else:
        # Infer from cosize
        cosize_val = cosize(layout)
        cols = int(np.sqrt(cosize_val))
        while cols > 0 and cosize_val % cols != 0:
            cols -= 1
        rows = cosize_val // cols if cols > 0 else cosize_val

    # Build the inverse mapping with grid dimensions
    tv_map = _compute_tv_mapping(layout, grid_cols=cols, grid_rows=rows,
                                 thr_id_layout=thr_id_layout,
                                 col_major=col_major)

    # Build color palette based on number of threads (cycle through 8 colors like cute-viz)
    n_colors = num_threads if num_threads else min(num_t, 8)
    if colorize:
        colors = _make_rainbow_palette(n_colors)
    else:
        colors = _make_grayscale_palette(n_colors)

    # Set up axes with label margin
    ax.set_xlim(-label_margin, cols + label_margin)
    ax.set_ylim(-label_margin, rows + label_margin)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        if title_position == "top":
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        else:  # "bottom"
            ax.text(cols / 2, rows + 0.5, title,
                    ha='center', va='top', fontsize=10, fontweight='bold')

    # Draw each cell
    for i in range(rows):
        for j in range(cols):
            # Use (row, col) tuple as key since _compute_tv_mapping returns tuples
            key = (i, j)

            if key in tv_map:
                phys_t, v_idx, logical_t = tv_map[key]
                t_idx = phys_t       # physical thread ID for label
                color_t = logical_t  # logical thread ID for color
            else:
                t_idx, v_idx, color_t = -1, -1, -1

            # Color by logical thread index (so T0 and T16 get different colors)
            color_idx = color_t % len(colors) if color_t >= 0 else 0
            facecolor = colors[color_idx]
            edgecolor = 'black'
            linewidth = 1

            # Draw cell
            rect = patches.Rectangle(
                (j, i), cell_size, cell_size,
                facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth
            )
            ax.add_patch(rect)

            # Draw "Tx" at top quarter and "Vy" at bottom quarter (cute-viz style)
            text_color = 'white' if _is_dark(facecolor) else 'black'
            if t_idx >= 0:
                # Thread label at top quarter of cell
                ax.text(j + 0.5, i + 0.3, f"T{t_idx}",
                        ha='center', va='center', fontsize=7, color=text_color)
                # Value label at bottom quarter of cell
                ax.text(j + 0.5, i + 0.7, f"V{v_idx}",
                        ha='center', va='center', fontsize=7, color=text_color)
            else:
                ax.text(j + 0.5, i + 0.5, "?",
                        ha='center', va='center', fontsize=7, color=text_color)


def draw_tv_layout(layout, filename=None,
                   title: Optional[str] = None,
                   dpi: int = 150,
                   figsize: Optional[Tuple[float, float]] = None,
                   colorize: bool = False,
                   num_threads: Optional[int] = None,
                   grid_shape: Optional[Tuple[int, int]] = None,
                   thr_id_layout=None,
                   col_major: bool = True):
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
        num_threads: Override number of colors (defaults to T dimension)
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
    r = rank(layout)
    if r != 2:
        raise ValueError(f"TV layout must be rank 2, got rank {r}")

    # Determine grid dimensions
    if grid_shape:
        rows, cols = grid_shape
    else:
        # Infer from cosize (may not work well for sparse layouts)
        cosize_val = cosize(layout)
        cols = int(np.sqrt(cosize_val))
        while cols > 0 and cosize_val % cols != 0:
            cols -= 1
        rows = cosize_val // cols if cols > 0 else cosize_val

    if figsize is None:
        figsize = (cols * 0.6 + 1.5, rows * 0.5 + 1)

    fig, ax = plt.subplots(figsize=figsize)
    _draw_tv_grid(ax, layout, title=title or f"TV: {layout}",
                  colorize=colorize, num_threads=num_threads,
                  grid_rows=rows, grid_cols=cols,
                  thr_id_layout=thr_id_layout,
                  col_major=col_major)
    _save_figure(fig, filename, dpi)


def draw_mma_layout(layout_a, layout_b, layout_c, filename=None,
                    tile_mnk: Optional[Tuple[int, int, int]] = None,
                    main_title: Optional[str] = None,
                    dpi: int = 150,
                    colorize: bool = True,
                    thr_id_layout=None):
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
    # Infer M, N, K from tile_mnk or layout dimensions
    if tile_mnk:
        M, N, K = tile_mnk
    else:
        # Fallback: try to infer from cosize
        cosize_a = cosize(layout_a)
        cosize_c = cosize(layout_c)
        # Assume A is M×K and C is M×N
        M = int(np.sqrt(cosize_a))
        while M > 0 and cosize_a % M != 0:
            M -= 1
        K = cosize_a // M if M > 0 else cosize_a
        N = cosize_c // M if M > 0 else cosize_c

    # Color palette (matching cute-viz: 8 pastel colors)
    if colorize:
        rgb_colors = [
            (175/255, 175/255, 255/255),  # light blue
            (175/255, 255/255, 175/255),  # light green
            (255/255, 255/255, 175/255),  # light yellow
            (255/255, 175/255, 175/255),  # light red
            (210/255, 210/255, 255/255),  # pale blue
            (210/255, 255/255, 210/255),  # pale green
            (255/255, 255/255, 210/255),  # pale yellow
            (255/255, 210/255, 210/255),  # pale red
        ]
    else:
        rgb_colors = _make_grayscale_palette(8)

    # Cell size and spacing (in figure units)
    cell_size = 1.0
    gap = 2.0  # Gap between matrices
    label_margin = 1.5  # Space for labels

    # Calculate total figure dimensions
    # Layout: B is at top-right (aligned with C columns), A is bottom-left, C is bottom-right
    total_width = K + gap + N + label_margin
    total_height = K + gap + M + label_margin

    # Scale for reasonable figure size
    scale = 0.35
    fig_width = total_width * scale + 1.5
    fig_height = total_height * scale + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def draw_tv_matrix(layout, offset_x, offset_y, matrix_rows, matrix_cols,
                       title, title_above=True, col_major=True):
        """Draw a TV matrix at the given offset with cute-viz style labeling."""
        tv_map = _compute_tv_mapping(layout, grid_cols=matrix_cols,
                                     grid_rows=matrix_rows,
                                     thr_id_layout=thr_id_layout,
                                     col_major=col_major)

        # Title
        if title:
            if title_above:
                ax.text(offset_x + matrix_cols / 2, offset_y - 0.6, title,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(offset_x + matrix_cols / 2, offset_y + matrix_rows + 0.6, title,
                        ha='center', va='top', fontsize=10, fontweight='bold')

        for i in range(matrix_rows):
            for j in range(matrix_cols):
                # Use (row, col) tuple as key
                key = (i, j)

                if key in tv_map:
                    phys_t, v_idx, logical_t = tv_map[key]
                    t_idx = phys_t
                else:
                    t_idx, v_idx, logical_t = -1, -1, -1

                color_idx = logical_t % len(rgb_colors) if logical_t >= 0 else 0
                facecolor = rgb_colors[color_idx]

                # Cell position
                x = offset_x + j
                y = offset_y + i

                rect = patches.Rectangle(
                    (x, y), cell_size, cell_size,
                    facecolor=facecolor, edgecolor='black', linewidth=0.5
                )
                ax.add_patch(rect)

                # Thread label at top quarter of cell (cute-viz style)
                if t_idx >= 0:
                    ax.text(x + 0.5, y + 0.3, f"T{t_idx}",
                            ha='center', va='center', fontsize=6, color='black')
                    # Value label at bottom quarter of cell
                    ax.text(x + 0.5, y + 0.7, f"V{v_idx}",
                            ha='center', va='center', fontsize=6, color='black')
                else:
                    ax.text(x + 0.5, y + 0.5, "?",
                            ha='center', va='center', fontsize=6, color='gray')

    # Position matrices
    # B: top-right (aligned with C columns) - B matrix has K rows, N cols
    b_offset_x = K + gap
    b_offset_y = 0

    # A: bottom-left - A matrix has M rows, K cols
    a_offset_x = 0
    a_offset_y = K + gap

    # C: bottom-right (aligned with B columns and A rows) - C matrix has M rows, N cols
    c_offset_x = K + gap
    c_offset_y = K + gap

    # Draw matrices
    draw_tv_matrix(layout_b, b_offset_x, b_offset_y, K, N, f"B ({K}×{N})", title_above=True, col_major=False)
    draw_tv_matrix(layout_a, a_offset_x, a_offset_y, M, K, f"A ({M}×{K})", title_above=False)
    draw_tv_matrix(layout_c, c_offset_x, c_offset_y, M, N, f"C ({M}×{N})", title_above=False)

    # Add dimension labels
    # K dimension labels for B (left side of B, showing row indices)
    for k in range(K):
        ax.text(b_offset_x - 0.4, k + 0.5, str(k), ha='right', va='center', fontsize=7, color='dimgray')

    # N dimension labels for B (top of B, showing column indices)
    for n in range(N):
        ax.text(b_offset_x + n + 0.5, -0.4, str(n), ha='center', va='bottom', fontsize=7, color='dimgray')

    # M dimension labels for A (left side of A)
    for m in range(M):
        ax.text(a_offset_x - 0.4, a_offset_y + m + 0.5, str(m), ha='right', va='center', fontsize=7, color='dimgray')

    # K dimension labels for A (top of A, showing column indices)
    for k in range(K):
        ax.text(a_offset_x + k + 0.5, a_offset_y - 0.4, str(k), ha='center', va='bottom', fontsize=7, color='dimgray')

    # Set axes limits with some padding
    ax.set_xlim(-label_margin, total_width)
    ax.set_ylim(-label_margin, total_height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    _save_figure(fig, filename, dpi)


def draw_tiled_grid(grid: dict, rows: int, cols: int,
                    filename=None, dpi: int = 150,
                    title: Optional[str] = None):
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
    colors = _make_rainbow_palette(8)
    font = max(4, min(7, int(60 / max(rows, cols))))
    fig, ax = plt.subplots(figsize=(cols * 0.45 + 1.5, rows * 0.4 + 1.0))
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
    for i in range(rows):
        for j in range(cols):
            if (i, j) in grid:
                pt, v, lt = grid[(i, j)]
                fc = colors[lt % len(colors)]
            else:
                pt, v, fc = -1, -1, '#FFFFFF'
            ax.add_patch(patches.Rectangle(
                (j, i), 1, 1, facecolor=fc,
                edgecolor='black', linewidth=0.5))
            if pt >= 0:
                tc = 'white' if _is_dark(fc) else 'black'
                ax.text(j + 0.5, i + 0.35, f'T{pt}',
                        ha='center', va='center', fontsize=font, color=tc)
                ax.text(j + 0.5, i + 0.7, f'V{v}',
                        ha='center', va='center', fontsize=font, color=tc)
    plt.tight_layout()
    _save_figure(fig, filename, dpi)


def draw_swizzle(base_layout, swizzle, filename=None,
                 dpi: int = 150,
                 figsize: Optional[Tuple[float, float]] = None,
                 colorize: bool = False,
                 num_shades: int = 8):
    """Draw side-by-side comparison of linear vs swizzled layout.

    For swizzles with base=0 (affecting low bits), colors by value % num_shades
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
        num_shades: Number of colors/shades in palette
    """
    sw_layout = compose(swizzle, base_layout)

    linear_idx = _get_indices_2d(base_layout)
    swizzle_idx = _get_indices_2d(sw_layout)

    rows, cols = linear_idx.shape

    if swizzle.base > 0 and cols > (1 << swizzle.base):
        # Full element view with block-based coloring: show every element but
        # color by which block of 2^base elements it belongs to.  This matches
        # the presentation style used by NVIDIA's CuTe swizzle visualizations
        # where all values are visible and color bands reveal the permutation.
        block_size = 1 << swizzle.base
        blocks_per_row = cols // block_size

        if figsize is None:
            figsize = (cols * 0.6 + 3, rows * 0.5 + 1.5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Color by block identity: (val >> base) % blocks_per_row
        # In the linear layout these form contiguous color bands;
        # in the swizzled layout the bands get permuted across rows.
        _draw_grid_by_bits(ax1, linear_idx, swizzle.base,
                           title=f"Linear: {base_layout}",
                           colorize=colorize, num_shades=blocks_per_row)
        _draw_grid_by_bits(ax2, swizzle_idx, swizzle.base,
                           title=f"Swizzled: {swizzle}",
                           colorize=colorize, num_shades=blocks_per_row)
    elif swizzle.base == 0:
        # Element-level view: color by value % num_shades
        if figsize is None:
            figsize = (cols * 1.0 + 3, rows * 0.5 + 1.5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        _draw_grid_by_value_mod(ax1, linear_idx, title=f"Linear: {base_layout}",
                                colorize=colorize, num_shades=num_shades)
        _draw_grid_by_value_mod(ax2, swizzle_idx, title=f"Swizzled: {swizzle}",
                                colorize=colorize, num_shades=num_shades)
    else:
        # base>0 but grid not wide enough for block view — fall back to bit-shift
        if figsize is None:
            figsize = (cols * 1.0 + 3, rows * 0.5 + 1.5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        bit_shift = swizzle.base
        distinct_groups = len(set(int(v) >> bit_shift for v in linear_idx.flat))
        effective_shades = max(num_shades, distinct_groups)
        _draw_grid_by_bits(ax1, linear_idx, bit_shift, title=f"Linear: {base_layout}",
                           colorize=colorize, num_shades=effective_shades)
        _draw_grid_by_bits(ax2, swizzle_idx, bit_shift, title=f"Swizzled: {swizzle}",
                           colorize=colorize, num_shades=effective_shades)

    plt.tight_layout()
    _save_figure(fig, filename, dpi)


def _draw_grid_by_bits(ax, indices: np.ndarray, bit_shift: int,
                       cell_size: float = 1.0,
                       title: Optional[str] = None,
                       colorize: bool = False,
                       num_shades: int = 8):
    """Draw a grid with coloring based on shifted bits.

    Colors cells by (value >> bit_shift) % num_shades, showing the bits
    that a swizzle affects. This ensures the swizzle effect is visible
    regardless of which bits are being XORed.
    """
    rows, cols = indices.shape

    # Build color palette
    if colorize:
        colors = _make_rainbow_palette(num_shades)
    else:
        colors = _make_grayscale_palette(num_shades)

    # Set up axes
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    # Draw cells
    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])

            # Color by shifted bits: (value >> bit_shift) % num_shades
            color_idx = (idx >> bit_shift) % len(colors)
            facecolor = colors[color_idx]

            rect = patches.Rectangle(
                (j, i), cell_size, cell_size,
                facecolor=facecolor, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)

            # Draw index text
            text_color = 'white' if _is_dark(facecolor) else 'black'
            ax.text(j + 0.5, i + 0.5, str(idx),
                    ha='center', va='center', fontsize=8, color=text_color)

    # Row labels (left)
    for i in range(rows):
        ax.text(-0.3, i + 0.5, str(i),
                ha='right', va='center', fontsize=7, color='gray')

    # Column labels (top)
    for j in range(cols):
        ax.text(j + 0.5, -0.3, str(j),
                ha='center', va='bottom', fontsize=7, color='gray')


def _draw_grid_by_value_mod(ax, indices: np.ndarray,
                            cell_size: float = 1.0,
                            title: Optional[str] = None,
                            colorize: bool = False,
                            num_shades: int = 8):
    """Draw a grid with coloring based on value % num_shades.

    This is the standard cute-viz coloring scheme. It works well for swizzles
    with base=0 (affecting low bits) where the permutation is visible directly.
    """
    rows, cols = indices.shape

    # Build color palette
    if colorize:
        colors = _make_rainbow_palette(num_shades)
    else:
        colors = _make_grayscale_palette(num_shades)

    # Set up axes
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    # Draw cells
    for i in range(rows):
        for j in range(cols):
            idx = int(indices[i, j])

            # Color by value modulo num_shades
            color_idx = idx % len(colors)
            facecolor = colors[color_idx]

            rect = patches.Rectangle(
                (j, i), cell_size, cell_size,
                facecolor=facecolor, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)

            # Draw index text
            text_color = 'white' if _is_dark(facecolor) else 'black'
            ax.text(j + 0.5, i + 0.5, str(idx),
                    ha='center', va='center', fontsize=8, color=text_color)

    # Row labels (left)
    for i in range(rows):
        ax.text(-0.3, i + 0.5, str(i),
                ha='right', va='center', fontsize=7, color='gray')

    # Column labels (top)
    for j in range(cols):
        ax.text(j + 0.5, -0.3, str(j),
                ha='center', va='bottom', fontsize=7, color='gray')


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
            # Cartesian product
            import itertools
            for combo in itertools.product(*sub_iters):
                yield combo
        else:
            # scalar shape: enumerate 0..shape-1
            for i in range(shape):
                yield i
    elif is_tuple(spec):
        if is_tuple(shape):
            assert len(spec) == len(shape)
            sub_iters = [_expand_hier_slice(s, sh) for s, sh in zip(spec, shape)]
            import itertools
            for combo in itertools.product(*sub_iters):
                yield combo
        else:
            raise TypeError(f"Tuple spec {spec} for scalar shape {shape}")
    else:
        # Concrete integer: single value
        yield spec


def draw_slice(layout, slice_spec, filename=None,
               title: Optional[str] = None,
               dpi: int = 150,
               figsize: Optional[Tuple[float, float]] = None,
               colorize: bool = False,
               color_layout: Optional[Layout] = None,
               num_shades: int = 8):
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
        num_shades: Number of colors/shades in palette
    """
    indices = _get_indices_2d(layout)
    rows, cols = indices.shape

    # Compute highlighted indices
    highlight = set()
    r = rank(layout)

    def _is_flat_spec(s):
        """Check if a slice component is flat (int, None, or slice)."""
        return s is None or isinstance(s, (int, slice))

    if isinstance(slice_spec, int):
        # Single linear index
        highlight.add(layout(slice_spec))
    elif isinstance(slice_spec, tuple) and r == 2:
        row_spec, col_spec = slice_spec

        if _is_flat_spec(row_spec) and _is_flat_spec(col_spec):
            # Flat 2D slicing (original path)
            # Parse row range
            if row_spec is None:
                row_range = range(rows)
            elif isinstance(row_spec, int):
                row_range = [row_spec]
            else:
                row_range = range(*row_spec.indices(rows))

            # Parse column range
            if col_spec is None:
                col_range = range(cols)
            elif isinstance(col_spec, int):
                col_range = [col_spec]
            else:
                col_range = range(*col_spec.indices(cols))

            for i in row_range:
                for j in col_range:
                    highlight.add(layout(i, j))
        else:
            # Hierarchical slicing: expand nested specs with None wildcards
            import itertools
            row_coords = list(_expand_hier_slice(row_spec, layout.shape[0]))
            col_coords = list(_expand_hier_slice(col_spec, layout.shape[1]))
            for rc, cc in itertools.product(row_coords, col_coords):
                highlight.add(layout(rc, cc))

    if figsize is None:
        figsize = (cols * 0.5 + 1, rows * 0.5 + 1)

    if title is None:
        title = f"{layout}[{slice_spec}]"

    fig, ax = plt.subplots(figsize=figsize)
    color_indices = _get_color_indices_2d(layout, color_layout)
    _draw_grid(ax, indices, highlight=highlight, title=title,
               colorize=colorize, color_layout=color_layout,
               color_indices=color_indices, num_shades=num_shades)
    _save_figure(fig, filename, dpi)


def show_layout(layout, title: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None,
                colorize: bool = False,
                color_layout: Optional[Layout] = None,
                num_shades: int = 8):
    """Display a layout inline (for Jupyter notebooks).

    Args:
        layout: Layout object to visualize
        title: Optional title
        figsize: Figure size in inches
        colorize: If True, use rainbow colors for distinct cells
        color_layout: Optional layout controlling cell coloring in the same
            logical coordinate space as `layout` (None = color by value)
        num_shades: Number of colors/shades in palette

    Returns:
        matplotlib Figure
    """
    indices = _get_indices_2d(layout)
    rows, cols = indices.shape

    if figsize is None:
        figsize = (cols * 0.5 + 1, rows * 0.5 + 1)

    fig, ax = plt.subplots(figsize=figsize)
    color_indices = _get_color_indices_2d(layout, color_layout)
    _draw_grid(ax, indices, title=title or str(layout),
               colorize=colorize, color_layout=color_layout,
               color_indices=color_indices, num_shades=num_shades)
    return fig


def show_swizzle(base_layout, swizzle,
                 figsize: Optional[Tuple[float, float]] = None,
                 colorize: bool = False,
                 num_shades: int = 8):
    """Display swizzle comparison inline (for Jupyter notebooks).

    Colors by the bits that the swizzle affects: (value >> swizzle.base) % num_shades.

    Args:
        base_layout: Base Layout object
        swizzle: Swizzle object
        figsize: Figure size in inches
        colorize: If True, use rainbow colors (makes swizzle movement clearer)
        num_shades: Number of colors/shades in palette

    Returns:
        matplotlib Figure
    """
    sw_layout = compose(swizzle, base_layout)

    linear_idx = _get_indices_2d(base_layout)
    swizzle_idx = _get_indices_2d(sw_layout)

    rows, cols = linear_idx.shape

    if figsize is None:
        figsize = (cols * 1.0 + 3, rows * 0.5 + 1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Color by the bits that the swizzle affects
    bit_shift = swizzle.base
    _draw_grid_by_bits(ax1, linear_idx, bit_shift, title=f"Linear: {base_layout}",
                       colorize=colorize, num_shades=num_shades)
    _draw_grid_by_bits(ax2, swizzle_idx, bit_shift, title=f"Swizzled: {swizzle}",
                       colorize=colorize, num_shades=num_shades)

    plt.tight_layout()
    return fig


# =============================================================================
# Demo
# =============================================================================

def demo(output_dir: str = "."):
    """Generate example visualizations in all formats."""
    from pathlib import Path
    from .layouts import logical_divide

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
    print(f"✓ Row-major 8x8 (SVG, PNG, PDF)")

    layout_col = Layout((8, 8), (1, 8))
    draw_layout(layout_col, output / "col_major.svg")
    print(f"✓ Column-major 8x8")

    # Swizzle comparisons
    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), output / "swizzle_303.svg")
    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), output / "swizzle_303.png")
    print(f"✓ Swizzle(3,0,3) comparison (SVG, PNG)")

    draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(2, 0, 3), output / "swizzle_203.svg")
    print(f"✓ Swizzle(2,0,3) comparison")

    draw_swizzle(Layout((4, 8), (8, 1)), Swizzle(2, 1, 3), output / "swizzle_213.svg")
    print(f"✓ Swizzle(2,1,3) on 4x8")

    # Slice highlights
    layout_4x8 = Layout((4, 8), (8, 1))
    draw_slice(layout_4x8, (2, None), output / "slice_row.svg")
    draw_slice(layout_4x8, (2, None), output / "slice_row.png")
    print(f"✓ Slice row 2 (SVG, PNG)")

    draw_slice(layout_4x8, (None, 5), output / "slice_col.svg")
    print(f"✓ Slice column 5")

    draw_slice(layout_4x8, (2, 5), output / "slice_single.svg")
    print(f"✓ Slice single element (2, 5)")

    draw_slice(layout_4x8, (slice(1, 3), slice(2, 6)), output / "slice_rect.svg")
    print(f"✓ Slice rectangular region")

    # Hierarchical layout
    hier = Layout(((2, 4), 8), ((1, 2), 8))
    draw_layout(hier, output / "hierarchical.svg")
    draw_layout(hier, output / "hierarchical.png")
    print(f"✓ Hierarchical layout (SVG, PNG)")

    # Logical divide
    divided = Layout((4, 4), (1, 4))  # Result of logical_divide
    draw_layout(divided, output / "divide_result.svg")
    print(f"✓ Logical divide result")

    # Different sizes
    draw_layout(Layout((4, 16), (16, 1)), output / "wide.svg")
    draw_layout(Layout((16, 4), (4, 1)), output / "tall.svg")
    print(f"✓ Wide and tall layouts")

    print(f"\n✓ All visualizations saved to {output_dir}/")
    print(f"  Formats: .svg (vector), .png (raster), .pdf (print)")


if __name__ == "__main__":
    import sys
    demo(sys.argv[1] if len(sys.argv) > 1 else "./viz_output")
