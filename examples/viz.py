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

# ruff: noqa: F405

"""Visualization Examples Based on CuTe C++ Documentation.

This file demonstrates the visualization capabilities of the layouts library,
following examples from NVIDIA's CuTe (CUDA Templates) documentation.

Run this script after installing the package:
    pip install -e ".[viz]"
    python3 examples/viz.py

Output will be saved to ./examples_output/ directory.

Reference: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/
"""

from pathlib import Path
import sys

# Prefer the local repo sources when running this script from a checkout.
# An installed `tensor-layouts` distribution is still required for package metadata.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tensor_layouts import *
from tensor_layouts.atoms_nv import *
from tensor_layouts.atoms_amd import *
from tensor_layouts.viz import *


def setup_output_dir(name: str = "examples_output") -> Path:
    """Create output directory for examples."""
    output = Path(name)
    output.mkdir(parents=True, exist_ok=True)
    return output


# =============================================================================
# Section 1: Output Formats (SVG, PNG, PDF)
# =============================================================================


def example_output_formats(output: Path):
    """Demonstrate SVG, PNG, and PDF output formats.

    The visualization library supports three output formats:
    - SVG: Vector format, ideal for documentation and web
    - PNG: Raster format, good for presentations and embedding
    - PDF: Print-ready format, ideal for papers and reports

    The format is determined by the file extension.

    Coloring is controlled by color_layout, which is itself a Layout:
    - color_layout=None: color by cell value (default)
    - color_layout=Layout((r,c), (1, 0)): color by logical row
    - color_layout=Layout((r,c), (0, 1)): color by logical column
    - color_layout=Layout(1, 0): uniform color (no variation)

    For ordinary 2D layouts, displayed cell (row, col) is colored by
    evaluating color_layout at that same logical coordinate.
    """
    print("\n" + "=" * 60)
    print("1. Output Formats (SVG, PNG, PDF) and Coloring")
    print("=" * 60)

    # Create a simple layout to demonstrate all formats
    layout = Layout((4, 8), (8, 1))

    # SVG output - vector format (default, best for most uses)
    draw_layout(layout, output / "format_example.svg", title="(4,8):(8,1)")
    print(f"✓ SVG: format_example.svg (vector, scalable)")

    # PNG output - raster format with configurable DPI
    draw_layout(layout, output / "format_example.png", title="(4,8):(8,1)", dpi=150)
    print(f"✓ PNG: format_example.png (raster, 150 dpi)")

    # PDF output - print-ready format
    draw_layout(layout, output / "format_example.pdf", title="(4,8):(8,1)")
    print(f"✓ PDF: format_example.pdf (print-ready)")

    # Demonstrate color_layout options
    layout_8x8 = Layout((8, 8), (8, 1))

    # Color by value (default) - same value = same color
    draw_layout(
        layout_8x8, output / "color_by_value.svg", title="color_layout=None (by value)"
    )
    print(f"✓ Color by value: color_by_value.svg")

    # Color by column - darker across columns (cute-viz style)
    draw_layout(
        layout_8x8,
        output / "color_by_col.svg",
        title="color_layout=(8,8):(0,1) (by column)",
        color_layout=Layout((8, 8), (0, 1)),
    )
    print(f"✓ Color by column: color_by_col.svg")

    # Color by row - darker down rows
    draw_layout(
        layout_8x8,
        output / "color_by_row.svg",
        title="color_layout=(8,8):(1,0) (by row)",
        color_layout=Layout((8, 8), (1, 0)),
    )
    print(f"✓ Color by row: color_by_row.svg")

    # Uniform color - no variation
    draw_layout(
        layout_8x8,
        output / "color_uniform.svg",
        title="color_layout=1:0 (uniform)",
        color_layout=Layout(1, 0),
    )
    print(f"✓ Uniform color: color_uniform.svg")

    # Rainbow colors with different color_layout
    draw_layout(
        layout_8x8,
        output / "color_by_col_rainbow.svg",
        title="colorize=True, by column",
        colorize=True,
        color_layout=Layout((8, 8), (0, 1)),
    )
    print(f"✓ Rainbow by column: color_by_col_rainbow.svg")

    # color_by shorthand — equivalent to the manual color_layout above
    draw_layout(
        layout_8x8,
        output / "color_by_row_shorthand.svg",
        title='color_by="row"',
        color_by="row",
    )
    draw_layout(
        layout_8x8,
        output / "color_by_col_shorthand.svg",
        title='color_by="column"',
        color_by="column",
    )
    print(
        f"✓ color_by shorthand: color_by_row_shorthand.svg, color_by_col_shorthand.svg"
    )

    # Swizzle comparison showing row-group coloring (reveals permutation effect)
    base = Layout((8, 8), (8, 1))
    sw = Swizzle(3, 0, 3)
    draw_swizzle(base, sw, output / "swizzle_example.svg")
    draw_swizzle(base, sw, output / "swizzle_example_color.svg", colorize=True)
    print(
        f"✓ Swizzle with row-group coloring: swizzle_example.svg, swizzle_example_color.svg"
    )


# =============================================================================
# Section 2: 1D Layouts
# =============================================================================


def example_1d_layouts(output: Path):
    """1D contiguous and strided layouts.

    CuTe Reference: Layout Algebra - 1D layouts

    A 1D layout maps indices to memory locations. With stride=1, elements
    are contiguous. With stride>1, elements are strided.
    """
    print("\n" + "=" * 60)
    print("2. 1D Layouts")
    print("=" * 60)

    # Contiguous 1D layout: 8 elements, stride 1
    layout_1d_contiguous = Layout(8, 1)
    draw_layout(
        layout_1d_contiguous, output / "1d_contiguous.svg", title="1D Contiguous: 8:1"
    )
    print(f"✓ 1D Contiguous: 8:1")
    print(f"  Maps index i → offset i (e.g., 3 → 3)")

    # Strided 1D layout: 8 elements, stride 2
    layout_1d_strided = Layout(8, 2)
    draw_layout(layout_1d_strided, output / "1d_strided.svg", title="1D Strided: 8:2")
    print(f"✓ 1D Strided: 8:2")
    print(f"  Maps index i → offset 2*i (e.g., 3 → 6)")

    # Strided 1D layout: 4 elements, stride 4
    layout_1d_stride4 = Layout(4, 4)
    draw_layout(layout_1d_stride4, output / "1d_stride4.svg", title="1D Stride-4: 4:4")
    print(f"✓ 1D Stride-4: 4:4")
    print(f"  Maps index i → offset 4*i (e.g., 2 → 8)")


# =============================================================================
# Section 3: 2D Layouts
# =============================================================================


def example_2d_layouts(output: Path):
    """2D row-major and column-major layouts.

    CuTe Reference: Layout Algebra - 2D layouts

    Row-major: consecutive elements in same row are contiguous
    Column-major: consecutive elements in same column are contiguous
    """
    print("\n" + "=" * 60)
    print("3. 2D Layouts")
    print("=" * 60)

    # Row-major 4x3: shape (4 rows, 3 cols), stride (3, 1)
    # Row i, Col j → offset = i*3 + j
    row_major_4x3 = Layout((4, 3), (3, 1))
    draw_layout(
        row_major_4x3,
        output / "2d_row_major_4x3.svg",
        title="Row-Major 4×3: (4,3):(3,1)",
    )
    print(f"✓ Row-Major 4×3: (4,3):(3,1)")
    print(f"  offset(i,j) = i*3 + j*1")

    # Column-major 4x3: shape (4 rows, 3 cols), stride (1, 4)
    # Row i, Col j → offset = i*1 + j*4
    col_major_4x3 = Layout((4, 3), (1, 4))
    draw_layout(
        col_major_4x3,
        output / "2d_col_major_4x3.svg",
        title="Col-Major 4×3: (4,3):(1,4)",
    )
    print(f"✓ Col-Major 4×3: (4,3):(1,4)")
    print(f"  offset(i,j) = i*1 + j*4")

    # 8x8 Row-major: shape (8 rows, 8 cols), stride (8, 1)
    # This is the common layout for matrix operations
    row_major_8x8 = Layout((8, 8), (8, 1))
    draw_layout(
        row_major_8x8,
        output / "2d_row_major_8x8.svg",
        title="Row-Major 8×8: (8,8):(8,1)",
    )
    draw_layout(
        row_major_8x8,
        output / "2d_row_major_8x8_color.svg",
        title="Row-Major 8×8: (8,8):(8,1)",
        colorize=True,
    )
    print(f"✓ Row-Major 8×8: (8,8):(8,1) [grayscale and colorized]")
    print(f"  offset(i,j) = i*8 + j*1")

    # 8x8 Column-major: shape (8 rows, 8 cols), stride (1, 8)
    col_major_8x8 = Layout((8, 8), (1, 8))
    draw_layout(
        col_major_8x8,
        output / "2d_col_major_8x8.svg",
        title="Col-Major 8×8: (8,8):(1,8)",
    )
    draw_layout(
        col_major_8x8,
        output / "2d_col_major_8x8_color.svg",
        title="Col-Major 8×8: (8,8):(1,8)",
        colorize=True,
    )
    print(f"✓ Col-Major 8×8: (8,8):(1,8) [grayscale and colorized]")
    print(f"  offset(i,j) = i*1 + j*8")


# =============================================================================
# Section 4: Hierarchical Layouts
# =============================================================================


def example_hierarchical_layouts(output: Path):
    """Hierarchical (nested) layouts - flattened and nested views.

    CuTe Reference: Layout Algebra - Hierarchical Layouts

    Hierarchical layouts have nested shapes/strides. They're used to represent
    tiled access patterns where the outer level selects tiles and inner level
    selects within tiles.
    """
    print("\n" + "=" * 60)
    print("4. Hierarchical Layouts")
    print("=" * 60)

    # =========================================================================
    # Example 1: 2×2 tiles in a 3×4 grid (cute-viz example)
    # Shape: ((2, 3), (2, 4)) = ((inner_row, outer_row), (inner_col, outer_col))
    # Total: 6 rows × 8 columns = 48 elements
    # =========================================================================
    print("\n  --- 2×2 Tiles in 3×4 Grid ---")
    hier_2x2_3x4 = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))

    # Show the mapping for first tile
    print(f"  Layout: ((2,3), (2,4)) : ((1,6), (2,12))")
    print(f"  Shape: 2×2 tiles arranged in 3×4 grid = 6×8 total")
    print(f"  Mapping examples:")
    for i in range(2):
        for j in range(2):
            idx = hier_2x2_3x4((i, 0), (j, 0))
            print(f"    (({i},0),({j},0)) → {idx}")

    # Flat view (default)
    draw_layout(
        hier_2x2_3x4,
        output / "hier_2x2_3x4_flat.svg",
        title=f"Flat: {hier_2x2_3x4}",
        flatten_hierarchical=True,
    )
    print(f"✓ Flat view: hier_2x2_3x4_flat.svg")

    # Nested pedagogical view:
    #   - each cell shows row=... (nested row coordinate)
    #   - each cell shows col=... (nested column coordinate)
    #   - each cell shows offset=... (resulting offset)
    #   - axes stay simple (R0, R1, ... / C0, C1, ...)
    draw_layout(
        hier_2x2_3x4,
        output / "hier_2x2_3x4_nested.svg",
        title=f"Nested: {hier_2x2_3x4}",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    print(f"✓ Nested view: hier_2x2_3x4_nested.svg")

    # =========================================================================
    # Example 2: 2×2 tiles in 2×2 grid (4×4 total) — the project logo layout
    # =========================================================================
    print("\n  --- 2×2 Tiles in 2×2 Grid (Logo Layout) ---")
    logo_layout = Layout(((2, 2), (2, 2)), ((1, 4), (2, 8)))

    # This is the layout shown in the project logo (docs/images/logo.svg).
    # It's a 4×4 Morton (Z-order) layout: blocked_product of 2×2 Z-tiles.
    # Inner 2×2 tiles use stride (1, 2) (column-major within each tile),
    # and the 2×2 outer grid uses stride (4, 8) to place tiles.
    #
    # Tile (0,0): 0 2    Tile (0,1): 8 10
    #             1 3                9 11
    #
    # Tile (1,0): 4 6    Tile (1,1): 12 14
    #             5 7                13 15

    draw_layout(
        logo_layout,
        output / "hier_2x2_tiles_flat.svg",
        title=f"Flat: {logo_layout}",
        flatten_hierarchical=True,
    )
    draw_layout(
        logo_layout,
        output / "hier_2x2_tiles_nested.svg",
        title=f"Nested: {logo_layout}",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    print(f"✓ Hierarchical 2×2 in 2×2 (logo layout): {logo_layout}")
    print(
        f"  Nested view is pedagogical: row=... / col=... show nested coordinates, offset=... shows mapping"
    )

    # =========================================================================
    # Example 3: 3-level asymmetric hierarchy with per-level axis labels
    # =========================================================================
    print("\n  --- 3-Level Asymmetric Hierarchy ---")
    hier_3level = Layout(
        ((2, 3, 2), (3, 2, 2)),
        ((1, 2, 6), (12, 36, 72)),
    )
    draw_layout(
        hier_3level,
        output / "hier_3level_asymmetric_flat.svg",
        title=f"Flat: {hier_3level}",
        flatten_hierarchical=True,
    )
    draw_layout(
        hier_3level,
        output / "hier_3level_asymmetric_nested.svg",
        title=f"Nested: {hier_3level}",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    print("✓ 3-level asymmetric hierarchy")
    print("  Row shape = (2,3,2), Col shape = (3,2,2)")
    print("  Nested view labels hierarchy levels at tile/block granularity")
    print("  and uses matching colors for boundary lines and level labels.")
    print("  Axis labels use row[k]=... / col[k]=... to match cell notation.")
    print("  Output: hier_3level_asymmetric_nested.svg")

    # =========================================================================
    # Example 4: 4-level asymmetric hierarchy with larger finest-level tiles
    # =========================================================================
    print("\n  --- 4-Level Asymmetric Hierarchy (Small Cells) ---")
    hier_4level = Layout(
        ((3, 2, 2, 2), (4, 2, 2, 2)),
        ((1, 3, 6, 12), (24, 96, 192, 384)),
    )
    draw_layout(
        hier_4level,
        output / "hier_4level_asymmetric_flat.svg",
        title=f"Flat: {hier_4level}",
        flatten_hierarchical=True,
    )
    for ext in ("svg", "pdf", "png"):
        draw_layout(
            hier_4level,
            output / f"hier_4level_asymmetric_nested.{ext}",
            title=f"Nested: {hier_4level}",
            flatten_hierarchical=False,
            label_hierarchy_levels=True,
            dpi=300 if ext == "png" else 150,
        )
    print("✓ 4-level asymmetric hierarchy")
    print("  Row shape = (3,2,2,2), Col shape = (4,2,2,2)")
    print("  This example makes cells much smaller, so it is useful for checking")
    print("  whether hierarchy-level labels and colored boundaries remain readable.")
    print("  Output: hier_4level_asymmetric_nested.{svg,pdf,png}")

    # Flatten the hierarchical layout (algebra operation)
    flat_layout = flatten(logo_layout)
    draw_layout(
        flat_layout, output / "hier_flattened.svg", title=f"flatten(): {flat_layout}"
    )
    print(f"✓ Flattened (algebra): {flat_layout}")

    # Coalesce to merge contiguous dimensions
    coal_layout = coalesce(logo_layout)
    draw_layout(
        coal_layout, output / "hier_coalesced.svg", title=f"coalesce(): {coal_layout}"
    )
    print(f"✓ Coalesced: {coal_layout}")

    # =========================================================================
    # Examples from "Cute Layout Representation and Algebra" by Cris Cecka
    # =========================================================================
    print("\n  --- From Cecka, 'CuTe Layout Representation and Algebra' ---")

    # (4,8):(1,4) — column-major 4x8
    cecka_1 = Layout((4, 8), (1, 4))
    draw_layout(cecka_1, output / "cecka_4x8_col.svg", title="(4,8):(1,4)")
    print(f"✓ (4,8):(1,4) — column-major 4×8")

    # (4,8):(8,1) — row-major 4x8
    cecka_2 = Layout((4, 8), (8, 1))
    draw_layout(cecka_2, output / "cecka_4x8_row.svg", title="(4,8):(8,1)")
    print(f"✓ (4,8):(8,1) — row-major 4×8")

    # (4,8):(1,5) — non-injective layout (stride 5 with shape 8 wraps)
    cecka_3 = Layout((4, 8), (1, 5))
    draw_layout(cecka_3, output / "cecka_4x8_s1_s5.svg", title="(4,8):(1,5)")
    print(f"✓ (4,8):(1,5) — non-injective (surjective) layout")

    # (4,(4,2)):(4,(1,16)) — hierarchical column dimension
    # Nested rendering explicitly shows how the hierarchical column coordinate
    # maps to the final offset for each displayed cell.
    cecka_4 = Layout((4, (4, 2)), (4, (1, 16)))
    draw_layout(
        cecka_4,
        output / "cecka_hier_col.svg",
        title="(4,(4,2)):(4,(1,16))",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    draw_layout(
        cecka_4,
        output / "cecka_hier_col_flat.svg",
        title="(4,(4,2)):(4,(1,16))",
        flatten_hierarchical=True,
    )
    print(f"✓ (4,(4,2)):(4,(1,16)) — hierarchical column")

    # ((2,2),(4,2)):((1,8),(2,16)) — hierarchical in both modes
    # This is a good example where explicit row=... / col=... labels help explain
    # the two-level row and column structure.
    cecka_5 = Layout(((2, 2), (4, 2)), ((1, 8), (2, 16)))
    draw_layout(
        cecka_5,
        output / "cecka_hier_both.svg",
        title="((2,2),(4,2)):((1,8),(2,16))",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    draw_layout(
        cecka_5,
        output / "cecka_hier_both_flat.svg",
        title="((2,2),(4,2)):((1,8),(2,16))",
        flatten_hierarchical=True,
    )
    print(f"✓ ((2,2),(4,2)):((1,8),(2,16)) — hierarchical both modes")

    # ((2,2),(2,4)):((0,2),(0,4)) — zero-stride (broadcast) layout
    # The pedagogical nested view is especially useful here because repeated
    # offsets are easier to interpret when the source coordinates are explicit.
    cecka_6 = Layout(((2, 2), (2, 4)), ((0, 2), (0, 4)))
    draw_layout(
        cecka_6,
        output / "cecka_broadcast.svg",
        title="((2,2),(2,4)):((0,2),(0,4))",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    draw_layout(
        cecka_6,
        output / "cecka_broadcast_flat.svg",
        title="((2,2),(2,4)):((0,2),(0,4))",
        flatten_hierarchical=True,
    )
    print(f"✓ ((2,2),(2,4)):((0,2),(0,4)) — broadcast (zero-stride) layout")

    # Morton/Z-order layout using blocked_product (CuTe pattern)
    # morton1 = 2x2 Z-order tile
    # morton2 = blocked_product(morton1, morton1) -> 4x4
    # morton3 = blocked_product(morton1, morton2) -> 8x8
    morton1 = Layout((2, 2), (1, 2))
    morton2 = blocked_product(morton1, morton1)
    morton3 = blocked_product(morton1, morton2)
    draw_layout(morton1, output / "hier_morton_2x2.svg", title=f"Morton 2×2: {morton1}")
    draw_layout(morton2, output / "hier_morton_4x4.svg", title=f"Morton 4×4: {morton2}")
    draw_layout(morton3, output / "hier_morton_8x8.svg", title=f"Morton 8×8: {morton3}")
    draw_layout(
        morton3,
        output / "hier_morton_8x8_color.svg",
        title=f"Morton 8×8: {morton3}",
        colorize=True,
    )
    print(f"✓ Morton 2×2: {morton1}")
    print(f"✓ Morton 4×4: {morton2}")
    print(f"✓ Morton 8×8: {morton3}")

    # Show nested mode access
    # Mode 0 is the row dimension with shape (2, 2)
    mode0 = mode(logo_layout, 0)
    draw_layout(mode0, output / "hier_mode0.svg", title=f"Mode 0 (rows): {mode0}")
    print(f"✓ Mode 0 (rows): {mode0}")

    # Mode 1 is the column dimension with shape (2, 2)
    mode1 = mode(logo_layout, 1)
    draw_layout(mode1, output / "hier_mode1.svg", title=f"Mode 1 (cols): {mode1}")
    print(f"✓ Mode 1 (cols): {mode1}")


# =============================================================================
# Section 5: Swizzled Layouts
# =============================================================================


def example_swizzled_layouts(output: Path):
    """Swizzled layouts for GPU shared memory bank conflict avoidance.

    CuTe Reference: Swizzle Functions

    Swizzle applies an XOR operation to indices, redistributing elements
    across memory banks to avoid bank conflicts in GPU shared memory.

    Swizzle(B, M, S):
      - B (bits): number of bits to XOR
      - M (base): starting bit position
      - S (shift): offset for XOR source bits

    Effect: bits at positions [M, M+B) are XORed with bits at [M+S, M+S+B)

    Common patterns:
      - Swizzle(B, 0, 3): Classic LDMATRIX patterns for 8×8 fp16 tiles
      - Swizzle(B, 4, 3): GMMA/TMA patterns (SW32, SW64, SW128)

    Using colorize=True makes the cell movement much clearer.
    """
    print("\n" + "=" * 60)
    print("5. Swizzled Layouts (colorized for clarity)")
    print("=" * 60)

    # =========================================================================
    # Part A: Classic LDMATRIX swizzles - Swizzle(B, 0, 3) family
    # Used for 8×8 fp16 tiles with LDMATRIX instruction
    # =========================================================================
    print("\n  --- Classic LDMATRIX Swizzles: Swizzle(B, 0, 3) ---")

    base_8x8 = Layout((8, 8), (8, 1))

    # Swizzle(3, 0, 3) - 8-bank redistribution (most common)
    sw_303 = Swizzle(3, 0, 3)
    draw_swizzle(base_8x8, sw_303, output / "swizzle_8x8_303.svg", colorize=True)
    print(f"✓ Swizzle(3,0,3) on 8×8: XOR bits [0,3) with [3,6) → 8-bank")

    # Swizzle(2, 0, 3) - 4-bank redistribution
    sw_203 = Swizzle(2, 0, 3)
    draw_swizzle(base_8x8, sw_203, output / "swizzle_8x8_203.svg", colorize=True)
    print(f"✓ Swizzle(2,0,3) on 8×8: XOR bits [0,2) with [3,5) → 4-bank")

    # Swizzle(1, 0, 3) - 2-bank redistribution
    sw_103 = Swizzle(1, 0, 3)
    draw_swizzle(base_8x8, sw_103, output / "swizzle_8x8_103.svg", colorize=True)
    print(f"✓ Swizzle(1,0,3) on 8×8: XOR bit 0 with bit 3 → 2-bank")

    # Column-major variant
    base_8x8_col = Layout((8, 8), (1, 8))
    draw_swizzle(
        base_8x8_col, sw_303, output / "swizzle_8x8_col_303.svg", colorize=True
    )
    print(f"✓ Swizzle(3,0,3) on 8×8 col-major")

    # 16x8 variant (common for tensor core)
    base_16x8 = Layout((16, 8), (8, 1))
    draw_swizzle(base_16x8, sw_303, output / "swizzle_16x8_303.svg", colorize=True)
    print(f"✓ Swizzle(3,0,3) on 16×8 row-major")

    # =========================================================================
    # Part B: GMMA/TMA swizzles - Swizzle(B, 4, 3) family (SM90+)
    # Used for Tensor Memory Accelerator and GMMA operations
    # SW32 = Swizzle(1,4,3), SW64 = Swizzle(2,4,3), SW128 = Swizzle(3,4,3)
    #
    # CuTe canonical shapes (from mma_traits_sm90_gmma.hpp):
    #   The base layout is defined in BITS as (N_bits, 8):(1, N_bits)
    #   For byte-level view, the canonical row-major shapes are:
    #     SW32:  (8, 32):(32, 1)   -- 32 bytes per row, 8 rows
    #     SW64:  (8, 64):(64, 1)   -- 64 bytes per row, 8 rows
    #     SW128: (8, 128):(128, 1) -- 128 bytes per row, 8 rows
    #   The column count must be 2^(base+bits) for the swizzle pattern
    #   to fully manifest as within-row element permutations.
    # =========================================================================
    print("\n  --- GMMA/TMA Swizzles: Swizzle(B, 4, 3) ---")

    # Swizzle(1, 4, 3) - SW32 (32-byte swizzle width)
    # Canonical byte layout: 8 rows × 32 columns (32 bytes per row)
    sw_143 = Swizzle(1, 4, 3)
    base_8x32 = Layout((8, 32), (32, 1))
    draw_swizzle(base_8x32, sw_143, output / "swizzle_8x32_143_SW32.svg", colorize=True)
    print(f"✓ Swizzle(1,4,3) SW32 on 8×32: XOR bit 4 with bit 7")

    # Swizzle(2, 4, 3) - SW64 (64-byte swizzle width)
    # Canonical byte layout: 8 rows × 64 columns (64 bytes per row)
    sw_243 = Swizzle(2, 4, 3)
    base_8x64 = Layout((8, 64), (64, 1))
    draw_swizzle(base_8x64, sw_243, output / "swizzle_8x64_243_SW64.svg", colorize=True)
    print(f"✓ Swizzle(2,4,3) SW64 on 8×64: XOR bits [4,6) with [7,9)")

    # Swizzle(3, 4, 3) - SW128 (128-byte swizzle width, maximum bandwidth)
    # Canonical byte layout: 8 rows × 128 columns (128 bytes per row)
    sw_343 = Swizzle(3, 4, 3)
    base_8x128 = Layout((8, 128), (128, 1))
    draw_swizzle(
        base_8x128, sw_343, output / "swizzle_8x128_343_SW128.svg", colorize=True
    )
    print(f"✓ Swizzle(3,4,3) SW128 on 8×128: XOR bits [4,7) with [7,10)")

    # =========================================================================
    # Part C: No swizzle (identity) for comparison
    # =========================================================================
    print("\n  --- No Swizzle (Identity) ---")

    # Swizzle(0, M, S) is identity - no XOR applied
    sw_043 = Swizzle(0, 4, 3)
    draw_swizzle(
        base_8x128, sw_043, output / "swizzle_8x128_043_none.svg", colorize=True
    )
    print(f"✓ Swizzle(0,4,3) on 8×128: Identity (no XOR)")


# =============================================================================
# Section 6: Thread-Value (TV) Layouts
# =============================================================================


def example_thread_value_layouts(output: Path):
    """Thread-Value (TV) layouts for GPU parallelism.

    CuTe Reference: Thread-Value Layouts (MMA/Copy atoms)

    TV layouts describe how data is distributed across threads and values
    (registers per thread). Shape is (Threads, Values), showing which
    thread owns which elements.

    Each cell is labeled "TxVy" showing thread index x and value index y.
    """
    print("\n" + "=" * 60)
    print("6. Thread-Value (TV) Layouts")
    print("=" * 60)

    # Simple TV layout: 4 threads, 2 values each = 8 elements
    # Thread 0: V0, V1; Thread 1: V0, V1; etc.
    tv_4x2 = Layout((4, 2), (2, 1))
    draw_tv_layout(
        tv_4x2,
        output / "tv_4threads_2values.svg",
        title="TV: (4,2):(2,1) - 4 threads, 2 values each",
    )
    draw_tv_layout(
        tv_4x2,
        output / "tv_4threads_2values_color.svg",
        title="TV: (4,2):(2,1)",
        colorize=True,
    )
    print(f"✓ TV Layout 4×2: 4 threads, 2 values each")
    print(f"  Thread t owns values V0, V1 at offsets 2*t and 2*t+1")

    # TV layout with interleaved threads
    tv_4x2_col = Layout((4, 2), (1, 4))
    draw_tv_layout(
        tv_4x2_col,
        output / "tv_4threads_2values_interleaved.svg",
        title="TV interleaved: (4,2):(1,4)",
    )
    print(f"✓ TV Layout 4×2 interleaved: offsets t and t+4")

    # 8x4 TV layout (smaller than full warp for clarity)
    tv_8x4 = Layout((8, 4), (4, 1))
    draw_tv_layout(
        tv_8x4, output / "tv_8x4.svg", title="TV: (8,4):(4,1) - 8 threads, 4 values"
    )
    draw_tv_layout(
        tv_8x4, output / "tv_8x4_color.svg", title="TV: (8,4):(4,1)", colorize=True
    )
    print(f"✓ TV Layout 8×4: 8 threads, 4 values each")

    # 8x8 TV layout (common for LDMATRIX)
    tv_8x8 = Layout((8, 8), (8, 1))
    draw_tv_layout(
        tv_8x8, output / "tv_8x8.svg", title="TV: (8,8):(8,1) - 8 threads, 8 values"
    )
    draw_tv_layout(
        tv_8x8, output / "tv_8x8_color.svg", title="TV: (8,8):(8,1)", colorize=True
    )
    print(f"✓ TV Layout 8×8: 8 threads, 8 values each (LDMATRIX style)")

    # Also show the regular layout view for comparison
    draw_layout(
        tv_8x8,
        output / "tv_8x8_offsets.svg",
        title="TV: (8,8):(8,1) - Memory offsets view",
    )
    print(f"  (Also showing memory offset view for comparison)")


# =============================================================================
# Section 7: Copy Atom Traits (LDMATRIX, STMATRIX, TMA)
# =============================================================================


def example_copy_atoms(output: Path):
    """Copy atom TV layouts across GPU architectures.

    Sources:
      copy_traits_sm75.hpp  — SM75 LDSM (ldmatrix)
      copy_traits_sm90.hpp  — SM90 STSM (stmatrix)
      copy_traits_sm90_tma.hpp — SM90 TMA (Tensor Memory Accelerator)
      mma_traits_sm90_gmma.hpp — SM90 GMMA shared memory layouts

    Copy traits define Src and Dst layouts in *bit* coordinates.
    upcast(layout, element_bits) converts to element-level TV layouts.

    TMA atoms are single-threaded bulk transfers between global and shared
    memory.  Their interesting aspect is the shared memory swizzle patterns.
    """
    print("\n" + "=" * 60)
    print("7. Copy Atom Traits")
    print("=" * 60)

    element_bits = 16  # fp16

    # =====================================================================
    # SM75 LDMATRIX — ldmatrix.sync.aligned.m8n8.shared.b16
    # =====================================================================
    print("\n  --- SM75 LDMATRIX (ldmatrix.sync.aligned) ---")

    ldsm_atoms = [
        SM75_U32x1_LDSM_N,
        SM75_U32x4_LDSM_N,
        SM75_U16x2_LDSM_T,
        SM75_U16x4_LDSM_T,
        SM75_U16x8_LDSM_T,
    ]
    for atom in ldsm_atoms:
        # draw_copy_atom handles upcast from bit to element coords automatically
        draw_copy_atom(
            atom, element_bits=element_bits, filename=output / f"{atom.name}_copy.svg"
        )

        dst = upcast(atom.dst_layout_bits, element_bits)
        n_thr = size(atom.thr_id)
        n_val = size(mode(dst, 1))
        print(f"✓ {atom.name}  Dst: {dst}  ({n_thr} thr × {n_val} val)")

    # =====================================================================
    # SM90 STMATRIX — stmatrix.sync.aligned.m8n8.shared.b16
    # Inverse of SM75 LDMATRIX: STSM Src = LDSM Dst, STSM Dst = LDSM Src
    # =====================================================================
    print("\n  --- SM90 STMATRIX (stmatrix.sync.aligned) ---")

    stsm_atoms = [SM90_U32x4_STSM_N, SM90_U16x8_STSM_T]
    for atom in stsm_atoms:
        draw_copy_atom(
            atom, element_bits=element_bits, filename=output / f"{atom.name}_copy.svg"
        )
        print(f"✓ {atom.name}  ({atom.ptx})")

    # =====================================================================
    # SM90 TMA — Tensor Memory Accelerator
    # From copy_traits_sm90_tma.hpp
    # TMA atoms are single-threaded (ThrID = Layout<_1>).
    # The interesting aspect is the GMMA shared memory swizzle patterns
    # that TMA writes into, defined in mma_traits_sm90_gmma.hpp.
    # =====================================================================
    print("\n  --- SM90 TMA (Tensor Memory Accelerator) ---")
    print("  TMA atoms: ThrID = Layout<_1> (single-threaded bulk transfer)")
    print("  SM90_TMA_LOAD:  global → shared  (Layout<_1, NumBitsPerTMA>)")
    print("  SM90_TMA_STORE: shared → global  (Layout<_1, NumBitsPerTMA>)")
    print("  SM90_TMA_LOAD_MULTICAST: with CTA multicast")

    # TMA writes into swizzled shared memory.  The canonical GMMA smem
    # layouts are the interesting part to visualize.
    # These come from mma_traits_sm90_gmma.hpp:
    #   Layout_K_SW128_Atom_Bits = Swizzle<3,4,3> ∘ (8, 1024):(1024, 1)
    #   For fp16: Swizzle<3,4,3> ∘ (8, 64):(64, 1)  = 8 rows × 64 cols
    print("\n  TMA target: GMMA K-major SW128 smem layout (fp16):")
    base_tma = Layout((8, 64), (64, 1))
    draw_swizzle(
        base_tma, Swizzle(3, 4, 3), output / "SM90_TMA_GMMA_K_SW128.svg", colorize=True
    )
    print(f"✓ SM90 TMA → GMMA K-major SW128: Swizzle(3,4,3) ∘ (8,64):(64,1)")

    print("\n  TMA target: GMMA M|N-major SW128 smem layout (fp16):")
    base_tma_mn = Layout((64, 8), (1, 64))
    draw_swizzle(
        base_tma_mn,
        Swizzle(3, 4, 3),
        output / "SM90_TMA_GMMA_MN_SW128.svg",
        colorize=True,
    )
    print(f"✓ SM90 TMA → GMMA M|N-major SW128: Swizzle(3,4,3) ∘ (64,8):(1,64)")

    # =====================================================================
    # LDMATRIX shared memory + swizzle (classic pattern)
    # =====================================================================
    print("\n  --- LDMATRIX Shared Memory with Swizzle ---")
    smem_8x8 = Layout((8, 8), (8, 1))
    draw_swizzle(
        smem_8x8, Swizzle(3, 0, 3), output / "ldmatrix_smem_swizzle.svg", colorize=True
    )
    print(f"✓ LDMATRIX shared memory with Swizzle(3,0,3)")


# =============================================================================
# Section 8: MMA Atom Traits
# =============================================================================


def _draw_mma_atom(atom, output: Path):
    """Draw A, B, C, and combined figures for one MMA atom."""
    name = atom.name
    M, N, K = atom.shape_mnk
    thr = atom.thr_id

    # For atoms with broadcast (stride-0), cosize < M*K. Use cosize-based
    # grid dimensions so every cell maps to a thread — no '?' cells.
    cs_a = cosize(atom.a_layout)
    cs_b = cosize(atom.b_layout)
    cs_c = cosize(atom.c_layout)
    a_rows, a_cols = M, cs_a // M if cs_a % M == 0 else K
    b_rows, b_cols = cs_b // N if cs_b % N == 0 else K, N
    c_rows, c_cols = M, cs_c // M if cs_c % M == 0 else N

    draw_tv_layout(
        atom.a_layout,
        output / f"{name}_A.svg",
        title=f"{name}  A ({a_rows}×{a_cols})",
        colorize=True,
        grid_shape=(a_rows, a_cols),
        thr_id_layout=thr,
    )

    draw_tv_layout(
        atom.b_layout,
        output / f"{name}_B.svg",
        title=f"{name}  B ({b_rows}×{b_cols})",
        colorize=True,
        grid_shape=(b_rows, b_cols),
        thr_id_layout=thr,
        col_major=False,
    )

    draw_tv_layout(
        atom.c_layout,
        output / f"{name}_C.svg",
        title=f"{name}  C ({c_rows}×{c_cols})",
        colorize=True,
        grid_shape=(c_rows, c_cols),
        thr_id_layout=thr,
    )

    draw_mma_layout(
        atom.a_layout,
        atom.b_layout,
        atom.c_layout,
        output / f"{name}_combined.svg",
        tile_mnk=(a_rows, c_cols, a_cols),
        main_title=name,
        colorize=True,
        thr_id_layout=thr,
    )

    n_thr = size(mode(atom.c_layout, 0))
    n_val_a = size(mode(atom.a_layout, 1))
    n_val_b = size(mode(atom.b_layout, 1))
    n_val_c = size(mode(atom.c_layout, 1))
    print(f"✓ {name}")
    print(f"    {atom.ptx}")
    print(f"    {n_thr} threads, A:{n_val_a} B:{n_val_b} C:{n_val_c} vals/thr")


def _draw_tiled_mma(atom, atom_layout, output: Path, tile_mnk=None):
    """Draw a TiledMMA: atom tiled across multiple quadpairs.

    Equivalent to the C++ code:
        TiledMMA mma = make_tiled_mma(atom, atom_layout, Tile<M,N,K>);
        print_latex(mma);

    Args:
        atom: MMAAtom to tile
        atom_layout: Layout describing atom arrangement in M×N space
                     E.g. Layout((2,2), (2,1)) for 2×2 n-major
        output: Output directory
        tile_mnk: Optional (M, N, K) final tile. If larger than the atom
                  arrangement, replicates across values.
    """
    from tensor_layouts.layout_utils import tile_mma_grid

    M_a, N_a, K_a = atom.shape_mnk
    atom_shape = atom_layout.shape
    if isinstance(atom_shape, int):
        n_am, n_an = atom_shape, 1
    else:
        n_am, n_an = atom_shape

    label = f"{atom.name}_{n_am}x{n_an}"
    M, N, K = M_a * n_am, N_a * n_an, K_a
    if tile_mnk is not None:
        M, N, K = tile_mnk
        label = f"{atom.name}_{n_am}x{n_an}_{M}x{N}x{K}"

    # Compute tiled grids
    c_grid, _ = tile_mma_grid(atom, atom_layout, "C", tile_mnk=tile_mnk)
    a_grid, _ = tile_mma_grid(atom, atom_layout, "A", tile_mnk=tile_mnk)
    b_grid, _ = tile_mma_grid(atom, atom_layout, "B", tile_mnk=tile_mnk)

    draw_tiled_grid(
        c_grid, M, N, output / f"{label}_C.svg", title=f"{label}  C ({M}×{N})"
    )
    draw_tiled_grid(
        a_grid, M, K, output / f"{label}_A.svg", title=f"{label}  A ({M}×{K})"
    )
    # B displayed as K×N (transposed)
    b_display = {}
    for (r, c), val in b_grid.items():
        phys_t, v, logical_t = val
        n_coord = r
        k_coord = c
        b_display[(k_coord, n_coord)] = val
    draw_tiled_grid(
        b_display, K, N, output / f"{label}_B.svg", title=f"{label}  B ({K}×{N})"
    )

    # Combined figure: A (left), B (top-right), C (bottom-right)
    draw_combined_mma_grid(
        a_grid,
        b_display,
        c_grid,
        M,
        N,
        K,
        output / f"{label}_combined.svg",
        title=label,
    )

    print(f"✓ Tiled MMA: {label}")
    print(
        f"    {size(atom_layout)} atoms ({n_am}×{n_an}), "
        f"tile {M}×{N}×{K}, "
        f"{size(atom_layout) * size(mode(atom.c_layout, 0))} threads"
    )


def example_mma_atom(output: Path):
    """MMA atom layouts across architectures.

    Atom definitions are imported from atoms_nv.py, which mirrors the C++
    MMA_Traits structs.  Each atom has separate a_layout, b_layout, c_layout.

    Reference: 0t_mma_atom.md in the CuTe documentation.
    """
    print("\n" + "=" * 60)
    print("8. MMA Atom Traits")
    print("=" * 60)

    # SM70 Volta — verified against 0t_mma_atom.md / HMMA.8x8x4.NT_Atom.png
    # Only the F32 accumulator variant has a reference image to verify against.
    print("\n  --- SM70 Volta (8 threads, quadpair) ---")
    for atom in [SM70_8x8x4_F32F16F16F32_NT]:
        _draw_mma_atom(atom, output)

    # Tiled MMA: 2×2 n-major atom layout → 16×16×4 tile using full warp
    # Equivalent to C++:
    #   make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
    #                  Layout<Shape<_2,_2>, Stride<_2,_1>>{});
    # Reference: HMMA.8x8x4.NT_2x2.png
    _draw_tiled_mma(SM70_8x8x4_F32F16F16F32_NT, Layout((2, 2), (2, 1)), output)

    # Tiled MMA expanded to 32×32×4 via value tiling
    # Equivalent to C++:
    #   make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
    #                  Layout<Shape<_2,_2>, Stride<_2,_1>>{},
    #                  Tile<_32,_32,_4>{});
    # Reference: HMMA.8x8x4.NT_2x2_32x32x4.png
    _draw_tiled_mma(
        SM70_8x8x4_F32F16F16F32_NT, Layout((2, 2), (2, 1)), output, tile_mnk=(32, 32, 4)
    )

    # SM80 Ampere
    print("\n  --- SM80 Ampere (32 threads, warp) ---")
    for atom in [SM80_16x8x8_F16F16F16F16_TN, SM80_16x8x16_F16F16F16F16_TN]:
        _draw_mma_atom(atom, output)

    # SM90 Hopper warp-level
    print("\n  --- SM90 Hopper warp-level (32 threads) ---")
    for atom in [SM90_16x8x4_F64F64F64F64_TN]:
        _draw_mma_atom(atom, output)

    # SM90 GMMA accumulators (128 threads)
    print("\n  --- SM90 Hopper GMMA (128 threads, warpgroup) ---")
    for atom in [SM90_64x8x16_F16F16F16_SS, SM90_64x64x16_F16F16F16_SS]:
        M, N, K = atom.shape_mnk
        draw_tv_layout(
            atom.c_layout,
            output / f"{atom.name}_C.svg",
            title=f"{atom.name}  C ({M}×{N})",
            colorize=True,
            grid_shape=(M, N),
        )
        n_vals = size(mode(atom.c_layout, 1))
        print(f"✓ {atom.name}  C: 128 thr × {n_vals} vals = {128*n_vals} elements")

    # SM75 Turing — 32 threads, first-gen asymmetric tiles
    print("\n  --- SM75 Turing (32 threads, warp) ---")
    for atom in [SM75_16x8x8_F32F16F16F32_TN]:
        _draw_mma_atom(atom, output)

    # SM89 Ada Lovelace — FP8 tensor cores
    print("\n  --- SM89 Ada Lovelace (32 threads, warp, FP8) ---")
    for atom in [SM89_16x8x32_F32E4M3E4M3F32_TN]:
        _draw_mma_atom(atom, output)

    # SM100 Blackwell UMMA — 1 "thread" (warp group), TMEM accumulator
    # SM100 uses a fundamentally different model from SM90: instead of
    # distributing elements across 128 threads with a hierarchical layout,
    # all elements live in TMEM (Tensor Memory) accessible by the entire
    # warp group as a single unit. The layout is trivial col-major.
    #
    # Side-by-side comparison: SM90 GMMA vs SM100 UMMA for 64×8 C accumulator
    print("\n  --- SM100 Blackwell UMMA (1 thread, warp group) ---")
    atom = SM100_128x128x16_F16F16F16_SS
    M, N, K = atom.shape_mnk
    print(f"✓ {atom.name}  C: {atom.c_layout}  ({M*N} elements)")

    # Compare: SM90 GMMA 64×8 C uses 128 threads with hierarchical layout
    sm90_atom = SM90_64x8x16_F16F16F16_SS
    sm90_c = sm90_atom.c_layout
    draw_tv_layout(
        sm90_c,
        output / "SM100_compare_SM90_64x8_C.svg",
        title="SM90 GMMA  C (64×8) — 128 threads",
        colorize=True,
        grid_shape=(64, 8),
    )
    print(f"✓ SM90 GMMA 64×8 C: {sm90_c}  (128 thr × {size(mode(sm90_c, 1))} vals)")

    # SM100 UMMA 64×8 C — same logical tile, 1 "thread", all values
    umma_atom = make_umma_atom_ss(64, 8)
    umma_c = umma_atom.c_layout
    draw_layout(
        umma_c,
        output / "SM100_compare_UMMA_64x8_C.svg",
        title="SM100 UMMA  C (64×8) — 1 thread, TMEM",
        flatten_hierarchical=True,
    )
    print(f"✓ SM100 UMMA 64×8 C: {umma_c}  (1 thr × {size(umma_c)} vals)")

    # SM120 Blackwell B200 — warp-level FP8
    print("\n  --- SM120 Blackwell B200 (32 threads, warp, FP8) ---")
    for atom in [SM120_16x8x32_F32E4M3E4M3F32_TN]:
        _draw_mma_atom(atom, output)

    # AMD CDNA3 MFMA — 64 threads (full wavefront)
    print("\n  --- AMD CDNA3 MFMA (64 threads, wavefront) ---")
    for atom in [CDNA3_32x32x16_F32F8F8_MFMA, CDNA3_16x16x32_F32F8F8_MFMA]:
        _draw_mma_atom(atom, output)

    # AMD CDNA3+ (MI350) — 64 threads
    print("\n  --- AMD CDNA3+ (MI350) MFMA (64 threads) ---")
    for atom in [CDNA3P_16x16x32_F32F16F16_MFMA]:
        _draw_mma_atom(atom, output)


# =============================================================================
# Section 9: Slicing Examples
# =============================================================================


def example_slicing(output: Path):
    """Slicing layouts - row, column, and complex discontinuous slices.

    CuTe Reference: Layout Algebra - Slicing

    Slicing extracts a subset of elements from a layout, producing a
    new layout that describes the selected elements.
    """
    print("\n" + "=" * 60)
    print("9. Slicing Examples")
    print("=" * 60)

    # Base layout for slicing examples: 8x8 row-major
    base = Layout((8, 8), (8, 1))

    # Row slice: select row 3 (all columns)
    draw_slice(
        base, (3, None), output / "slice_row3.svg", title="Row Slice: layout(3, :)"
    )
    print(f"✓ Row slice: layout(3, :)")
    print(f"  Selects all 8 elements in row 3")

    # Column slice: select column 5 (all rows)
    draw_slice(
        base, (None, 5), output / "slice_col5.svg", title="Column Slice: layout(:, 5)"
    )
    print(f"✓ Column slice: layout(:, 5)")
    print(f"  Selects all 8 elements in column 5")

    # Single element
    draw_slice(
        base, (4, 6), output / "slice_element.svg", title="Single Element: layout(4, 6)"
    )
    print(f"✓ Single element: layout(4, 6)")

    # Rectangular region: rows 2-5, columns 1-4
    draw_slice(
        base,
        (slice(2, 6), slice(1, 5)),
        output / "slice_rect.svg",
        title="Rectangular: layout[2:6, 1:5]",
    )
    print(f"✓ Rectangular region: layout[2:6, 1:5]")
    print(f"  Selects 4×4 = 16 elements")

    # Multiple disjoint rows using logical_divide
    # First, show the divided layout
    print(f"\n  Complex Discontinuous Slices:")

    # Divide 8 rows into 4 groups of 2
    divided = logical_divide(base, Layout((2, 4), (1, 2)))
    draw_layout(
        divided, output / "slice_divided_base.svg", title="Divided: 2-row groups"
    )
    print(f"✓ Divided layout: groups of 2 rows")

    # Tile-based slicing: select every other 2x2 tile
    tiled = Layout(((2, 4), (2, 4)), ((1, 16), (2, 8)))
    draw_layout(
        tiled, output / "slice_tiled.svg", title="Tiled: ((2,4),(2,4)):((1,16),(2,8))"
    )
    print(f"✓ Tiled layout: 2×2 tiles in 8×8")

    # Strided row access (every other row)
    strided_rows = Layout((4, 8), (16, 1))
    draw_layout(
        strided_rows,
        output / "slice_strided_rows.svg",
        title="Strided Rows: (4,8):(16,1)",
    )
    print(f"✓ Strided rows: every other row")

    # Strided column access (every other column)
    strided_cols = Layout((8, 4), (8, 2))
    draw_layout(
        strided_cols,
        output / "slice_strided_cols.svg",
        title="Strided Cols: (8,4):(8,2)",
    )
    print(f"✓ Strided columns: every other column")

    # Diagonal-like pattern using hierarchical layout
    # Access elements (0,0), (1,1), (2,2), (3,3), ...
    diag = Layout(8, 9)  # stride 9 = 8+1 gives diagonal
    draw_layout(diag, output / "slice_diagonal.svg", title="Diagonal: 8:9")
    print(f"✓ Diagonal access: stride 9 (row_stride + 1)")

    # =========================================================================
    # Hierarchical slicing from Cecka, "CuTe Layout Representation and Algebra"
    # Tensor: ((3,2),((2,3),2)):((4,1),((2,15),100))
    # =========================================================================
    print(f"\n  --- Cecka Hierarchical Slicing ---")
    cecka_t = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
    draw_layout(
        cecka_t,
        output / "cecka_slice_base.svg",
        title=f"Tensor: {cecka_t}",
        flatten_hierarchical=True,
    )
    draw_layout(
        cecka_t,
        output / "cecka_slice_base_nested.svg",
        title=f"Tensor: {cecka_t}",
        flatten_hierarchical=False,
        label_hierarchy_levels=True,
    )
    print(f"✓ Base tensor: {cecka_t}")

    # Slice (2, None) — fix mode-0 to flat index 2, keep all of mode-1
    draw_slice(cecka_t, (2, None), output / "cecka_slice_2_None.svg", title="(2,:)")
    print(f"✓ Slice (2, :) — fix row to 2")

    # Slice (None, 5) — keep all of mode-0, fix mode-1 to flat index 5
    draw_slice(cecka_t, (None, 5), output / "cecka_slice_None_5.svg", title="(:,5)")
    print(f"✓ Slice (:, 5) — fix col to 5")

    # Slice (2, ((0,None),None)) — fix mode-0 to 2, partially slice mode-1
    draw_slice(
        cecka_t,
        (2, ((0, None), None)),
        output / "cecka_slice_2_0NN.svg",
        title="(2,((0,:),:))",
    )
    print(f"✓ Slice (2, ((0,:),:)) — fix row=2, inner-col-0=0, rest free")

    # Slice ((None,1),(None,0)) — fix outer-row=1, inner-col-outer=0
    draw_slice(
        cecka_t,
        ((None, 1), (None, 0)),
        output / "cecka_slice_N1_N0.svg",
        title="((:,1),(:,0))",
    )
    print(f"✓ Slice ((:,1), (:,0)) — outer-row=1, mode-1 partially fixed")

    # Slice ((None,0),((0,None),1)) — outer-row=0, inner-col-0=0, outer-col=1
    draw_slice(
        cecka_t,
        ((None, 0), ((0, None), 1)),
        output / "cecka_slice_N0_0N1.svg",
        title="((:,0),((0,:),1))",
    )
    print(f"✓ Slice ((:,0), ((0,:),1)) — outer-row=0, inner-0=0, outer-col=1")

    # Slice ((1,None),((None,0),None)) — inner-row=1, middle-col=0
    draw_slice(
        cecka_t,
        ((1, None), ((None, 0), None)),
        output / "cecka_slice_1N_N0N.svg",
        title="((1,:),((:,0),:))",
    )
    print(f"✓ Slice ((1,:), ((:,0),:)) — inner-row=1, middle-col=0")


# =============================================================================
# Section 10: Layout Algebra Operations
# =============================================================================


def example_algebra_operations(output: Path):
    """Layout algebra operations: compose, complement, divide, product.

    CuTe Reference: Layout Algebra

    These operations combine layouts to create more complex access patterns.
    """
    print("\n" + "=" * 60)
    print("10. Layout Algebra Operations")
    print("=" * 60)

    # Composition: compose two layouts
    # outer(inner(i)) - chains two mappings
    inner = Layout((4, 2), (1, 4))
    outer = Layout(8, 2)
    composed = compose(outer, inner)
    draw_layout(inner, output / "algebra_inner.svg", title=f"Inner: {inner}")
    draw_layout(
        composed,
        output / "algebra_composed.svg",
        title=f"Composed: compose({outer}, {inner})",
    )
    print(f"✓ Composition: compose({outer}, {inner}) = {composed}")

    # Complement: find layout that covers remaining indices
    base = Layout((4, 2), (2, 1))
    comp = complement(base, 16)
    draw_layout(base, output / "algebra_base.svg", title=f"Base: {base}")
    draw_layout(
        comp,
        output / "algebra_complement.svg",
        title=f"Complement: complement({base}, 16)",
    )
    print(f"✓ Complement: complement({base}, 16) = {comp}")

    # Logical divide: tile a layout
    matrix = Layout((8, 8), (8, 1))
    tiler = Layout((2, 2), (1, 2))
    divided = logical_divide(matrix, tiler)
    draw_layout(matrix, output / "algebra_matrix.svg", title=f"Matrix: {matrix}")
    draw_layout(
        divided,
        output / "algebra_divided.svg",
        title=f"Divided: logical_divide by {tiler}",
    )
    print(f"✓ Logical divide: 8×8 by 2×2 tiler")

    # Logical product: replicate a layout
    tile = Layout((2, 2), (2, 1))
    grid = Layout((4, 4), (1, 4))
    product = logical_product(tile, grid)
    draw_layout(tile, output / "algebra_tile.svg", title=f"Tile: {tile}")
    draw_layout(
        product,
        output / "algebra_product.svg",
        title=f"Product: logical_product({tile}, {grid})",
    )
    print(f"✓ Logical product: {tile} × {grid}")

    # Rank >= 3 results: flat_divide and flat_product produce rank-3 layouts
    # that are now automatically rendered as multi-panel 2D grids
    fd = flat_divide(matrix, Layout(2, 1))
    draw_layout(
        fd,
        output / "algebra_flat_divide.svg",
        title=f"flat_divide result (rank {rank(fd)})",
    )
    print(f"✓ flat_divide: shape={fd.shape}, rank={rank(fd)} → multi-panel")

    fp = flat_product(Layout((2, 2), (1, 2)), Layout(4, 1))
    draw_layout(
        fp,
        output / "algebra_flat_product.svg",
        title=f"flat_product result (rank {rank(fp)})",
    )
    print(f"✓ flat_product: shape={fp.shape}, rank={rank(fp)} → multi-panel")


# =============================================================================
# Section 11: Tensor Slicing with Visualization
# =============================================================================


def example_tensor_slicing(output: Path):
    """Tensor slicing - visualizing tensors directly.

    CuTe Reference: Tensor Operations

    Tensors combine an offset with a layout. draw_layout accepts Tensors
    directly, showing offset-adjusted values in each cell.
    """
    print("\n" + "=" * 60)
    print("11. Tensor Visualization")
    print("=" * 60)

    # Create a tensor with a base offset — cells show offset-adjusted values
    layout = Layout((4, 8), (8, 1))
    tensor = Tensor(layout, offset=0)
    draw_layout(tensor, output / "tensor_base.svg")
    print(f"✓ Base tensor (offset=0): {tensor}")

    # Non-zero offset shifts all values
    tensor_16 = Tensor(layout, offset=16)
    draw_layout(tensor_16, output / "tensor_offset16.svg")
    print(f"✓ Offset tensor (offset=16): cell (0,0) = {tensor_16(0, 0)}")

    # Side-by-side comparison using draw_composite
    draw_composite(
        [tensor, tensor_16],
        output / "tensor_offset_compare.svg",
        titles=["offset=0", "offset=16"],
        main_title="Tensor Offset Comparison",
    )
    print(f"✓ Tensor comparison: tensor_offset_compare.svg")

    # Swizzled tensor — swizzle applied to total linear offset
    sw = Swizzle(3, 0, 3)
    sw_layout = Layout((8, 8), (8, 1), swizzle=sw)
    sw_tensor = Tensor(sw_layout, offset=0)
    draw_layout(sw_tensor, output / "tensor_swizzled.svg")
    print(f"✓ Swizzled tensor: {sw_tensor}")

    # Tensor slicing: fixing coordinates adjusts the offset
    row2 = tensor[2, :]
    print(f"\n  tensor[2, :] = {row2}")
    print(f"  tensor[2, :](0) = {row2(0)}, tensor(2, 0) = {tensor(2, 0)}")
    draw_layout(row2, output / "tensor_slice_row2.svg", title=f"tensor[2, :] = {row2}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main(output_dir: str = "examples_output"):
    """Run all visualization examples."""
    output = setup_output_dir(output_dir)

    print("=" * 70)
    print("CuTe Layout Visualization Examples")
    print("Based on NVIDIA CuTe C++ Documentation")
    print("=" * 70)

    # Run all examples
    example_output_formats(output)
    example_1d_layouts(output)
    example_2d_layouts(output)
    example_hierarchical_layouts(output)
    example_swizzled_layouts(output)
    example_thread_value_layouts(output)
    example_copy_atoms(output)
    example_mma_atom(output)
    example_slicing(output)
    example_algebra_operations(output)
    example_tensor_slicing(output)

    print("\n" + "=" * 70)
    print(f"✓ All examples completed!")
    print(f"  Output directory: {output}")
    print("=" * 70)

    # Summary of files by format
    svg_files = list(output.glob("*.svg"))
    png_files = list(output.glob("*.png"))
    pdf_files = list(output.glob("*.pdf"))
    print(f"\nGenerated files:")
    print(f"  SVG: {len(svg_files)} files (vector)")
    print(f"  PNG: {len(png_files)} files (raster)")
    print(f"  PDF: {len(pdf_files)} files (print)")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "examples_output"
    main(output_dir)
