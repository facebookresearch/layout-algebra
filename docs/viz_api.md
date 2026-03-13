# Visualization API

This document covers the `layout_algebra.viz` module for drawing layouts,
swizzle comparisons, MMA atoms, and more.

Requires: `pip install layout-algebra[viz]` (adds matplotlib).

For runnable examples see [`examples/viz.py`](../examples/viz.py)
and the Jupyter notebook [`examples/viz.ipynb`](../examples/viz.ipynb).
For the core layout algebra see [`docs/layout_api.md`](layout_api.md).

## Output and Display

Every `draw_*` function accepts a `filename` parameter:

| `filename` | Behavior |
|------------|----------|
| `None` (default) | Display inline in Jupyter, or `plt.show()` |
| `"out.svg"` | Save as SVG (vector) |
| `"out.png"` | Save as PNG (raster) at specified `dpi` |
| `"out.pdf"` | Save as PDF (vector) |

The `show_*` functions always display inline and return the matplotlib
`Figure` for further customization.

## draw_layout

Draw a single layout as a grid of cells showing memory offsets.

```python
from layout_algebra import Layout
from layout_algebra.viz import draw_layout

draw_layout(Layout((8, 8), (8, 1)), title="Row-Major 8x8", colorize=True)
```

![draw_layout](images/draw_layout.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | `Layout` | required | The layout to draw |
| `filename` | `str` | `None` | Output path (format from extension) |
| `title` | `str` | `None` | Title above the grid |
| `dpi` | `int` | `150` | Resolution for raster output |
| `figsize` | `(w, h)` | auto | Figure size in inches |
| `colorize` | `bool` | `False` | Use rainbow colors (by cell value) |
| `color_layout` | `Layout` | `None` | Custom coloring (see below) |
| `num_shades` | `int` | `8` | Number of distinct grayscale shades |
| `flatten_hierarchical` | `bool` | `True` | Flatten nested shapes to 2D grid |

### Coloring

By default cells are shaded in grayscale by their offset value.  Use
`colorize=True` for rainbow colors.

The `color_layout` parameter gives fine control over which cells share
colors. It is evaluated in the same logical coordinate space as the layout
being drawn, so displayed cell `(row, col)` is colored by the value of
`color_layout` at that logical coordinate. For ordinary 2D layouts, this
means `color_layout` should usually have the same shape as the layout being
drawn:

```python
layout = Layout((8, 8), (8, 1))

# Color by row (same row = same color)
draw_layout(layout, color_layout=Layout((8, 8), (1, 0)), colorize=True)

# Color by column (same column = same color)
draw_layout(layout, color_layout=Layout((8, 8), (0, 1)), colorize=True)

# Uniform color (no variation)
draw_layout(layout, color_layout=Layout(1, 0))
```

| By row | By column | Uniform |
|--------|-----------|---------|
| ![by row](images/color_by_row.png) | ![by col](images/color_by_col.png) | ![uniform](images/color_uniform.png) |

### Hierarchical Layouts

When `flatten_hierarchical=True` (default), nested shapes are flattened
to a 2D grid.  Set it to `False` to show tile boundaries:

```python
hier = Layout(((2, 3), (2, 4)), ((1, 6), (2, 12)))
draw_layout(hier, flatten_hierarchical=False, title="With tile boundaries")
```

![hierarchical](images/hierarchical.png)

## draw_swizzle

Side-by-side comparison of a linear layout and its swizzled version.

```python
from layout_algebra import Layout, Swizzle
from layout_algebra.viz import draw_swizzle

draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), colorize=True)
```

![draw_swizzle](images/draw_swizzle.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_layout` | `Layout` | required | The linear (un-swizzled) layout |
| `swizzle` | `Swizzle` | required | The swizzle to apply |
| `filename` | `str` | `None` | Output path |
| `dpi` | `int` | `150` | Resolution |
| `figsize` | `(w, h)` | auto | Figure size |
| `colorize` | `bool` | `False` | Rainbow colors |
| `num_shades` | `int` | `8` | Grayscale shades |

## draw_tv_layout

Draw a Thread-Value layout with T (thread ID) and V (value index) labels
in each cell.  Used for visualizing how GPU threads map to matrix elements.

```python
from layout_algebra.atoms_nv import *
from layout_algebra.viz import draw_tv_layout

atom = SM80_16x8x16_F16F16F16F16_TN
draw_tv_layout(atom.c_layout, title="SM80 16x8x16 C (Thread-Value)", colorize=True)
```

![draw_tv_layout](images/draw_tv_layout.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | `Layout` | required | Thread-value layout `(thread, value)` |
| `filename` | `str` | `None` | Output path |
| `title` | `str` | `None` | Title |
| `dpi` | `int` | `150` | Resolution |
| `figsize` | `(w, h)` | auto | Figure size |
| `colorize` | `bool` | `False` | Rainbow colors |
| `num_threads` | `int` | `None` | Override thread count for coloring |
| `grid_shape` | `(r, c)` | `None` | Force a specific grid shape |
| `thr_id_layout` | `Layout` | `None` | Custom thread-ID-to-color mapping |
| `col_major` | `bool` | `True` | Column-major grid ordering |

## draw_mma_layout

Draw an MMA atom's A, B, and C matrices in standard MMA arrangement
(B top-right, A bottom-left, C bottom-right).

```python
from layout_algebra.atoms_nv import *
from layout_algebra.viz import draw_mma_layout

atom = SM80_16x8x16_F16F16F16F16_TN
draw_mma_layout(atom.a_layout, atom.b_layout, atom.c_layout,
                tile_mnk=atom.shape_mnk, main_title=atom.name)
```

![draw_mma_layout](images/draw_mma_layout.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout_a` | `Layout` | required | A matrix TV layout |
| `layout_b` | `Layout` | required | B matrix TV layout |
| `layout_c` | `Layout` | required | C matrix TV layout |
| `filename` | `str` | `None` | Output path |
| `tile_mnk` | `(M, N, K)` | `None` | MMA tile dimensions |
| `main_title` | `str` | `None` | Overall title |
| `dpi` | `int` | `150` | Resolution |
| `colorize` | `bool` | `True` | Rainbow colors |
| `thr_id_layout` | `Layout` | `None` | Custom thread-ID-to-color mapping |

## draw_slice

Draw a layout with sliced elements highlighted.

Slice specs use `None` for free dimensions and integers (or nested tuples
of integers/None) for fixed dimensions.  This is especially useful for
visualizing hierarchical slicing patterns from CuTe.

```python
from layout_algebra import Layout
from layout_algebra.viz import draw_slice

# Cecka's hierarchical tensor: ((3,2),((2,3),2)):((4,1),((2,15),100))
# Slice ((1,:),((:,0),:)) — fix inner-row=1 and middle-col=0
layout = Layout(((3, 2), ((2, 3), 2)), ((4, 1), ((2, 15), 100)))
draw_slice(layout, ((1, None), ((None, 0), None)), title="((1,:),((:,0),:))")
```

![draw_slice](images/draw_slice.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | `Layout` | required | The layout to draw |
| `slice_spec` | `tuple` | required | Coordinate with `None` for free dims |
| `filename` | `str` | `None` | Output path |
| `title` | `str` | `None` | Title |
| `dpi` | `int` | `150` | Resolution |
| `figsize` | `(w, h)` | auto | Figure size |
| `colorize` | `bool` | `False` | Rainbow colors |
| `color_layout` | `Layout` | `None` | Custom coloring in the layout's logical coordinate space |
| `num_shades` | `int` | `8` | Grayscale shades |

## draw_composite

Draw multiple layouts in a multi-panel figure.

```python
from layout_algebra import Layout
from layout_algebra.viz import draw_composite

panels = [Layout((4, 4), (4, 1)), Layout((4, 4), (1, 4))]
draw_composite(panels, "comparison.png",
               titles=["Row-Major", "Column-Major"],
               main_title="Layout Comparison", colorize=True)
```

![draw_composite](images/draw_composite.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `panels` | `list` | required | List of Layouts to draw |
| `filename` | `str` | required | Output path |
| `arrangement` | `str` | `"horizontal"` | `"horizontal"` or `"vertical"` |
| `titles` | `list` | `None` | Per-panel titles |
| `main_title` | `str` | `None` | Overall title |
| `dpi` | `int` | `150` | Resolution |
| `panel_size` | `(w, h)` | `(4, 4)` | Size per panel |
| `colorize` | `bool` | `False` | Rainbow colors |
| `tv_mode` | `bool` | `False` | Use TV-layout rendering |

## draw_tiled_grid

Draw a tiled MMA grid produced by `tile_mma_grid()`.

```python
from layout_algebra import Layout
from layout_algebra.atoms_nv import *
from layout_algebra.layout_utils import tile_mma_grid
from layout_algebra.viz import draw_tiled_grid

atom = SM80_16x8x16_F16F16F16F16_TN
atom_layout = Layout((2, 2), (1, 2))  # 2x2 grid of atoms
grid, tile_shape = tile_mma_grid(atom, atom_layout, matrix='C')
draw_tiled_grid(grid, tile_shape[0], tile_shape[1],
                title="SM80 16x8x16 C — 2x2 atoms")
```

![draw_tiled_grid](images/draw_tiled_grid.png)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid` | `dict` | required | Grid data from `tile_mma_grid()` |
| `rows` | `int` | required | Number of tile rows |
| `cols` | `int` | required | Number of tile columns |
| `filename` | `str` | `None` | Output path |
| `dpi` | `int` | `150` | Resolution |
| `title` | `str` | `None` | Title |

## Jupyter Inline Display

`show_layout` and `show_swizzle` display inline and return the matplotlib
`Figure`:

```python
from layout_algebra.viz import show_layout, show_swizzle

fig = show_layout(Layout((8, 8), (8, 1)), colorize=True)
fig = show_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3))
```

| `show_layout` | `show_swizzle` |
|----------------|----------------|
| ![show_layout](images/show_layout.png) | ![show_swizzle](images/show_swizzle.png) |
