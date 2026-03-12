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

import pytest

from layout_algebra import Layout, Swizzle

try:
    import matplotlib.figure
    from layout_algebra.viz import show_layout, show_swizzle
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


requires_viz = pytest.mark.skipif(
    not HAS_VIZ,
    reason="layout_algebra.viz not available (needs matplotlib)"
)


@requires_viz
def test_show_layout_returns_figure_without_raising():
    """Smoke test for show_layout helper."""
    fig = show_layout(Layout((8, 8), (8, 1)))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


@requires_viz
def test_show_swizzle_returns_figure_without_raising():
    """Regression test for show_swizzle helper."""
    fig = show_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3))
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2
