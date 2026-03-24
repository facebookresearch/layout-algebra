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

"""MMA and Copy atom dataclass definitions.

Vendor-neutral base types used by both atoms_nv.py (NVIDIA) and
atoms_amd.py (AMD).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .layouts import Layout


@dataclass(frozen=True)
class MMAAtom:
    """MMA atom traits — mirrors C++ MMA_Traits<Operation>.

    Attributes:
        name:      Full operation name
        ptx:       Instruction string (PTX for NVIDIA, ISA mnemonic for AMD)
        shape_mnk: (M, N, K) logical shape of the MMA
        thr_id:    Layout mapping logical thread id -> physical thread index
        a_layout:  (T, V) -> col-major offset in (M, K)
        b_layout:  (T, V) -> col-major offset in (N, K)
        c_layout:  (T, V) -> col-major offset in (M, N)
    """
    name: str
    ptx: str
    shape_mnk: Tuple[int, int, int]
    thr_id: Optional[Layout]
    a_layout: Layout
    b_layout: Layout
    c_layout: Layout

    def __str__(self) -> str:
        m, n, k = self.shape_mnk
        return f"MMAAtom('{self.name}', {m}x{n}x{k})"


@dataclass(frozen=True)
class CopyAtom:
    """Copy atom traits — mirrors C++ Copy_Traits<Operation>.

    Layouts are in BIT coordinates as defined in the C++ source.

    Attributes:
        name:           Full operation name
        ptx:            Instruction string
        thr_id:         Layout<N> for N threads
        src_layout_bits: (thr, val) -> bit offset for source
        dst_layout_bits: (thr, val) -> bit offset for destination
    """
    name: str
    ptx: str
    thr_id: Layout
    src_layout_bits: Layout
    dst_layout_bits: Layout

    def __str__(self) -> str:
        return f"CopyAtom('{self.name}')"
