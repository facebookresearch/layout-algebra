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

"""Intel AMX (Advanced Matrix Extensions) tile atom definitions.

Maps AMX tile matrix-multiply instructions to the (Thread, Value) ->
element-offset framework used by CuTe-style layout algebra.

Conceptual mapping
==================

AMX is a true tile matrix multiply executed by a single CPU core.  The
instruction operates on 8 KiB tile registers (tmm0-tmm7), each holding
up to 16 rows x 64 bytes.  A single ``tdp*`` instruction computes the
full C[M,N] += A[M,K] * B[K,N] tile with no thread cooperation, so:

    T = 1   (one CPU core)
    V = M*K, N*K, or M*N  (all elements in the Value dimension)

This is the cleanest CPU -> MMAAtom mapping because AMX is a genuine
matrix multiply, unlike AVX512 VNNI which is a batched dot-product.

Tile dimensions
===============

All AMX tile-multiply instructions use M=16, N=16 output tiles.
The K dimension depends on the element type:

    BF16 (tdpbf16ps):  K=32  (2 BF16 per 32-bit pair, 64-byte rows)
    FP16 (tdpfp16ps):  K=32  (2 FP16 per 32-bit pair, 64-byte rows)
    INT8 (tdpbssd):    K=64  (4 INT8 per 32-bit group, 64-byte rows)

B-tile storage
--------------

The B tile register uses "VNNI format" (pairs/quads of K-elements packed
into 32-bit groups along the row), but the LOGICAL layout is still K x N.
Our atom layouts describe the logical element ordering, not the physical
register packing.

References
==========

- Intel Architecture Instruction Set Extensions Programming Reference
  (ISE), Chapter 3 -- AMX (TILECFG, TILELOADD, TDPBF16PS, TDPBSSD, etc.)
- Intel 64 and IA-32 Architectures Software Developer's Manual (SDM),
  Volume 2 -- Instruction Set Reference

Usage::

    from tensor_layouts.atoms_amx import AMX_16x16x32_F32BF16BF16F32
    atom = AMX_16x16x32_F32BF16BF16F32
    print(atom.shape_mnk)     # (16, 16, 32)
    print(atom.c_layout)      # (1, (16, 16)):(0, (1, 16))
"""

from .layouts import Layout
from .atoms import MMAAtom


# =============================================================================
# Intel AMX tile matrix multiply atoms — T=1 (single CPU core)
# Source: Intel ISE Chapter 3, AMX instructions
#
# tdpbf16ps  tmm, tmm, tmm  -- C[16,16] FP32 += A[16,32] BF16 * B[32,16] BF16
# tdpfp16ps  tmm, tmm, tmm  -- C[16,16] FP32 += A[16,32] FP16 * B[32,16] FP16
# tdpbssd    tmm, tmm, tmm  -- C[16,16] INT32 += A[16,64] INT8 * B[64,16] INT8
# tdpbsud    tmm, tmm, tmm  -- C[16,16] INT32 += A[16,64] INT8 * B[64,16] UINT8
# tdpbusd    tmm, tmm, tmm  -- C[16,16] INT32 += A[16,64] UINT8 * B[64,16] INT8
# tdpbuud    tmm, tmm, tmm  -- C[16,16] INT32 += A[16,64] UINT8 * B[64,16] UINT8
#
# All produce a 16x16 output tile.  K varies by datatype.
# =============================================================================

# -- BF16 -> FP32 -------------------------------------------------------------
AMX_16x16x32_F32BF16BF16F32 = MMAAtom(
    name="AMX_16x16x32_F32BF16BF16F32",
    ptx="tdpbf16ps",
    shape_mnk=(16, 16, 32),
    thr_id=Layout(1),
    # (T=1, V=512) -> col-major offset in (M=16, K=32)
    a_layout=Layout((1, (16, 32)), (0, (1, 16))),
    # (T=1, V=512) -> col-major offset in (N=16, K=32)
    b_layout=Layout((1, (16, 32)), (0, (1, 16))),
    # (T=1, V=256) -> col-major offset in (M=16, N=16)
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)

# -- FP16 -> FP32 -------------------------------------------------------------
AMX_16x16x32_F32F16F16F32 = MMAAtom(
    name="AMX_16x16x32_F32F16F16F32",
    ptx="tdpfp16ps",
    shape_mnk=(16, 16, 32),
    thr_id=Layout(1),
    a_layout=Layout((1, (16, 32)), (0, (1, 16))),
    b_layout=Layout((1, (16, 32)), (0, (1, 16))),
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)

# -- INT8 x INT8 -> INT32 (signed x signed) -----------------------------------
AMX_16x16x64_S32S8S8S32 = MMAAtom(
    name="AMX_16x16x64_S32S8S8S32",
    ptx="tdpbssd",
    shape_mnk=(16, 16, 64),
    thr_id=Layout(1),
    # (T=1, V=1024) -> col-major offset in (M=16, K=64)
    a_layout=Layout((1, (16, 64)), (0, (1, 16))),
    # (T=1, V=1024) -> col-major offset in (N=16, K=64)
    b_layout=Layout((1, (16, 64)), (0, (1, 16))),
    # (T=1, V=256) -> col-major offset in (M=16, N=16)
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)

# -- INT8 x UINT8 -> INT32 (signed x unsigned) --------------------------------
AMX_16x16x64_S32S8U8S32 = MMAAtom(
    name="AMX_16x16x64_S32S8U8S32",
    ptx="tdpbsud",
    shape_mnk=(16, 16, 64),
    thr_id=Layout(1),
    a_layout=Layout((1, (16, 64)), (0, (1, 16))),
    b_layout=Layout((1, (16, 64)), (0, (1, 16))),
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)

# -- UINT8 x INT8 -> INT32 (unsigned x signed) --------------------------------
AMX_16x16x64_S32U8S8S32 = MMAAtom(
    name="AMX_16x16x64_S32U8S8S32",
    ptx="tdpbusd",
    shape_mnk=(16, 16, 64),
    thr_id=Layout(1),
    a_layout=Layout((1, (16, 64)), (0, (1, 16))),
    b_layout=Layout((1, (16, 64)), (0, (1, 16))),
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)

# -- UINT8 x UINT8 -> INT32 (unsigned x unsigned) -----------------------------
AMX_16x16x64_S32U8U8S32 = MMAAtom(
    name="AMX_16x16x64_S32U8U8S32",
    ptx="tdpbuud",
    shape_mnk=(16, 16, 64),
    thr_id=Layout(1),
    a_layout=Layout((1, (16, 64)), (0, (1, 16))),
    b_layout=Layout((1, (16, 64)), (0, (1, 16))),
    c_layout=Layout((1, (16, 16)), (0, (1, 16))),
)
