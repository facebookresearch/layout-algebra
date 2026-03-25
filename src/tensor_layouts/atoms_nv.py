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

"""CuTe atom definitions: MMA and Copy atom traits.

Mirrors the C++ MMA_Traits<> and Copy_Traits<> structs from the NVIDIA
Cutlass library (include/cute/atom/).

Each MMA atom maps (thread_idx, value_idx) → element coordinate using
column-major encoding:
    A: (T, V) → m + k*M   in the M×K matrix
    B: (T, V) → n + k*N   in the N×K matrix
    C: (T, V) → m + n*M   in the M×N matrix

CUTLASS source paths (relative to cutlass-3 root):
    include/cute/atom/mma_traits_sm70.hpp
    include/cute/atom/mma_traits_sm75.hpp
    include/cute/atom/mma_traits_sm80.hpp
    include/cute/atom/mma_traits_sm89.hpp
    include/cute/atom/mma_traits_sm90.hpp
    include/cute/atom/mma_traits_sm90_gmma.hpp
    include/cute/atom/copy_traits_sm75.hpp
    include/cute/atom/copy_traits_sm90.hpp

PTX ISA documentation:
    https://docs.nvidia.com/cuda/parallel-thread-execution/
    §9.7.13  Matrix Multiply-Accumulate
    §9.7.12  Matrix Load/Store

Usage:
    from tensor_layouts.atoms_nv import SM70_8x8x4_F32F16F16F32_NT, SM80_16x8x8_F16F16F16F16_TN
    print(SM70_8x8x4_F32F16F16F32_NT.a_layout)
"""

from .layouts import Layout
from .atoms import MMAAtom, CopyAtom

# =============================================================================
# SM61 Pascal DP MMA atoms — 1 "thread" (scalar)
# Source: include/cute/atom/mma_traits_sm61.hpp
# PTX:    dp4a / dp2a (Dot Product and Accumulate)
# =============================================================================

SM61_1x1x4_S32S8S8S32 = MMAAtom(
    name="SM61_DP4A",
    ptx="dp4a.s32.s32",
    shape_mnk=(1, 1, 4),
    thr_id=Layout(1),
    a_layout=Layout((1, 4)),
    b_layout=Layout((1, 4)),
    c_layout=Layout((1, 1)),
)

SM61_1x1x2_S32S16S16S32 = MMAAtom(
    name="SM61_DP2A",
    ptx="dp2a.s32.s32",
    shape_mnk=(1, 1, 2),
    thr_id=Layout(1),
    a_layout=Layout((1, 2)),
    b_layout=Layout((1, 2)),
    c_layout=Layout((1, 1)),
)


# =============================================================================
# Reusable layout aliases
# From mma_traits_sm70.hpp (anonymous namespace, lines 41-57)
# =============================================================================

# Logical thread id → warp lane index (quadpair: lanes 0-3 and 16-19)
SM70_QuadPair = Layout((4, 2), (1, 16))  # line 44
SM70_8x4_Row = Layout((8, 4), (1, 8))  # line 47: (T8,V4) → (M8,K4)
SM70_8x4_Col = Layout(((4, 2), 4), ((8, 4), 1))  # line 50: (T8,V4) → (M8,K4)
SM70_8x8_16b = Layout((8, 8), (1, 8))  # line 53: (T8,V8) → (M8,N8) fp16 accum
SM70_8x8_32b = Layout(
    ((2, 2, 2), (2, 2, 2)),  # line 56: (T8,V8) → (M8,N8) fp32 accum
    ((1, 16, 4), (8, 2, 32)),
)

# =============================================================================
# From mma_traits_sm80.hpp (lines 41-55)
# =============================================================================

SM80_8x4 = Layout(((4, 8), 1), ((8, 1), 0))  # line 42: (T32,V1) → (M8,N8)
SM80_8x8_Row = Layout(((4, 8), 2), ((16, 1), 8))  # line 46: (T32,V2) → (M8,N8)
SM80_8x16_Row = Layout(((4, 8), 4), ((32, 1), 8))  # line 50: (T32,V4) → (M8,N16)
SM80_16x8_Row = Layout(
    ((4, 8), (2, 2)), ((32, 1), (16, 8))  # line 53: (T32,V4) → (M16,N8)
)


# =============================================================================
# SM70 Volta MMA atoms — 8 threads (quadpair)
# Source: include/cute/atom/mma_traits_sm70.hpp
# PTX:    mma.sync.aligned.m8n8k4.{row|col}.{row|col}.{f16|f32}.f16.f16.{f16|f32}
#         PTX ISA §9.7.13.4.1 (Volta HMMA)
# =============================================================================

# line 64 — fp16 accumulator, A=row-major, B=col-major
SM70_8x8x4_F16F16F16F16_TN = MMAAtom(
    name="SM70_8x8x4_F16F16F16F16_TN",
    ptx="mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Row,
    b_layout=SM70_8x4_Row,
    c_layout=SM70_8x8_16b,
)

# line 81 — fp16 accumulator, A=col-major, B=row-major
SM70_8x8x4_F16F16F16F16_NT = MMAAtom(
    name="SM70_8x8x4_F16F16F16F16_NT",
    ptx="mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Col,
    b_layout=SM70_8x4_Col,
    c_layout=SM70_8x8_16b,
)

# line 98
SM70_8x8x4_F16F16F16F16_NN = MMAAtom(
    name="SM70_8x8x4_F16F16F16F16_NN",
    ptx="mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Col,
    b_layout=SM70_8x4_Row,
    c_layout=SM70_8x8_16b,
)

# line 115
SM70_8x8x4_F16F16F16F16_TT = MMAAtom(
    name="SM70_8x8x4_F16F16F16F16_TT",
    ptx="mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Row,
    b_layout=SM70_8x4_Col,
    c_layout=SM70_8x8_16b,
)

# line 132 — fp32 accumulator, A=row-major, B=col-major
SM70_8x8x4_F32F16F16F32_TN = MMAAtom(
    name="SM70_8x8x4_F32F16F16F32_TN",
    ptx="mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Row,
    b_layout=SM70_8x4_Row,
    c_layout=SM70_8x8_32b,
)

# line 149 — fp32 accumulator, A=col-major, B=row-major
# Reference image: media/images/cute/HMMA.8x8x4.NT_Atom.png
SM70_8x8x4_F32F16F16F32_NT = MMAAtom(
    name="SM70_8x8x4_F32F16F16F32_NT",
    ptx="mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Col,
    b_layout=SM70_8x4_Col,
    c_layout=SM70_8x8_32b,
)

# line 166
SM70_8x8x4_F32F16F16F32_NN = MMAAtom(
    name="SM70_8x8x4_F32F16F16F32_NN",
    ptx="mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Col,
    b_layout=SM70_8x4_Row,
    c_layout=SM70_8x8_32b,
)

# line 183
SM70_8x8x4_F32F16F16F32_TT = MMAAtom(
    name="SM70_8x8x4_F32F16F16F32_TT",
    ptx="mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32",
    shape_mnk=(8, 8, 4),
    thr_id=SM70_QuadPair,
    a_layout=SM70_8x4_Row,
    b_layout=SM70_8x4_Col,
    c_layout=SM70_8x8_32b,
)


# =============================================================================
# SM75 Turing MMA atoms — 32 threads (warp)
# Source: include/cute/atom/mma_traits_sm75.hpp
# PTX:    mma.sync.aligned.m16n8k8 / m8n8k16
#         PTX ISA §9.7.13.4.2 (Turing)
# =============================================================================

SM75_16x8x8_F32F16F16F32_TN = MMAAtom(
    name="SM75_16x8x8_F32F16F16F32_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2)), ((32, 1), (16, 8))),
    b_layout=Layout(((4, 8), 2), ((16, 1), 8)),
    c_layout=Layout(((4, 8), (2, 2)), ((32, 1), (16, 8))),
)

SM75_8x8x16_S32S8S8S32_TN = MMAAtom(
    name="SM75_8x8x16_S32S8S8S32_TN",
    ptx="mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32",
    shape_mnk=(8, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), 4), ((32, 1), 8)),
    b_layout=Layout(((4, 8), 4), ((32, 1), 8)),
    c_layout=Layout(((4, 8), 2), ((16, 1), 8)),
)


# =============================================================================
# SM80 Ampere MMA atoms — 32 threads (warp)
# Source: include/cute/atom/mma_traits_sm80.hpp
# PTX:    mma.sync.aligned.m{8,16}n8k{4,8,16,32,64,128,256}.row.col
#         PTX ISA §9.7.13.4.3 (Ampere)
#
# Reusable layout aliases (from mma_traits_sm80.hpp lines 41-55):
#   SM80_8x4      = ((4,8),1):((8,1),0)         — (T32,V1)->(M8,K4)
#   SM80_8x8_Row  = ((4,8),2):((16,1),8)        — (T32,V2)->(M8,K8)
#   SM80_8x16_Row = ((4,8),4):((32,1),8)        — (T32,V4)->(M8,K16)
#   SM80_16x8_Row = ((4,8),(2,2)):((32,1),(16,8)) — (T32,V4)->(M16,N8)
# =============================================================================

# --- FP16 ---
SM80_16x8x8_F16F16F16F16_TN = MMAAtom(
    name="SM80_16x8x8_F16F16F16F16_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=SM80_16x8_Row,
    b_layout=SM80_8x8_Row,
    c_layout=SM80_16x8_Row,
)

SM80_16x8x16_F16F16F16F16_TN = MMAAtom(
    name="SM80_16x8x16_F16F16F16F16_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
    b_layout=Layout(((4, 8), (2, 2)), ((16, 1), (8, 64))),
    c_layout=SM80_16x8_Row,
)

# --- FP32 accumulator with FP16 inputs ---
SM80_16x8x8_F32F16F16F32_TN = MMAAtom(
    name="SM80_16x8x8_F32F16F16F32_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=SM80_16x8_Row,
    b_layout=SM80_8x8_Row,
    c_layout=SM80_16x8_Row,
)

SM80_16x8x16_F32F16F16F32_TN = MMAAtom(
    name="SM80_16x8x16_F32F16F16F32_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
    b_layout=Layout(((4, 8), (2, 2)), ((16, 1), (8, 64))),
    c_layout=SM80_16x8_Row,
)

# --- BF16 (same layouts as FP16) ---
SM80_16x8x8_F32BF16BF16F32_TN = MMAAtom(
    name="SM80_16x8x8_F32BF16BF16F32_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=SM80_16x8_Row,
    b_layout=SM80_8x8_Row,
    c_layout=SM80_16x8_Row,
)

SM80_16x8x16_F32BF16BF16F32_TN = MMAAtom(
    name="SM80_16x8x16_F32BF16BF16F32_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128))),
    b_layout=Layout(((4, 8), (2, 2)), ((16, 1), (8, 64))),
    c_layout=SM80_16x8_Row,
)

# --- TF32 (TensorFloat-32) ---
SM80_16x8x4_F32TF32TF32F32_TN = MMAAtom(
    name="SM80_16x8x4_F32TF32TF32F32_TN",
    ptx="mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32",
    shape_mnk=(16, 8, 4),
    thr_id=None,
    a_layout=Layout(((4, 8), 2), ((16, 1), 8)),
    b_layout=SM80_8x4,
    c_layout=SM80_16x8_Row,
)

SM80_16x8x8_F32TF32TF32F32_TN = MMAAtom(
    name="SM80_16x8x8_F32TF32TF32F32_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2)), ((16, 1), (8, 64))),
    b_layout=Layout(((4, 8), 2), ((8, 1), 32)),
    c_layout=SM80_16x8_Row,
)

# --- FP64 ---
SM80_8x8x4_F64F64F64F64_TN = MMAAtom(
    name="SM80_8x8x4_F64F64F64F64_TN",
    ptx="mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64",
    shape_mnk=(8, 8, 4),
    thr_id=None,
    a_layout=SM80_8x4,
    b_layout=SM80_8x4,
    c_layout=SM80_8x8_Row,
)

# --- INT8 (s8×s8, s8×u8, u8×s8, u8×u8 all share layouts at same tile size) ---

# 8x8x16
SM80_8x8x16_S32S8S8S32_TN = MMAAtom(
    name="SM80_8x8x16_S32S8S8S32_TN",
    ptx="mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32",
    shape_mnk=(8, 8, 16),
    thr_id=None,
    a_layout=SM80_8x16_Row,
    b_layout=SM80_8x16_Row,
    c_layout=SM80_8x8_Row,
)

# 16x8x16
SM80_16x8x16_S32S8S8S32_TN = MMAAtom(
    name="SM80_16x8x16_S32S8S8S32_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), (4, 2)), ((64, 1), (16, 8))),
    b_layout=SM80_8x16_Row,
    c_layout=SM80_16x8_Row,
)

# 16x8x32
SM80_16x8x32_S32S8S8S32_TN = MMAAtom(
    name="SM80_16x8x32_S32S8S8S32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
    b_layout=Layout(((4, 8), (4, 2)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

# --- INT4 ---

# 8x8x32
SM80_8x8x32_S32S4S4S32_TN = MMAAtom(
    name="SM80_8x8x32_S32S4S4S32_TN",
    ptx="mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32",
    shape_mnk=(8, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), 8), ((64, 1), 8)),
    b_layout=Layout(((4, 8), 8), ((64, 1), 8)),
    c_layout=SM80_8x8_Row,
)

# 16x8x32
SM80_16x8x32_S32S4S4S32_TN = MMAAtom(
    name="SM80_16x8x32_S32S4S4S32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2)), ((128, 1), (16, 8))),
    b_layout=Layout(((4, 8), 8), ((32, 1), 8)),
    c_layout=SM80_16x8_Row,
)

# 16x8x64
SM80_16x8x64_S32S4S4S32_TN = MMAAtom(
    name="SM80_16x8x64_S32S4S4S32_TN",
    ptx="mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32",
    shape_mnk=(16, 8, 64),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
    b_layout=Layout(((4, 8), (8, 2)), ((64, 1), (8, 256))),
    c_layout=SM80_16x8_Row,
)

# --- Binary (U1) ---

SM80_8x8x128_S32U1U1S32_TN_XORPOPC = MMAAtom(
    name="SM80_8x8x128_S32U1U1S32_TN_XORPOPC",
    ptx="mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
    shape_mnk=(8, 8, 128),
    thr_id=None,
    a_layout=Layout(((4, 8), 32), ((256, 1), 8)),
    b_layout=Layout(((4, 8), 32), ((256, 1), 8)),
    c_layout=SM80_8x8_Row,
)

SM80_16x8x128_S32U1U1S32_TN_XORPOPC = MMAAtom(
    name="SM80_16x8x128_S32U1U1S32_TN_XORPOPC",
    ptx="mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc",
    shape_mnk=(16, 8, 128),
    thr_id=None,
    a_layout=Layout(((4, 8), (32, 2)), ((512, 1), (16, 8))),
    b_layout=Layout(((4, 8), 32), ((256, 1), 8)),
    c_layout=SM80_16x8_Row,
)

SM80_16x8x256_S32U1U1S32_TN_XORPOPC = MMAAtom(
    name="SM80_16x8x256_S32U1U1S32_TN_XORPOPC",
    ptx="mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
    shape_mnk=(16, 8, 256),
    thr_id=None,
    a_layout=Layout(((4, 8), (32, 2, 2)), ((512, 1), (16, 8, 2048))),
    b_layout=Layout(((4, 8), (32, 2)), ((256, 1), (8, 1024))),
    c_layout=SM80_16x8_Row,
)


# =============================================================================
# SM89 Ada Lovelace FP8 MMA atoms — 32 threads (warp)
# Source: include/cute/atom/mma_traits_sm89.hpp
# PTX:    mma.sync.aligned.m16n8k32.row.col.{f32|f16}.{e4m3|e5m2}.{e4m3|e5m2}.{f32|f16}
#         PTX ISA §9.7.13.4.6 (FP8)
#
# All 8 FP8 variants share the same A/B/C layouts (same as SM80_16x8x32_S32S8S8S32).
# =============================================================================

# FP32 accumulator variants
SM89_16x8x32_F32E4M3E4M3F32_TN = MMAAtom(
    name="SM89_16x8x32_F32E4M3E4M3F32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
    b_layout=Layout(((4, 8), (4, 2)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

SM89_16x8x32_F32E4M3E5M2F32_TN = MMAAtom(
    name="SM89_16x8x32_F32E4M3E5M2F32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

SM89_16x8x32_F32E5M2E5M2F32_TN = MMAAtom(
    name="SM89_16x8x32_F32E5M2E5M2F32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

SM89_16x8x32_F32E5M2E4M3F32_TN = MMAAtom(
    name="SM89_16x8x32_F32E5M2E4M3F32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

# FP16 accumulator variants (same layouts)
SM89_16x8x32_F16E4M3E4M3F16_TN = MMAAtom(
    name="SM89_16x8x32_F16E4M3E4M3F16_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

SM89_16x8x32_F16E4M3E5M2F16_TN = MMAAtom(
    name="SM89_16x8x32_F16E4M3E5M2F16_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

SM89_16x8x32_F16E5M2E5M2F16_TN = MMAAtom(
    name="SM89_16x8x32_F16E5M2E5M2F16_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)

SM89_16x8x32_F16E5M2E4M3F16_TN = MMAAtom(
    name="SM89_16x8x32_F16E5M2E4M3F16_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.a_layout,
    b_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.b_layout,
    c_layout=SM89_16x8x32_F32E4M3E4M3F32_TN.c_layout,
)


# =============================================================================
# SM90 Hopper warp-level MMA atoms — 32 threads
# Source: include/cute/atom/mma_traits_sm90.hpp
# PTX:    mma.sync.aligned.m16n8k{4,8,16}.row.col.{f64|c64}
#         PTX ISA §9.7.13.4.4 (FP64 / Complex FP64)
# =============================================================================

# --- FP64 ---
SM90_16x8x4_F64F64F64F64_TN = MMAAtom(
    name="SM90_16x8x4_F64F64F64F64_TN",
    ptx="mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64",
    shape_mnk=(16, 8, 4),
    thr_id=None,
    a_layout=Layout(((4, 8), 2), ((16, 1), 8)),
    b_layout=SM80_8x4,
    c_layout=SM80_16x8_Row,
)

# line 67
SM90_16x8x8_F64F64F64F64_TN = MMAAtom(
    name="SM90_16x8x8_F64F64F64F64_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 2)), ((16, 1), (8, 64))),
    b_layout=Layout(((4, 8), 2), ((8, 1), 32)),
    c_layout=SM80_16x8_Row,
)

# line 87
SM90_16x8x16_F64F64F64F64_TN = MMAAtom(
    name="SM90_16x8x16_F64F64F64F64_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=Layout(((4, 8), (2, 4)), ((16, 1), (8, 64))),
    b_layout=Layout(((4, 8), 4), ((8, 1), 32)),
    c_layout=SM80_16x8_Row,
)

# --- Complex FP64 (same layouts as FP64, different value types) ---

SM90_16x8x4_C64C64C64C64_TN = MMAAtom(
    name="SM90_16x8x4_C64C64C64C64_TN",
    ptx="mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 (complex)",
    shape_mnk=(16, 8, 4),
    thr_id=None,
    a_layout=SM90_16x8x4_F64F64F64F64_TN.a_layout,
    b_layout=SM90_16x8x4_F64F64F64F64_TN.b_layout,
    c_layout=SM90_16x8x4_F64F64F64F64_TN.c_layout,
)

SM90_16x8x8_C64C64C64C64_TN = MMAAtom(
    name="SM90_16x8x8_C64C64C64C64_TN",
    ptx="mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 (complex)",
    shape_mnk=(16, 8, 8),
    thr_id=None,
    a_layout=SM90_16x8x8_F64F64F64F64_TN.a_layout,
    b_layout=SM90_16x8x8_F64F64F64F64_TN.b_layout,
    c_layout=SM90_16x8x8_F64F64F64F64_TN.c_layout,
)

SM90_16x8x16_C64C64C64C64_TN = MMAAtom(
    name="SM90_16x8x16_C64C64C64C64_TN",
    ptx="mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 (complex)",
    shape_mnk=(16, 8, 16),
    thr_id=None,
    a_layout=SM90_16x8x16_F64F64F64F64_TN.a_layout,
    b_layout=SM90_16x8x16_F64F64F64F64_TN.b_layout,
    c_layout=SM90_16x8x16_F64F64F64F64_TN.c_layout,
)


# =============================================================================
# SM90 Hopper GMMA (warpgroup) atoms — 128 threads
# Source: include/cute/atom/mma_traits_sm90_gmma.hpp
# PTX:    wgmma.mma_async.sync.aligned.m64nNk16.f16.f16.f16
#         PTX ISA §9.7.14 (Warpgroup MMA)
#
# CLayout_64xN template (line 432):
#   Shape:  ((4, 8, 4), (2, 2, N/8))
#   Stride: ((128, 1, 16), (64, 8, 512))
#
# A/B use shared memory descriptors; all 128 threads see the entire
# tile with stride-0 in the thread dimension (line 436-443 in 0t_mma_atom.md).
# =============================================================================


def gmma_c_layout(n: int) -> Layout:
    """CLayout_64xN: accumulator layout for SM90 GMMA with N columns.
    Source: mma_traits_sm90_gmma.hpp line 432."""
    return Layout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))


def gmma_ab_layout(m: int, k: int) -> Layout:
    """ABLayout<M,K>: shared memory descriptor layout — all threads see entire tile.
    Source: mma_traits_sm90_gmma.hpp; 0t_mma_atom.md lines 436-443."""
    return Layout((128, (m, k)), (0, (1, m)))


# line 657 — SM90_64x64x16_F16F16F16_SS
SM90_64x8x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x8x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16",
    shape_mnk=(64, 8, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(8, 16),
    c_layout=gmma_c_layout(8),
)

SM90_64x16x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x16x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16",
    shape_mnk=(64, 16, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(16, 16),
    c_layout=gmma_c_layout(16),
)

SM90_64x32x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x32x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16",
    shape_mnk=(64, 32, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(32, 16),
    c_layout=gmma_c_layout(32),
)

SM90_64x64x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x64x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16",
    shape_mnk=(64, 64, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(64, 16),
    c_layout=gmma_c_layout(64),
)

SM90_64x128x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x128x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16",
    shape_mnk=(64, 128, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(128, 16),
    c_layout=gmma_c_layout(128),
)

SM90_64x256x16_F16F16F16_SS = MMAAtom(
    name="SM90_64x256x16_F16F16F16_SS",
    ptx="wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16",
    shape_mnk=(64, 256, 16),
    thr_id=None,
    a_layout=gmma_ab_layout(64, 16),
    b_layout=gmma_ab_layout(256, 16),
    c_layout=gmma_c_layout(256),
)


# =============================================================================
# SM90 GMMA factory — generate any GMMA atom with parametric N
# Source: include/cute/atom/mma_traits_sm90_gmma.hpp
#         include/cute/atom/mma_traits_sm90_gmma_ext.hpp (extended N values)
#
# The "ext" file adds N values beyond {8,16,32,64,128,256}: every multiple
# of 8 from 8 to 256 (i.e. 24, 40, 48, ..., 248) and data types like
# E4M3, E5M2, S8, U8 in all SS/RS combinations.
#
# All use the same CLayout_64xN and ABLayout<M,K> templates — so we
# provide a factory instead of enumerating hundreds of concrete atoms.
# =============================================================================


def make_gmma_atom_ss(
    n: int, k: int = 16, d_type: str = "F16", ab_type: str | None = None
) -> MMAAtom:
    """Create an SM90 GMMA SS atom for 64×N×K with the given data types.

    Args:
        n: N dimension (must be a multiple of 8, 8 ≤ N ≤ 256)
        k: K dimension (16 for F16/BF16, 8 for TF32/F64, 32 for S8/E4M3)
        d_type: accumulator/output data type (F16, F32, S32, etc.)
        ab_type: A/B input data type; defaults to d_type if None
    """
    if ab_type is None:
        ab_type = d_type
    if not (8 <= n <= 256 and n % 8 == 0):
        raise ValueError(f"N must be 8..256, multiple of 8, got {n}")
    name = f"SM90_64x{n}x{k}_{d_type}{ab_type}{ab_type}_SS"
    return MMAAtom(
        name=name,
        ptx=f"wgmma.mma_async.sync.aligned.m64n{n}k{k}",
        shape_mnk=(64, n, k),
        thr_id=None,
        a_layout=gmma_ab_layout(64, k),
        b_layout=gmma_ab_layout(n, k),
        c_layout=gmma_c_layout(n),
    )


# Representative ext atoms (N values not in the base set)
SM90_64x24x16_F16F16F16_SS = make_gmma_atom_ss(24)
SM90_64x48x16_F16F16F16_SS = make_gmma_atom_ss(48)
SM90_64x96x16_F16F16F16_SS = make_gmma_atom_ss(96)
SM90_64x192x16_F16F16F16_SS = make_gmma_atom_ss(192)

# F32-accumulator variants (same thread-value layouts, different register width)
# These are commonly referenced in NVIDIA documentation and CUTLASS kernels.
# PTX: wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16
SM90_64x8x16_F32F16F16_SS = make_gmma_atom_ss(8, d_type="F32", ab_type="F16")
SM90_64x16x16_F32F16F16_SS = make_gmma_atom_ss(16, d_type="F32", ab_type="F16")
SM90_64x32x16_F32F16F16_SS = make_gmma_atom_ss(32, d_type="F32", ab_type="F16")
SM90_64x64x16_F32F16F16_SS = make_gmma_atom_ss(64, d_type="F32", ab_type="F16")
SM90_64x128x16_F32F16F16_SS = make_gmma_atom_ss(128, d_type="F32", ab_type="F16")
SM90_64x256x16_F32F16F16_SS = make_gmma_atom_ss(256, d_type="F32", ab_type="F16")

# BF16 variants (K=16)
SM90_64x64x16_F32BF16BF16_SS = make_gmma_atom_ss(64, d_type="F32", ab_type="BF16")
SM90_64x128x16_F32BF16BF16_SS = make_gmma_atom_ss(128, d_type="F32", ab_type="BF16")
SM90_64x256x16_F32BF16BF16_SS = make_gmma_atom_ss(256, d_type="F32", ab_type="BF16")

# TF32 variants (K=8)
SM90_64x64x8_F32TF32TF32_SS = make_gmma_atom_ss(64, k=8, d_type="F32", ab_type="TF32")
SM90_64x128x8_F32TF32TF32_SS = make_gmma_atom_ss(128, k=8, d_type="F32", ab_type="TF32")
SM90_64x256x8_F32TF32TF32_SS = make_gmma_atom_ss(256, k=8, d_type="F32", ab_type="TF32")

# INT8 GMMA atoms (K=32 for 8-bit types, 256 bits / 8 bits = 32)
SM90_64x64x32_S32S8S8_SS = make_gmma_atom_ss(64, k=32, d_type="S32", ab_type="S8")
SM90_64x128x32_S32S8S8_SS = make_gmma_atom_ss(128, k=32, d_type="S32", ab_type="S8")
SM90_64x256x32_S32S8S8_SS = make_gmma_atom_ss(256, k=32, d_type="S32", ab_type="S8")

# FP8 E4M3 GMMA atoms (K=32 for 8-bit types)
SM90_64x64x32_F32E4M3E4M3_SS = make_gmma_atom_ss(64, k=32, d_type="F32", ab_type="E4M3")
SM90_64x128x32_F32E4M3E4M3_SS = make_gmma_atom_ss(
    128, k=32, d_type="F32", ab_type="E4M3"
)
SM90_64x256x32_F32E4M3E4M3_SS = make_gmma_atom_ss(
    256, k=32, d_type="F32", ab_type="E4M3"
)
SM90_64x64x32_F16E4M3E4M3_SS = make_gmma_atom_ss(64, k=32, d_type="F16", ab_type="E4M3")
SM90_64x128x32_F16E4M3E4M3_SS = make_gmma_atom_ss(
    128, k=32, d_type="F16", ab_type="E4M3"
)
SM90_64x256x32_F16E4M3E4M3_SS = make_gmma_atom_ss(
    256, k=32, d_type="F16", ab_type="E4M3"
)

# FP8 E5M2 GMMA atoms
SM90_64x64x32_F32E5M2E5M2_SS = make_gmma_atom_ss(64, k=32, d_type="F32", ab_type="E5M2")
SM90_64x128x32_F32E5M2E5M2_SS = make_gmma_atom_ss(
    128, k=32, d_type="F32", ab_type="E5M2"
)
SM90_64x256x32_F32E5M2E5M2_SS = make_gmma_atom_ss(
    256, k=32, d_type="F32", ab_type="E5M2"
)
SM90_64x64x32_F16E5M2E5M2_SS = make_gmma_atom_ss(64, k=32, d_type="F16", ab_type="E5M2")
SM90_64x128x32_F16E5M2E5M2_SS = make_gmma_atom_ss(
    128, k=32, d_type="F16", ab_type="E5M2"
)
SM90_64x256x32_F16E5M2E5M2_SS = make_gmma_atom_ss(
    256, k=32, d_type="F16", ab_type="E5M2"
)


# =============================================================================
# SM90 GMMA Sparse atoms — 128 threads (warpgroup), structured sparsity
# Source: include/cute/atom/mma_traits_sm90_gmma_sparse.hpp
#         include/cute/atom/mma_traits_sm90_gmma_sparse_ext.hpp
#
# Sparse GMMA doubles the K dimension (2:4 structured sparsity on A).
# Layout pattern identical to dense GMMA — same CLayout_64xN, same ABLayout.
# K_sparse = 2 * K_dense (e.g. K=32 for F16 sparse vs K=16 for F16 dense).
# =============================================================================


def make_gmma_sparse_atom_ss(
    n: int, k: int = 32, d_type: str = "F16", ab_type: str | None = None
) -> MMAAtom:
    """Create an SM90 GMMA sparse SS atom for 64×N×K."""
    if ab_type is None:
        ab_type = d_type
    if not (8 <= n <= 256 and n % 8 == 0):
        raise ValueError(f"N must be 8..256, multiple of 8, got {n}")
    name = f"SM90_64x{n}x{k}_{d_type}{ab_type}{ab_type}_SS_SPARSE"
    return MMAAtom(
        name=name,
        ptx=f"wgmma.mma_async.sp.sync.aligned.m64n{n}k{k}",
        shape_mnk=(64, n, k),
        thr_id=None,
        a_layout=gmma_ab_layout(64, k),
        b_layout=gmma_ab_layout(n, k),
        c_layout=gmma_c_layout(n),
    )


# F16 sparse (K=32, double the dense K=16)
SM90_64x64x32_F16F16F16_SS_SPARSE = make_gmma_sparse_atom_ss(64)
SM90_64x128x32_F16F16F16_SS_SPARSE = make_gmma_sparse_atom_ss(128)
SM90_64x256x32_F16F16F16_SS_SPARSE = make_gmma_sparse_atom_ss(256)

# TF32 sparse (K=16, double the dense K=8)
SM90_64x64x16_F32TF32TF32_SS_SPARSE = make_gmma_sparse_atom_ss(
    64, k=16, d_type="F32", ab_type="TF32"
)
SM90_64x128x16_F32TF32TF32_SS_SPARSE = make_gmma_sparse_atom_ss(
    128, k=16, d_type="F32", ab_type="TF32"
)
SM90_64x256x16_F32TF32TF32_SS_SPARSE = make_gmma_sparse_atom_ss(
    256, k=16, d_type="F32", ab_type="TF32"
)

# INT8 sparse (K=64, double the dense K=32)
SM90_64x64x64_S32S8S8_SS_SPARSE = make_gmma_sparse_atom_ss(
    64, k=64, d_type="S32", ab_type="S8"
)
SM90_64x128x64_S32S8S8_SS_SPARSE = make_gmma_sparse_atom_ss(
    128, k=64, d_type="S32", ab_type="S8"
)
SM90_64x256x64_S32S8S8_SS_SPARSE = make_gmma_sparse_atom_ss(
    256, k=64, d_type="S32", ab_type="S8"
)


# =============================================================================
# SM100 Blackwell UMMA (Unified MMA) atoms — 1 "thread" (warp group)
# Source: include/cute/atom/mma_traits_sm100.hpp
# PTX:    tcgen05.mma (UMMA instructions)
#
# SM100 uses TMEM (Tensor Memory) for accumulator and shared memory
# descriptors for A/B. The entire warp group operates as a single unit,
# so ThrID = Layout<1> and all elements are in the "value" dimension.
#
# Layout pattern:
#   ALayout = (1, (M, K)) : (0, (1, M))   — col-major M×K
#   BLayout = (1, (N, K)) : (0, (1, N))   — col-major N×K
#   CLayout = (1, (M, N)) : (0, (1, M))   — col-major M×N
#
# K = 256 / sizeof_bits(type):
#   FP16/BF16: K=16,  TF32: K=8
# M ∈ {64, 128},  N ∈ {8, 16, 24, ..., 256} (multiples of 8)
# =============================================================================


def umma_layout(rows: int, cols: int) -> Layout:
    """SM100 UMMA layout: (1, (rows, cols)) : (0, (1, rows)) — col-major."""
    return Layout((1, (rows, cols)), (0, (1, rows)))


# --- F16/BF16 SS (both operands from shared memory) ---

SM100_64x64x16_F16F16F16_SS = MMAAtom(
    name="SM100_64x64x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m64n64k16.f16.f16.f16",
    shape_mnk=(64, 64, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(64, 16),
    b_layout=umma_layout(64, 16),
    c_layout=umma_layout(64, 64),
)

SM100_64x128x16_F16F16F16_SS = MMAAtom(
    name="SM100_64x128x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m64n128k16.f16.f16.f16",
    shape_mnk=(64, 128, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(64, 16),
    b_layout=umma_layout(128, 16),
    c_layout=umma_layout(64, 128),
)

SM100_64x256x16_F16F16F16_SS = MMAAtom(
    name="SM100_64x256x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m64n256k16.f16.f16.f16",
    shape_mnk=(64, 256, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(64, 16),
    b_layout=umma_layout(256, 16),
    c_layout=umma_layout(64, 256),
)

SM100_128x64x16_F16F16F16_SS = MMAAtom(
    name="SM100_128x64x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m128n64k16.f16.f16.f16",
    shape_mnk=(128, 64, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(128, 16),
    b_layout=umma_layout(64, 16),
    c_layout=umma_layout(128, 64),
)

SM100_128x128x16_F16F16F16_SS = MMAAtom(
    name="SM100_128x128x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m128n128k16.f16.f16.f16",
    shape_mnk=(128, 128, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(128, 16),
    b_layout=umma_layout(128, 16),
    c_layout=umma_layout(128, 128),
)

SM100_128x256x16_F16F16F16_SS = MMAAtom(
    name="SM100_128x256x16_F16F16F16_SS",
    ptx="tcgen05.mma ... m128n256k16.f16.f16.f16",
    shape_mnk=(128, 256, 16),
    thr_id=Layout(1),
    a_layout=umma_layout(128, 16),
    b_layout=umma_layout(256, 16),
    c_layout=umma_layout(128, 256),
)

# --- TF32 SS (K=8 because 256/32=8) ---

SM100_64x64x8_F32TF32TF32F32_SS = MMAAtom(
    name="SM100_64x64x8_F32TF32TF32F32_SS",
    ptx="tcgen05.mma ... m64n64k8.f32.tf32.tf32.f32",
    shape_mnk=(64, 64, 8),
    thr_id=Layout(1),
    a_layout=umma_layout(64, 8),
    b_layout=umma_layout(64, 8),
    c_layout=umma_layout(64, 64),
)

SM100_128x128x8_F32TF32TF32F32_SS = MMAAtom(
    name="SM100_128x128x8_F32TF32TF32F32_SS",
    ptx="tcgen05.mma ... m128n128k8.f32.tf32.tf32.f32",
    shape_mnk=(128, 128, 8),
    thr_id=Layout(1),
    a_layout=umma_layout(128, 8),
    b_layout=umma_layout(128, 8),
    c_layout=umma_layout(128, 128),
)


# --- SM100 UMMA factory ---


def make_umma_atom_ss(
    m: int, n: int, k: int = 16, d_type: str = "F16", ab_type: str | None = None
) -> MMAAtom:
    """Create an SM100 UMMA SS atom for M×N×K with the given data types."""
    if ab_type is None:
        ab_type = d_type
    name = f"SM100_{m}x{n}x{k}_{d_type}{ab_type}{ab_type}_SS"
    return MMAAtom(
        name=name,
        ptx=f"tcgen05.mma ... m{m}n{n}k{k}",
        shape_mnk=(m, n, k),
        thr_id=Layout(1),
        a_layout=umma_layout(m, k),
        b_layout=umma_layout(n, k),
        c_layout=umma_layout(m, n),
    )


# F32-accumulator with F16 inputs
SM100_64x64x16_F32F16F16_SS = make_umma_atom_ss(64, 64, d_type="F32", ab_type="F16")
SM100_64x128x16_F32F16F16_SS = make_umma_atom_ss(64, 128, d_type="F32", ab_type="F16")
SM100_64x256x16_F32F16F16_SS = make_umma_atom_ss(64, 256, d_type="F32", ab_type="F16")
SM100_128x64x16_F32F16F16_SS = make_umma_atom_ss(128, 64, d_type="F32", ab_type="F16")
SM100_128x128x16_F32F16F16_SS = make_umma_atom_ss(128, 128, d_type="F32", ab_type="F16")
SM100_128x256x16_F32F16F16_SS = make_umma_atom_ss(128, 256, d_type="F32", ab_type="F16")

# F32-accumulator with BF16 inputs
SM100_64x64x16_F32BF16BF16_SS = make_umma_atom_ss(64, 64, d_type="F32", ab_type="BF16")
SM100_64x128x16_F32BF16BF16_SS = make_umma_atom_ss(
    64, 128, d_type="F32", ab_type="BF16"
)
SM100_64x256x16_F32BF16BF16_SS = make_umma_atom_ss(
    64, 256, d_type="F32", ab_type="BF16"
)
SM100_128x64x16_F32BF16BF16_SS = make_umma_atom_ss(
    128, 64, d_type="F32", ab_type="BF16"
)
SM100_128x128x16_F32BF16BF16_SS = make_umma_atom_ss(
    128, 128, d_type="F32", ab_type="BF16"
)
SM100_128x256x16_F32BF16BF16_SS = make_umma_atom_ss(
    128, 256, d_type="F32", ab_type="BF16"
)

# F16-accumulator with BF16 inputs
SM100_64x64x16_F16BF16BF16_SS = make_umma_atom_ss(64, 64, d_type="F16", ab_type="BF16")
SM100_64x128x16_F16BF16BF16_SS = make_umma_atom_ss(
    64, 128, d_type="F16", ab_type="BF16"
)
SM100_64x256x16_F16BF16BF16_SS = make_umma_atom_ss(
    64, 256, d_type="F16", ab_type="BF16"
)
SM100_128x64x16_F16BF16BF16_SS = make_umma_atom_ss(
    128, 64, d_type="F16", ab_type="BF16"
)
SM100_128x128x16_F16BF16BF16_SS = make_umma_atom_ss(
    128, 128, d_type="F16", ab_type="BF16"
)
SM100_128x256x16_F16BF16BF16_SS = make_umma_atom_ss(
    128, 256, d_type="F16", ab_type="BF16"
)


# =============================================================================
# SM120 MMA atoms — 32 threads (warp)
# Source: include/cute/atom/mma_traits_sm120.hpp
# PTX:    mma.sync.aligned.m16n8k{32,64}
#
# SM120 inherits layout patterns from SM80.
# =============================================================================

# SM120 F8F6F4 16x8x32 — inherits SM80_16x8x32 layouts
SM120_16x8x32_F32E4M3E4M3F32_TN = MMAAtom(
    name="SM120_16x8x32_F32E4M3E4M3F32_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
    b_layout=Layout(((4, 8), (4, 2)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

# SM120 block-scaled MXF8F6F4 16x8x64
SM120_16x8x64_F32E4M3E4M3F32_TN = MMAAtom(
    name="SM120_16x8x64_F32E4M3E4M3F32_TN",
    ptx="mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32",
    shape_mnk=(16, 8, 64),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
    b_layout=Layout(((4, 8), (8, 2)), ((64, 1), (8, 256))),
    c_layout=SM80_16x8_Row,
)

# --- SM120 Sparse (structured 2:4 sparsity) ---
# Source: include/cute/atom/mma_traits_sm120_sparse.hpp

# SM120 sparse 16x8x64 (FP8, 2:4 sparsity doubles K from 32 to 64)
SM120_16x8x64_F32E4M3E4M3F32_TN_SPARSE = MMAAtom(
    name="SM120_16x8x64_F32E4M3E4M3F32_TN_SPARSE",
    ptx="mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 (sparse)",
    shape_mnk=(16, 8, 64),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
    b_layout=Layout(((4, 8), (4, 4)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

# SM120 sparse block-scaled 16x8x128 (FP4, 2:4 sparsity)
SM120_16x8x128_F32E4M3E4M3F32_TN_SPARSE = MMAAtom(
    name="SM120_16x8x128_F32E4M3E4M3F32_TN_SPARSE",
    ptx="mma.sync.aligned.m16n8k128.row.col.f32.e4m3.e4m3.f32 (sparse)",
    shape_mnk=(16, 8, 128),
    thr_id=None,
    a_layout=Layout(((4, 8), (16, 2, 2)), ((256, 1), (16, 8, 1024))),
    b_layout=Layout(((4, 8), (8, 4)), ((64, 1), (8, 256))),
    c_layout=SM80_16x8_Row,
)

# --- SM120 F16-accumulator variants (same layouts as F32, different register width) ---

SM120_16x8x32_F16E4M3E4M3F16_TN = MMAAtom(
    name="SM120_16x8x32_F16E4M3E4M3F16_TN",
    ptx="mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16",
    shape_mnk=(16, 8, 32),
    thr_id=None,
    a_layout=Layout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256))),
    b_layout=Layout(((4, 8), (4, 2)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

SM120_16x8x64_F16E4M3E4M3F16_TN = MMAAtom(
    name="SM120_16x8x64_F16E4M3E4M3F16_TN",
    ptx="mma.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16",
    shape_mnk=(16, 8, 64),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
    b_layout=Layout(((4, 8), (8, 2)), ((64, 1), (8, 256))),
    c_layout=SM80_16x8_Row,
)

SM120_16x8x64_F16E4M3E4M3F16_TN_SPARSE = MMAAtom(
    name="SM120_16x8x64_F16E4M3E4M3F16_TN_SPARSE",
    ptx="mma.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 (sparse)",
    shape_mnk=(16, 8, 64),
    thr_id=None,
    a_layout=Layout(((4, 8), (8, 2, 2)), ((128, 1), (16, 8, 512))),
    b_layout=Layout(((4, 8), (4, 4)), ((32, 1), (8, 128))),
    c_layout=SM80_16x8_Row,
)

SM120_16x8x128_F16E4M3E4M3F16_TN_SPARSE = MMAAtom(
    name="SM120_16x8x128_F16E4M3E4M3F16_TN_SPARSE",
    ptx="mma.sync.aligned.m16n8k128.row.col.f16.e4m3.e4m3.f16 (sparse)",
    shape_mnk=(16, 8, 128),
    thr_id=None,
    a_layout=Layout(((4, 8), (16, 2, 2)), ((256, 1), (16, 8, 1024))),
    b_layout=Layout(((4, 8), (8, 4)), ((64, 1), (8, 256))),
    c_layout=SM80_16x8_Row,
)


# =============================================================================
# SM50 Copy atoms — Warp Shuffle
# Source: include/cute/atom/copy_traits_sm50.hpp
# PTX:    shfl.sync (warp shuffle)
# =============================================================================

SM50_Shuffle_U32_2x2Trans_XOR1 = CopyAtom(
    name="SM50_Shuffle_U32_2x2Trans_XOR1",
    ptx="shfl.sync.bfly (XOR1 2x2 transpose)",
    thr_id=Layout(32),
    src_layout_bits=Layout((32, 64), (64, 1)),
    dst_layout_bits=Layout(((2, 16), (32, 2)), ((32, 128), (1, 64))),
)

SM50_Shuffle_U32_2x2Trans_XOR4 = CopyAtom(
    name="SM50_Shuffle_U32_2x2Trans_XOR4",
    ptx="shfl.sync.bfly (XOR4 2x2 transpose)",
    thr_id=Layout(32),
    src_layout_bits=Layout((32, 64), (64, 1)),
    dst_layout_bits=Layout(((4, 2, 4), (32, 2)), ((64, 32, 512), (1, 256))),
)


# =============================================================================
# SM75 Copy atoms — LDMATRIX
# Source: include/cute/atom/copy_traits_sm75.hpp
# PTX:    ldmatrix.sync.aligned.{x1,x2,x4}[.trans].m8n8.shared.b16
#         PTX ISA §9.7.12.4
# =============================================================================

SM75_U32x1_LDSM_N = CopyAtom(
    name="SM75_U32x1_LDSM_N",
    ptx="ldmatrix.sync.aligned.x1.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout(((8, 4), 128), ((128, 0), 1)),
    dst_layout_bits=Layout((32, 32), (32, 1)),
)

SM75_U32x2_LDSM_N = CopyAtom(
    name="SM75_U32x2_LDSM_N",
    ptx="ldmatrix.sync.aligned.x2.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout(((16, 2), 128), ((128, 0), 1)),
    dst_layout_bits=Layout((32, (32, 2)), (32, (1, 1024))),
)

SM75_U32x4_LDSM_N = CopyAtom(
    name="SM75_U32x4_LDSM_N",
    ptx="ldmatrix.sync.aligned.x4.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout((32, 128), (128, 1)),
    dst_layout_bits=Layout((32, (32, 4)), (32, (1, 1024))),
)

SM75_U16x2_LDSM_T = CopyAtom(
    name="SM75_U16x2_LDSM_T",
    ptx="ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout(((8, 4), 128), ((128, 0), 1)),
    dst_layout_bits=Layout(((4, 8), (16, 2)), ((256, 16), (1, 128))),
)

SM75_U16x4_LDSM_T = CopyAtom(
    name="SM75_U16x4_LDSM_T",
    ptx="ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout(((16, 2), 128), ((128, 0), 1)),
    dst_layout_bits=Layout(((4, 8), (16, 2, 2)), ((256, 16), (1, 128, 1024))),
)

SM75_U16x8_LDSM_T = CopyAtom(
    name="SM75_U16x8_LDSM_T",
    ptx="ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=Layout((32, 128), (128, 1)),
    dst_layout_bits=Layout(((4, 8), (16, 2, 4)), ((256, 16), (1, 128, 1024))),
)


# =============================================================================
# SM80 Copy atoms — CP.ASYNC
# Source: include/cute/atom/copy_traits_sm80.hpp
# PTX:    cp.async.{ca,cg}.shared.global
#         PTX ISA §9.7.8.24
#
# These are single-thread (ThrID=1) copies. Layouts are trivial:
#   (1, sizeof_bits) — one thread, all bits in the value dimension.
# Included for completeness; the layout algebra is uninteresting.
# =============================================================================

SM80_CP_ASYNC_CACHEALWAYS_16B = CopyAtom(
    name="SM80_CP_ASYNC_CACHEALWAYS_16B",
    ptx="cp.async.ca.shared.global [16B]",
    thr_id=Layout(1),
    src_layout_bits=Layout((1, 128)),
    dst_layout_bits=Layout((1, 128)),
)

SM80_CP_ASYNC_CACHEGLOBAL_16B = CopyAtom(
    name="SM80_CP_ASYNC_CACHEGLOBAL_16B",
    ptx="cp.async.cg.shared.global [16B]",
    thr_id=Layout(1),
    src_layout_bits=Layout((1, 128)),
    dst_layout_bits=Layout((1, 128)),
)


# =============================================================================
# SM90 Copy atoms — STMATRIX
# Source: include/cute/atom/copy_traits_sm90.hpp
# PTX:    stmatrix.sync.aligned.{x1,x2,x4}[.trans].m8n8.shared.b16
#         PTX ISA §9.7.12.5
# =============================================================================

SM90_U32x1_STSM_N = CopyAtom(
    name="SM90_U32x1_STSM_N",
    ptx="stmatrix.sync.aligned.x1.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U32x1_LDSM_N.dst_layout_bits,
    dst_layout_bits=SM75_U32x1_LDSM_N.src_layout_bits,
)

SM90_U32x2_STSM_N = CopyAtom(
    name="SM90_U32x2_STSM_N",
    ptx="stmatrix.sync.aligned.x2.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U32x2_LDSM_N.dst_layout_bits,
    dst_layout_bits=SM75_U32x2_LDSM_N.src_layout_bits,
)

SM90_U32x4_STSM_N = CopyAtom(
    name="SM90_U32x4_STSM_N",
    ptx="stmatrix.sync.aligned.x4.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U32x4_LDSM_N.dst_layout_bits,
    dst_layout_bits=SM75_U32x4_LDSM_N.src_layout_bits,
)

SM90_U16x2_STSM_T = CopyAtom(
    name="SM90_U16x2_STSM_T",
    ptx="stmatrix.sync.aligned.x1.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U16x2_LDSM_T.dst_layout_bits,
    dst_layout_bits=SM75_U16x2_LDSM_T.src_layout_bits,
)

SM90_U16x4_STSM_T = CopyAtom(
    name="SM90_U16x4_STSM_T",
    ptx="stmatrix.sync.aligned.x2.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U16x4_LDSM_T.dst_layout_bits,
    dst_layout_bits=SM75_U16x4_LDSM_T.src_layout_bits,
)

SM90_U16x8_STSM_T = CopyAtom(
    name="SM90_U16x8_STSM_T",
    ptx="stmatrix.sync.aligned.x4.trans.m8n8.shared.b16",
    thr_id=Layout(32),
    src_layout_bits=SM75_U16x8_LDSM_T.dst_layout_bits,
    dst_layout_bits=SM75_U16x8_LDSM_T.src_layout_bits,
)


# =============================================================================
# Convenience collections
# =============================================================================

MMA_ATOMS_SM61 = [
    SM61_1x1x4_S32S8S8S32,
    SM61_1x1x2_S32S16S16S32,
]

MMA_ATOMS_SM70 = [
    SM70_8x8x4_F16F16F16F16_TN,
    SM70_8x8x4_F16F16F16F16_NT,
    SM70_8x8x4_F16F16F16F16_NN,
    SM70_8x8x4_F16F16F16F16_TT,
    SM70_8x8x4_F32F16F16F32_TN,
    SM70_8x8x4_F32F16F16F32_NT,
    SM70_8x8x4_F32F16F16F32_NN,
    SM70_8x8x4_F32F16F16F32_TT,
]

MMA_ATOMS_SM75 = [
    SM75_16x8x8_F32F16F16F32_TN,
    SM75_8x8x16_S32S8S8S32_TN,
]

MMA_ATOMS_SM80 = [
    SM80_16x8x8_F16F16F16F16_TN,
    SM80_16x8x16_F16F16F16F16_TN,
    SM80_16x8x8_F32F16F16F32_TN,
    SM80_16x8x16_F32F16F16F32_TN,
    SM80_16x8x8_F32BF16BF16F32_TN,
    SM80_16x8x16_F32BF16BF16F32_TN,
    SM80_16x8x4_F32TF32TF32F32_TN,
    SM80_16x8x8_F32TF32TF32F32_TN,
    SM80_8x8x4_F64F64F64F64_TN,
    SM80_8x8x16_S32S8S8S32_TN,
    SM80_16x8x16_S32S8S8S32_TN,
    SM80_16x8x32_S32S8S8S32_TN,
    SM80_8x8x32_S32S4S4S32_TN,
    SM80_16x8x32_S32S4S4S32_TN,
    SM80_16x8x64_S32S4S4S32_TN,
    SM80_8x8x128_S32U1U1S32_TN_XORPOPC,
    SM80_16x8x128_S32U1U1S32_TN_XORPOPC,
    SM80_16x8x256_S32U1U1S32_TN_XORPOPC,
]

MMA_ATOMS_SM89 = [
    SM89_16x8x32_F32E4M3E4M3F32_TN,
    SM89_16x8x32_F32E4M3E5M2F32_TN,
    SM89_16x8x32_F32E5M2E5M2F32_TN,
    SM89_16x8x32_F32E5M2E4M3F32_TN,
    SM89_16x8x32_F16E4M3E4M3F16_TN,
    SM89_16x8x32_F16E4M3E5M2F16_TN,
    SM89_16x8x32_F16E5M2E5M2F16_TN,
    SM89_16x8x32_F16E5M2E4M3F16_TN,
]

MMA_ATOMS_SM90 = [
    SM90_16x8x4_F64F64F64F64_TN,
    SM90_16x8x8_F64F64F64F64_TN,
    SM90_16x8x16_F64F64F64F64_TN,
    SM90_16x8x4_C64C64C64C64_TN,
    SM90_16x8x8_C64C64C64C64_TN,
    SM90_16x8x16_C64C64C64C64_TN,
]

MMA_ATOMS_SM90_GMMA = [
    SM90_64x8x16_F16F16F16_SS,
    SM90_64x16x16_F16F16F16_SS,
    SM90_64x32x16_F16F16F16_SS,
    SM90_64x64x16_F16F16F16_SS,
    SM90_64x128x16_F16F16F16_SS,
    SM90_64x256x16_F16F16F16_SS,
]

MMA_ATOMS_SM90_GMMA_EXT = [
    SM90_64x24x16_F16F16F16_SS,
    SM90_64x48x16_F16F16F16_SS,
    SM90_64x96x16_F16F16F16_SS,
    SM90_64x192x16_F16F16F16_SS,
    SM90_64x8x16_F32F16F16_SS,
    SM90_64x16x16_F32F16F16_SS,
    SM90_64x32x16_F32F16F16_SS,
    SM90_64x64x16_F32F16F16_SS,
    SM90_64x128x16_F32F16F16_SS,
    SM90_64x256x16_F32F16F16_SS,
    SM90_64x64x16_F32BF16BF16_SS,
    SM90_64x128x16_F32BF16BF16_SS,
    SM90_64x256x16_F32BF16BF16_SS,
    SM90_64x64x8_F32TF32TF32_SS,
    SM90_64x128x8_F32TF32TF32_SS,
    SM90_64x256x8_F32TF32TF32_SS,
    SM90_64x64x32_S32S8S8_SS,
    SM90_64x128x32_S32S8S8_SS,
    SM90_64x256x32_S32S8S8_SS,
    SM90_64x64x32_F32E4M3E4M3_SS,
    SM90_64x128x32_F32E4M3E4M3_SS,
    SM90_64x256x32_F32E4M3E4M3_SS,
    SM90_64x64x32_F16E4M3E4M3_SS,
    SM90_64x128x32_F16E4M3E4M3_SS,
    SM90_64x256x32_F16E4M3E4M3_SS,
    SM90_64x64x32_F32E5M2E5M2_SS,
    SM90_64x128x32_F32E5M2E5M2_SS,
    SM90_64x256x32_F32E5M2E5M2_SS,
    SM90_64x64x32_F16E5M2E5M2_SS,
    SM90_64x128x32_F16E5M2E5M2_SS,
    SM90_64x256x32_F16E5M2E5M2_SS,
]

MMA_ATOMS_SM90_GMMA_SPARSE = [
    SM90_64x64x32_F16F16F16_SS_SPARSE,
    SM90_64x128x32_F16F16F16_SS_SPARSE,
    SM90_64x256x32_F16F16F16_SS_SPARSE,
    SM90_64x64x16_F32TF32TF32_SS_SPARSE,
    SM90_64x128x16_F32TF32TF32_SS_SPARSE,
    SM90_64x256x16_F32TF32TF32_SS_SPARSE,
    SM90_64x64x64_S32S8S8_SS_SPARSE,
    SM90_64x128x64_S32S8S8_SS_SPARSE,
    SM90_64x256x64_S32S8S8_SS_SPARSE,
]

MMA_ATOMS_SM100_UMMA = [
    SM100_64x64x16_F16F16F16_SS,
    SM100_64x128x16_F16F16F16_SS,
    SM100_64x256x16_F16F16F16_SS,
    SM100_128x64x16_F16F16F16_SS,
    SM100_128x128x16_F16F16F16_SS,
    SM100_128x256x16_F16F16F16_SS,
    SM100_64x64x16_F32F16F16_SS,
    SM100_64x128x16_F32F16F16_SS,
    SM100_64x256x16_F32F16F16_SS,
    SM100_128x64x16_F32F16F16_SS,
    SM100_128x128x16_F32F16F16_SS,
    SM100_128x256x16_F32F16F16_SS,
    SM100_64x64x16_F32BF16BF16_SS,
    SM100_64x128x16_F32BF16BF16_SS,
    SM100_64x256x16_F32BF16BF16_SS,
    SM100_128x64x16_F32BF16BF16_SS,
    SM100_128x128x16_F32BF16BF16_SS,
    SM100_128x256x16_F32BF16BF16_SS,
    SM100_64x64x16_F16BF16BF16_SS,
    SM100_64x128x16_F16BF16BF16_SS,
    SM100_64x256x16_F16BF16BF16_SS,
    SM100_128x64x16_F16BF16BF16_SS,
    SM100_128x128x16_F16BF16BF16_SS,
    SM100_128x256x16_F16BF16BF16_SS,
    SM100_64x64x8_F32TF32TF32F32_SS,
    SM100_128x128x8_F32TF32TF32F32_SS,
]

MMA_ATOMS_SM120 = [
    SM120_16x8x32_F32E4M3E4M3F32_TN,
    SM120_16x8x64_F32E4M3E4M3F32_TN,
    SM120_16x8x64_F32E4M3E4M3F32_TN_SPARSE,
    SM120_16x8x128_F32E4M3E4M3F32_TN_SPARSE,
    SM120_16x8x32_F16E4M3E4M3F16_TN,
    SM120_16x8x64_F16E4M3E4M3F16_TN,
    SM120_16x8x64_F16E4M3E4M3F16_TN_SPARSE,
    SM120_16x8x128_F16E4M3E4M3F16_TN_SPARSE,
]

COPY_ATOMS_SM50 = [
    SM50_Shuffle_U32_2x2Trans_XOR1,
    SM50_Shuffle_U32_2x2Trans_XOR4,
]

COPY_ATOMS_SM75 = [
    SM75_U32x1_LDSM_N,
    SM75_U32x2_LDSM_N,
    SM75_U32x4_LDSM_N,
    SM75_U16x2_LDSM_T,
    SM75_U16x4_LDSM_T,
    SM75_U16x8_LDSM_T,
]

COPY_ATOMS_SM80 = [
    SM80_CP_ASYNC_CACHEALWAYS_16B,
    SM80_CP_ASYNC_CACHEGLOBAL_16B,
]

COPY_ATOMS_SM90 = [
    SM90_U32x1_STSM_N,
    SM90_U32x2_STSM_N,
    SM90_U32x4_STSM_N,
    SM90_U16x2_STSM_T,
    SM90_U16x4_STSM_T,
    SM90_U16x8_STSM_T,
]
