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

"""AMD CDNA MFMA (Matrix Fused Multiply-Add) atom definitions.

Mirrors the lane-to-element mapping of AMD's MFMA instructions on CDNA
architectures (CDNA1/gfx908, CDNA2/gfx90a, CDNA3/gfx942, CDNA3+/gfx950).

Each MFMA atom maps (thread_idx, value_idx) -> element coordinate using
column-major encoding:
    A: (T, V) -> m + k*M   in the M x K matrix
    B: (T, V) -> n + k*N   in the N x K matrix
    C: (T, V) -> m + n*M   in the M x N matrix

References:
    - AMD CDNA ISA Reference Manuals:
      https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/
        instruction-set-architectures/
        amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
    - ROCm Composable Kernel (CK) source:
      include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp
      include/ck/utility/amd_xdlops.hpp
    - ROCm FlyDSL:
      https://github.com/ROCm/FlyDSL
    - AMD Matrix Instruction Calculator:
      https://github.com/ROCm/amd_matrix_instruction_calculator
    - AMD Matrix Cores blog:
      https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/


MFMA Register Layout
=====================

AMD MFMA instructions operate on a wavefront of 64 lanes. The output matrix
D (= C + A * B) is distributed across lanes and VGPRs according to a fixed
pattern determined by five structural parameters from the CK source code:

    group_size           -- consecutive rows held by one lane (always 4, except
                            FP64 which uses 1)
    num_groups_per_blk   -- number of row-groups per output block
    num_threads_per_blk  -- number of lanes per output block (= N dimension of
                            one block; each lane covers one column)
    num_input_blks       -- number of blocks the 64 lanes are split across
                            (= wave_size / num_threads_per_blk)
    num_output_blks      -- 1 for k-reduction variants, equals num_input_blks
                            for non-k-reduction

    num_regs_per_blk = group_size * num_groups_per_blk   (VGPRs per lane)

Given a lane ID `t` (0..63) and VGPR index `v` (0..num_regs_per_blk-1):

    blk_id = t // num_threads_per_blk      (which block, 0..num_input_blks-1)
    blk_td = t %  num_threads_per_blk      (thread within block = column)

    grp    = v // group_size                (which group, 0..num_groups_per_blk-1)
    g_off  = v %  group_size                (offset within group)

    row = grp * (num_input_blks * group_size) + blk_id * group_size + g_off
    col = blk_td

    col_major_offset = row + col * M

This mapping is encoded as CuTe Layout objects below. The thread dimension
captures (blk_id, blk_td) and the value dimension captures (grp, g_off),
composed so that the layout function maps (thread_idx, value_idx) to the
column-major element offset in the M x N output matrix.


Input Operand Layout (A and B)
===============================

For k-reduction MFMA variants (the common case for the instructions we care
about: 32x32x8f16, 16x16x16f16, etc.), each lane provides the *same* A/B
data to the matrix core -- i.e., the A input is broadcast across the
num_input_blks blocks. Each lane holds k_per_blk elements of A (packed in
VGPRs), and the lane ID within a block selects which row of A that lane
contributes.

For the A operand (M x K):
    row = blk_td   (lane position within block = row of A)
    col = 0..k_per_blk-1 (stored contiguously in the VGPR)

For the B operand (N x K):
    row = blk_td   (lane position within block = row of B = column of output)
    col = 0..k_per_blk-1 (stored contiguously in the VGPR)

Note: A and B are fed as packed register values (e.g., half4_t = 4 FP16
values packed into 2 VGPRs). The lane-to-row mapping for A and B is the
same: lane selects a row, VGPRs hold the K elements.

Usage:
    from tensor_layouts.atoms_amd import CDNA_32x32x8_F32F16F16_MFMA, CDNA_16x16x16_F32F16F16_MFMA
    print(CDNA_32x32x8_F32F16F16_MFMA.c_layout)
"""

from .atoms import MMAAtom
from .layouts import Layout


# =============================================================================
# Helper: construct CuTe layouts from MFMA structural parameters
# =============================================================================


def _mfma_c_layout(
    m: int,
    n: int,
    group_size: int,
    num_groups_per_blk: int,
    num_threads_per_blk: int,
    num_input_blks: int,
) -> Layout:
    """Build the (T64, V_regs) -> col-major(M, N) accumulator layout.

    The mapping from (lane_id, vgpr_idx) to matrix element (row, col):

        lane_id  -> (blk_id, blk_td) = (lane_id // num_threads_per_blk,
                                         lane_id %  num_threads_per_blk)
        vgpr_idx -> (grp, g_off)     = (vgpr_idx // group_size,
                                         vgpr_idx %  group_size)

        row = grp * (num_input_blks * group_size) + blk_id * group_size + g_off
        col = blk_td

        offset = row + col * M    (column-major)

    Thread dimension shape:  (num_input_blks, num_threads_per_blk)
    Thread dimension stride: (group_size,     m)

    Value dimension shape:   (num_groups_per_blk, group_size)
    Value dimension stride:  (num_input_blks * group_size, 1)
    """
    # Thread: (blk_id, blk_td) with sizes (num_input_blks, num_threads_per_blk)
    # Value:  (grp, g_off) with sizes (num_groups_per_blk, group_size)
    #
    # offset = g_off * 1
    #        + blk_id * group_size
    #        + grp * (num_input_blks * group_size)
    #        + blk_td * m    (column stride)
    return Layout(
        ((num_input_blks, num_threads_per_blk), (num_groups_per_blk, group_size)),
        ((group_size, m), (num_input_blks * group_size, 1)),
    )


def _mfma_input_layout(
    rows: int,
    k: int,
    num_threads_per_blk: int,
    num_input_blks: int,
    k_per_blk: int,
) -> Layout:
    """Build the (T64, V) -> col-major(rows, K) input layout for A or B.

    Both A and B use the same broadcast structure: each lane provides
    k_per_blk elements of one row, and the blk_id dimension has stride 0.

    For A, rows=M and the value stride is M.
    For B, rows=N and the value stride is N.

        row = lane_id % num_threads_per_blk
        col = vgpr_idx  (0..k_per_blk-1)

    stride-0 for blk_id (broadcast across blocks).
    """
    # Thread: (num_input_blks, num_threads_per_blk)
    # Value:  k_per_blk values per lane
    #
    # offset = (lane_id % num_threads_per_blk) * 1   (row, col-major)
    #        + vgpr_idx * rows                        (k dimension)
    # blk_id dimension has stride 0 (broadcast)
    return Layout(
        ((num_input_blks, num_threads_per_blk), k_per_blk),
        ((0, 1), rows),
    )


def make_mfma_atom(
    name: str,
    inst: str,
    m: int,
    n: int,
    k: int,
    group_size: int,
    num_groups_per_blk: int,
    num_threads_per_blk: int,
    num_input_blks: int,
    num_output_blks: int,
    k_per_blk: int,
    is_k_reduction: bool,
    num_v_a: int,
    num_v_b: int,
) -> MMAAtom:
    """Create an AMD MFMA atom with the given structural parameters.

    Parameters match the mfma_type<> struct fields from the CK source code
    (xdlops_gemm.hpp).
    """
    wave_size = 64
    num_regs_per_blk = group_size * num_groups_per_blk

    # Sanity checks matching the CK static_asserts
    if num_threads_per_blk != n:
        raise ValueError(f"num_threads_per_blk ({num_threads_per_blk}) != n ({n})")
    if num_regs_per_blk * num_input_blks != m:
        raise ValueError(
            f"num_regs_per_blk * num_input_blks "
            f"({num_regs_per_blk * num_input_blks}) != m ({m})"
        )
    if num_regs_per_blk * wave_size != m * n:
        raise ValueError(
            f"num_regs_per_blk * wave_size "
            f"({num_regs_per_blk * wave_size}) != m*n ({m * n})"
        )
    if wave_size != num_input_blks * num_threads_per_blk:
        raise ValueError(
            f"wave_size ({wave_size}) != "
            f"num_input_blks * num_threads_per_blk "
            f"({num_input_blks * num_threads_per_blk})"
        )

    # For k-reduction variants: K = k_per_blk * num_input_blks
    # For non-k-reduction: K = k_per_blk
    total_k = k_per_blk * num_input_blks if is_k_reduction else k_per_blk
    if total_k != k:
        raise ValueError(f"total_k ({total_k}) != k ({k})")

    c_layout = _mfma_c_layout(
        m,
        n,
        group_size,
        num_groups_per_blk,
        num_threads_per_blk,
        num_input_blks,
    )

    a_layout = _mfma_input_layout(
        m,
        k,
        num_threads_per_blk,
        num_input_blks,
        k_per_blk,
    )

    b_layout = _mfma_input_layout(
        n,
        k,
        num_threads_per_blk,
        num_input_blks,
        k_per_blk,
    )

    return MMAAtom(
        name=name,
        ptx=inst,
        shape_mnk=(m, n, k),
        thr_id=None,  # identity: lane_id = thread_idx % 64
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
    )


# =============================================================================
# CDNA1 (gfx908, MI100) — MFMA atoms
# =============================================================================

# --- FP32 accumulator, FP16 inputs ---

# v_mfma_f32_32x32x8f16: D[32x32] = C[32x32] + A[32x8]*B[8x32]
# 64 lanes, 16 VGPRs/lane, 1 output block (k-reduction)
# CK: group_size=4, num_groups_per_blk=4, num_threads_per_blk=32,
#     num_input_blks=2, num_output_blks=1, k_per_blk=4
CDNA_32x32x8_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA_32x32x8_F32F16F16_MFMA",
    inst="v_mfma_f32_32x32x8f16",
    m=32,
    n=32,
    k=8,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x16f16: D[16x16] = C[16x16] + A[16x16]*B[16x16]
# 64 lanes, 4 VGPRs/lane, 1 output block (k-reduction)
# CK: group_size=4, num_groups_per_blk=1, num_threads_per_blk=16,
#     num_input_blks=4, num_output_blks=1, k_per_blk=4
CDNA_16x16x16_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA_16x16x16_F32F16F16_MFMA",
    inst="v_mfma_f32_16x16x16f16",
    m=16,
    n=16,
    k=16,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_4x4x4f16: D[4x4] = C[4x4] + A[4x4]*B[4x4]
# 64 lanes, 4 VGPRs/lane, 1 output block (non-k-reduction)
# Treated as 4x64: all 64 lanes in a single block
# CK: group_size=4, num_groups_per_blk=1, num_threads_per_blk=64,
#     num_input_blks=1, num_output_blks=1, k_per_blk=4
# NOTE: CK treats this as m_per_blk=4, n_per_blk=64 (4 rows, 64 columns)
# but the ISA instruction is nominally 4x4. CK reshapes it.
CDNA_4x4x4_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA_4x4x4_F32F16F16_MFMA",
    inst="v_mfma_f32_4x4x4f16",
    m=4,
    n=64,
    k=4,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=64,
    num_input_blks=1,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=False,
    num_v_a=2,
    num_v_b=2,
)

# --- Non-k-reduction variants (larger K, multiple output blocks) ---

# v_mfma_f32_32x32x4f16: 2 output blocks (non-k-reduction)
# Each block is 32x32; 2 blocks means 64 effective M rows
# CK: group_size=4, num_groups_per_blk=4, num_threads_per_blk=32,
#     num_input_blks=2, num_output_blks=2, k_per_blk=4
CDNA_32x32x4_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA_32x32x4_F32F16F16_MFMA",
    inst="v_mfma_f32_32x32x4f16",
    m=32,
    n=32,
    k=4,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=2,
    k_per_blk=4,
    is_k_reduction=False,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x4f16: 4 output blocks (non-k-reduction)
# CK: group_size=4, num_groups_per_blk=1, num_threads_per_blk=16,
#     num_input_blks=4, num_output_blks=4, k_per_blk=4
CDNA_16x16x4_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA_16x16x4_F32F16F16_MFMA",
    inst="v_mfma_f32_16x16x4f16",
    m=16,
    n=16,
    k=4,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=4,
    k_per_blk=4,
    is_k_reduction=False,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA2 (gfx90a, MI200) — BF16_1K variants
# Same register layout as FP16 counterparts
# =============================================================================

# v_mfma_f32_32x32x8bf16_1k: identical layout to 32x32x8f16
CDNA_32x32x8_F32BF16BF16_1K_MFMA = make_mfma_atom(
    name="CDNA_32x32x8_F32BF16BF16_1K_MFMA",
    inst="v_mfma_f32_32x32x8bf16_1k",
    m=32,
    n=32,
    k=8,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x16bf16_1k: identical layout to 16x16x16f16
CDNA_16x16x16_F32BF16BF16_1K_MFMA = make_mfma_atom(
    name="CDNA_16x16x16_F32BF16BF16_1K_MFMA",
    inst="v_mfma_f32_16x16x16bf16_1k",
    m=16,
    n=16,
    k=16,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA1/2 — BF16 (non-1K, original) variants
# =============================================================================

# v_mfma_f32_32x32x4bf16: same output layout as 32x32x8f16 but k_per_blk=2
CDNA_32x32x4_F32BF16BF16_MFMA = make_mfma_atom(
    name="CDNA_32x32x4_F32BF16BF16_MFMA",
    inst="v_mfma_f32_32x32x4bf16",
    m=32,
    n=32,
    k=4,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=2,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x8bf16
CDNA_16x16x8_F32BF16BF16_MFMA = make_mfma_atom(
    name="CDNA_16x16x8_F32BF16BF16_MFMA",
    inst="v_mfma_f32_16x16x8bf16",
    m=16,
    n=16,
    k=8,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=2,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA1/2 — INT8 variants
# =============================================================================

# v_mfma_i32_32x32x8i8
CDNA_32x32x8_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA_32x32x8_I32I8I8_MFMA",
    inst="v_mfma_i32_32x32x8i8",
    m=32,
    n=32,
    k=8,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=1,
    num_v_b=1,
)

# v_mfma_i32_16x16x16i8
CDNA_16x16x16_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA_16x16x16_I32I8I8_MFMA",
    inst="v_mfma_i32_16x16x16i8",
    m=16,
    n=16,
    k=16,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=4,
    is_k_reduction=True,
    num_v_a=1,
    num_v_b=1,
)


# =============================================================================
# CDNA1/2 — FP32 variants
# =============================================================================

# v_mfma_f32_32x32x2f32
CDNA_32x32x2_F32F32F32_MFMA = make_mfma_atom(
    name="CDNA_32x32x2_F32F32F32_MFMA",
    inst="v_mfma_f32_32x32x2f32",
    m=32,
    n=32,
    k=2,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=1,
    is_k_reduction=True,
    num_v_a=1,
    num_v_b=1,
)

# v_mfma_f32_16x16x4f32
CDNA_16x16x4_F32F32F32_MFMA = make_mfma_atom(
    name="CDNA_16x16x4_F32F32F32_MFMA",
    inst="v_mfma_f32_16x16x4f32",
    m=16,
    n=16,
    k=4,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=1,
    is_k_reduction=True,
    num_v_a=1,
    num_v_b=1,
)


# =============================================================================
# CDNA2/3 — FP64 variants
# =============================================================================

# v_mfma_f64_16x16x4f64: group_size=1 (each VGPR holds one FP64 element)
CDNA_16x16x4_F64F64F64_MFMA = make_mfma_atom(
    name="CDNA_16x16x4_F64F64F64_MFMA",
    inst="v_mfma_f64_16x16x4f64",
    m=16,
    n=16,
    k=4,
    group_size=1,
    num_groups_per_blk=4,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=1,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA3 (gfx942, MI300) — Enhanced MFMA variants
# These have double the K throughput compared to CDNA2 equivalents
# =============================================================================

# --- FP16 ---

# v_mfma_f32_32x32x8_f16 (CDNA3 naming): same layout as CDNA2 32x32x8f16
# The CDNA3 version runs at 32 cycles instead of 64 (2x throughput)
CDNA3_32x32x8_F32F16F16_MFMA = CDNA_32x32x8_F32F16F16_MFMA

# v_mfma_f32_16x16x16_f16 (CDNA3): same layout as CDNA2 16x16x16f16
CDNA3_16x16x16_F32F16F16_MFMA = CDNA_16x16x16_F32F16F16_MFMA

# --- BF16 ---

# v_mfma_f32_32x32x8_bf16 (CDNA3): same layout as BF16_1K
CDNA3_32x32x8_F32BF16BF16_MFMA = CDNA_32x32x8_F32BF16BF16_1K_MFMA

# v_mfma_f32_16x16x16_bf16 (CDNA3): same layout as BF16_1K
CDNA3_16x16x16_F32BF16BF16_MFMA = CDNA_16x16x16_F32BF16BF16_1K_MFMA

# --- INT8 (CDNA3-enhanced, doubled K) ---

# v_mfma_i32_32x32x16i8: 32x32 output, K=16
# CK: group_size=4, num_groups_per_blk=4, num_threads_per_blk=32,
#     num_input_blks=2, num_output_blks=1, k_per_blk=8
CDNA3_32x32x16_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA3_32x32x16_I32I8I8_MFMA",
    inst="v_mfma_i32_32x32x16i8",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_i32_16x16x32i8: 16x16 output, K=32
# CK: group_size=4, num_groups_per_blk=1, num_threads_per_blk=16,
#     num_input_blks=4, num_output_blks=1, k_per_blk=8
CDNA3_16x16x32_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA3_16x16x32_I32I8I8_MFMA",
    inst="v_mfma_i32_16x16x32i8",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# --- XF32 (TF32-like, CDNA3) ---

# v_mfma_f32_32x32x4_xf32
CDNA3_32x32x4_F32XF32XF32_MFMA = make_mfma_atom(
    name="CDNA3_32x32x4_F32XF32XF32_MFMA",
    inst="v_mfma_f32_32x32x4_xf32",
    m=32,
    n=32,
    k=4,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=2,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x8_xf32
CDNA3_16x16x8_F32XF32XF32_MFMA = make_mfma_atom(
    name="CDNA3_16x16x8_F32XF32XF32_MFMA",
    inst="v_mfma_f32_16x16x8_xf32",
    m=16,
    n=16,
    k=8,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=2,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA3 (gfx942) — FP8 variants
# =============================================================================

# v_mfma_f32_32x32x16_fp8_fp8
CDNA3_32x32x16_F32F8F8_MFMA = make_mfma_atom(
    name="CDNA3_32x32x16_F32F8F8_MFMA",
    inst="v_mfma_f32_32x32x16_fp8_fp8",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x32_fp8_fp8
CDNA3_16x16x32_F32F8F8_MFMA = make_mfma_atom(
    name="CDNA3_16x16x32_F32F8F8_MFMA",
    inst="v_mfma_f32_16x16x32_fp8_fp8",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_32x32x16_bf8_bf8: same layout as fp8_fp8 32x32
CDNA3_32x32x16_F32BF8BF8_MFMA = make_mfma_atom(
    name="CDNA3_32x32x16_F32BF8BF8_MFMA",
    inst="v_mfma_f32_32x32x16_bf8_bf8",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x32_bf8_bf8
CDNA3_16x16x32_F32BF8BF8_MFMA = make_mfma_atom(
    name="CDNA3_16x16x32_F32BF8BF8_MFMA",
    inst="v_mfma_f32_16x16x32_bf8_bf8",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# Mixed FP8 variants (fp8 x bf8, bf8 x fp8) — same layouts
CDNA3_32x32x16_F32F8BF8_MFMA = make_mfma_atom(
    name="CDNA3_32x32x16_F32F8BF8_MFMA",
    inst="v_mfma_f32_32x32x16_fp8_bf8",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

CDNA3_16x16x32_F32F8BF8_MFMA = make_mfma_atom(
    name="CDNA3_16x16x32_F32F8BF8_MFMA",
    inst="v_mfma_f32_16x16x32_fp8_bf8",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

CDNA3_32x32x16_F32BF8F8_MFMA = make_mfma_atom(
    name="CDNA3_32x32x16_F32BF8F8_MFMA",
    inst="v_mfma_f32_32x32x16_bf8_fp8",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

CDNA3_16x16x32_F32BF8F8_MFMA = make_mfma_atom(
    name="CDNA3_16x16x32_F32BF8F8_MFMA",
    inst="v_mfma_f32_16x16x32_bf8_fp8",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# CDNA3+ (gfx950, MI325/MI350) — Double-rate MFMA variants
# =============================================================================

# v_mfma_f32_32x32x16_f16 (gfx950 only): 2x K vs 32x32x8f16
# CK: group_size=4, num_groups_per_blk=4, num_threads_per_blk=32,
#     num_input_blks=2, num_output_blks=1, k_per_blk=8
CDNA3P_32x32x16_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA3P_32x32x16_F32F16F16_MFMA",
    inst="v_mfma_f32_32x32x16_f16",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x32_f16 (gfx950 only): 2x K vs 16x16x16f16
# CK: group_size=4, num_groups_per_blk=1, num_threads_per_blk=16,
#     num_input_blks=4, num_output_blks=1, k_per_blk=8
CDNA3P_16x16x32_F32F16F16_MFMA = make_mfma_atom(
    name="CDNA3P_16x16x32_F32F16F16_MFMA",
    inst="v_mfma_f32_16x16x32_f16",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_32x32x16_bf16 (gfx950 only)
# CK: group_size=4, num_groups_per_blk=4, num_threads_per_blk=32,
#     num_input_blks=2, num_output_blks=1, k_per_blk=8
CDNA3P_32x32x16_F32BF16BF16_MFMA = make_mfma_atom(
    name="CDNA3P_32x32x16_F32BF16BF16_MFMA",
    inst="v_mfma_f32_32x32x16_bf16",
    m=32,
    n=32,
    k=16,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_f32_16x16x32_bf16 (gfx950 only)
CDNA3P_16x16x32_F32BF16BF16_MFMA = make_mfma_atom(
    name="CDNA3P_16x16x32_F32BF16BF16_MFMA",
    inst="v_mfma_f32_16x16x32_bf16",
    m=16,
    n=16,
    k=32,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=8,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_i32_32x32x32_i8 (gfx950 only)
CDNA3P_32x32x32_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA3P_32x32x32_I32I8I8_MFMA",
    inst="v_mfma_i32_32x32x32i8",
    m=32,
    n=32,
    k=32,
    group_size=4,
    num_groups_per_blk=4,
    num_threads_per_blk=32,
    num_input_blks=2,
    num_output_blks=1,
    k_per_blk=16,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)

# v_mfma_i32_16x16x64_i8 (gfx950 only)
CDNA3P_16x16x64_I32I8I8_MFMA = make_mfma_atom(
    name="CDNA3P_16x16x64_I32I8I8_MFMA",
    inst="v_mfma_i32_16x16x64i8",
    m=16,
    n=16,
    k=64,
    group_size=4,
    num_groups_per_blk=1,
    num_threads_per_blk=16,
    num_input_blks=4,
    num_output_blks=1,
    k_per_blk=16,
    is_k_reduction=True,
    num_v_a=2,
    num_v_b=2,
)


# =============================================================================
# Convenience collections
# =============================================================================

# Most commonly used MFMA atoms (k-reduction variants for GEMM)
MMA_ATOMS_CDNA_FP16 = [
    CDNA_32x32x8_F32F16F16_MFMA,
    CDNA_16x16x16_F32F16F16_MFMA,
    CDNA_4x4x4_F32F16F16_MFMA,
]

MMA_ATOMS_CDNA_BF16 = [
    CDNA_32x32x8_F32BF16BF16_1K_MFMA,
    CDNA_16x16x16_F32BF16BF16_1K_MFMA,
    CDNA_32x32x4_F32BF16BF16_MFMA,
    CDNA_16x16x8_F32BF16BF16_MFMA,
]

MMA_ATOMS_CDNA_INT8 = [
    CDNA_32x32x8_I32I8I8_MFMA,
    CDNA_16x16x16_I32I8I8_MFMA,
]

MMA_ATOMS_CDNA_FP32 = [
    CDNA_32x32x2_F32F32F32_MFMA,
    CDNA_16x16x4_F32F32F32_MFMA,
]

MMA_ATOMS_CDNA_FP64 = [
    CDNA_16x16x4_F64F64F64_MFMA,
]

MMA_ATOMS_CDNA3_INT8 = [
    CDNA3_32x32x16_I32I8I8_MFMA,
    CDNA3_16x16x32_I32I8I8_MFMA,
]

MMA_ATOMS_CDNA3_FP8 = [
    CDNA3_32x32x16_F32F8F8_MFMA,
    CDNA3_16x16x32_F32F8F8_MFMA,
    CDNA3_32x32x16_F32BF8BF8_MFMA,
    CDNA3_16x16x32_F32BF8BF8_MFMA,
    CDNA3_32x32x16_F32F8BF8_MFMA,
    CDNA3_16x16x32_F32F8BF8_MFMA,
    CDNA3_32x32x16_F32BF8F8_MFMA,
    CDNA3_16x16x32_F32BF8F8_MFMA,
]

MMA_ATOMS_CDNA3_XF32 = [
    CDNA3_32x32x4_F32XF32XF32_MFMA,
    CDNA3_16x16x8_F32XF32XF32_MFMA,
]

MMA_ATOMS_CDNA3P = [
    CDNA3P_32x32x16_F32F16F16_MFMA,
    CDNA3P_16x16x32_F32F16F16_MFMA,
    CDNA3P_32x32x16_F32BF16BF16_MFMA,
    CDNA3P_16x16x32_F32BF16BF16_MFMA,
    CDNA3P_32x32x32_I32I8I8_MFMA,
    CDNA3P_16x16x64_I32I8I8_MFMA,
]
