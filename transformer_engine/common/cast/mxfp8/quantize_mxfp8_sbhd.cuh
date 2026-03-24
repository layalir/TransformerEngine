/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_sbhd.cuh
 *  \brief CUDA kernel for fused SBHD->BHSD permutation + MXFP8 quantization.
 *
 *  This kernel takes a 4D tensor in SBHD (or BSHD) memory layout, reads it
 *  in BHSD logical order using address remapping, and produces MXFP8-quantized
 *  output in BHSD-contiguous format. This eliminates the need for a separate
 *  BF16/FP16 permute+contiguous copy before MXFP8 quantization.
 *
 *  The kernel processes 128x128 tiles (same tile structure as the existing
 *  group_quantize_mxfp8_kernel) and computes rowwise and/or columnwise MXFP8
 *  block scales. Input is loaded cooperatively via direct global memory reads
 *  with SBHD->BHSD address remapping. Output is written via TMA.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP8_SBHD_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP8_SBHD_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace sbhd_kernel {

using namespace dispatch::common;

// Reuse the same tile geometry as group_quantize_mxfp8_kernel
constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;

constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;

constexpr size_t THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;

constexpr size_t BUFF_DIM_Y = THREADS_Y;
constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;
constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
static_assert(BUFF_DIM_Y == 32);

constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
static_assert(STAGES >= 1);

constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4) / 1;  // 128
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 4

// Enum for the source memory layout
enum class SrcLayout {
  SBHD = 0,  // Physical memory layout: [S, B, H, D]
  BSHD = 1,  // Physical memory layout: [B, S, H, D]
};

/*! \brief Load a BUFF_DIM_Y x BUFF_DIM_X tile from SBHD/BSHD source into shared memory
 *         in BHSD row-major order.
 *
 *  The tile spans rows [row_start, row_start + BUFF_DIM_Y) and
 *  columns [col_start, col_start + BUFF_DIM_X) in the BHSD-flattened 2D view.
 *
 *  BHSD flattened 2D: rows = B*H*S, cols = D.
 *  For a given flattened row r: b = r / (H*S), h = (r / S) % H, s = r % S.
 */
template <typename IType, SrcLayout LAYOUT>
__device__ __forceinline__ void cooperative_load_sbhd_tile(
    IType *__restrict__ shmem_buf,
    const IType *__restrict__ src,
    const size_t row_start,
    const size_t col_start,
    const size_t S, const size_t B, const size_t H, const size_t D,
    const size_t total_rows,
    const size_t total_cols) {
  // Total elements to load: BUFF_DIM_Y * BUFF_DIM_X = 32 * 128 = 4096
  // With 128 threads, each thread loads 32 elements.
  constexpr size_t ELTS_PER_THREAD = (BUFF_DIM_Y * BUFF_DIM_X) / THREADS_PER_CHUNK;
  static_assert(ELTS_PER_THREAD == 32);

  const size_t HS = H * S;

  for (size_t i = 0; i < ELTS_PER_THREAD; ++i) {
    const size_t flat_idx = threadIdx.x * ELTS_PER_THREAD + i;
    const size_t local_row = flat_idx / BUFF_DIM_X;
    const size_t local_col = flat_idx % BUFF_DIM_X;

    const size_t global_row = row_start + local_row;
    const size_t global_col = col_start + local_col;

    IType val = static_cast<IType>(0);
    if (global_row < total_rows && global_col < total_cols) {
      // Decompose BHSD-flattened row into (b, h, s)
      const size_t b = global_row / HS;
      const size_t rem = global_row % HS;
      const size_t h = rem / S;
      const size_t s = rem % S;
      const size_t d = global_col;

      size_t src_offset;
      if constexpr (LAYOUT == SrcLayout::SBHD) {
        // Physical: [S, B, H, D]
        src_offset = s * (B * H * D) + b * (H * D) + h * D + d;
      } else {
        // Physical: [B, S, H, D]
        src_offset = b * (S * H * D) + s * (H * D) + h * D + d;
      }
      val = __ldg(&src[src_offset]);
    }
    shmem_buf[flat_idx] = val;
  }
}

/*! \brief Fused SBHD->BHSD permutation + MXFP8 quantization kernel.
 *
 *  Processes a single tensor. This is designed for attention Q/K/V which
 *  are quantized individually.
 *
 *  Grid: (blocks_X, blocks_Y) where blocks_X = ceil(D/128), blocks_Y = ceil(B*H*S/128)
 *  Block: THREADS_PER_CHUNK (128) threads
 *
 *  This kernel mirrors the processing logic of group_quantize_mxfp8_kernel exactly
 *  (same tile structure, scale computation, bank-conflict avoidance swizzle, output TMA),
 *  but replaces TMA input loads with cooperative __ldg reads from remapped addresses.
 *
 *  Restrictions vs. group_quantize_mxfp8_kernel:
 *  - No IS_DBIAS, IS_DACT, IS_ACT (all false -- pure cast, no fused activation)
 *  - Single tensor only (no grouped tensor offsets)
 *  - No noop flag
 */
template <typename IType, typename OType, bool ROWWISE_SCALING,
          bool COLWISE_SCALING, bool WITH_GEMM_SWIZZLED_SCALES, SrcLayout LAYOUT>
__global__ void __launch_bounds__(THREADS_PER_CHUNK) quantize_mxfp8_sbhd_kernel(
    const IType *const __restrict__ input_ptr,
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
    const __grid_constant__ CUtensorMap tensor_map_output_colwise,
    const size_t S, const size_t B, const size_t H, const size_t D,
    e8m0_t *const __restrict__ scales_rowwise_ptr,
    e8m0_t *const __restrict__ scales_colwise_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;
  using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;

  // Total BHSD-flattened 2D shape
  const size_t total_rows = B * H * S;  // first_logical_dim
  const size_t total_cols = D;          // last_logical_dim

  const size_t block_id_Y = blockIdx.y;
  const size_t block_id_X = blockIdx.x;

  const size_t block_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const size_t block_offset_X = block_id_X * CHUNK_DIM_X;

  // Early exit if block is completely out of bounds
  if (block_offset_Y >= total_rows || block_offset_X >= total_cols) {
    return;
  }

  const bool leading_thread = (threadIdx.x == 0);

  // Scale strides (same as existing kernel for a single tensor with cols=D, rows=B*H*S)
  const size_t scale_stride_rowwise = DIVUP_TO_MULTIPLE(DIVUP(total_cols, static_cast<size_t>(32)), 4);
  const size_t scale_stride_colwise = DIVUP_TO_MULTIPLE(total_cols, 128);

  // Thread indexing for rowwise processing
  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X;
  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;

  // Scale offsets
  const size_t scales_block_offset_Y_rowwise = block_id_Y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = block_id_X * CHUNK_DIM_X / SCALE_DIM_X;
  const size_t scales_block_offset_Y_colwise = block_id_Y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const size_t scales_block_offset_X_colwise = block_id_X * CHUNK_DIM_X;

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise;
  const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + threadIdx.x;

  // Bank conflict avoidance
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  // Shared memory layout
  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t in_mem = buff_size_aligned_in;
  constexpr size_t out_mem_rowwise = (ROWWISE_SCALING ? buff_size_aligned_out : 0);

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char *dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  OType *out_rowwise_data_sh = reinterpret_cast<OType *>(dshmem + in_mem);
  OType *out_colwise_data_sh = reinterpret_cast<OType *>(dshmem + in_mem + out_mem_rowwise);

  // Process STAGES of BUFF_DIM_Y rows each
#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;
    const size_t buff_offset = buff * BUFF_DIM;

    // ===== COOPERATIVE INPUT LOAD =====
    // Load a BUFF_DIM_Y x BUFF_DIM_X tile from SBHD/BSHD input into shared memory
    cooperative_load_sbhd_tile<IType, LAYOUT>(
        &in_sh[buff_offset], input_ptr,
        block_offset_Y + stage_offset_Y, block_offset_X,
        S, B, H, D,
        total_rows, total_cols);

    __syncthreads();  // Ensure all loads are visible to all threads

    // ===== COLWISE SCALING =====
    // Colwise: each thread handles one column (threadIdx.x), iterates over BUFF_DIM_Y=32 rows
    // This produces one scaling factor per column block of 32 consecutive rows.
    if constexpr (COLWISE_SCALING) {
      const size_t shmem_offset_base_colwise = buff_offset + threadIdx.x;
      float thread_amax = 0.0f;
      float in_compute_colwise[BUFF_DIM_Y];

      // 1. Read elements from shmem and find MXFP8-block AMAX (colwise block = 32 rows x 1 col)
      if constexpr (!std::is_same_v<IType, float>) {
        IType thread_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
        for (int i = 0; i < BUFF_DIM_Y; ++i) {
          const size_t shmem_offset = shmem_offset_base_colwise + i * BUFF_DIM_X;
          IType ival = in_sh[shmem_offset];
          thread_amax_f16 = __hmax(thread_amax_f16, __habs(ival));
          in_compute_colwise[i] = static_cast<float>(ival);
        }
        thread_amax = static_cast<float>(thread_amax_f16);
      } else {
#pragma unroll
        for (int i = 0; i < BUFF_DIM_Y; ++i) {
          const size_t shmem_offset = shmem_offset_base_colwise + i * BUFF_DIM_X;
          float elt = in_sh[shmem_offset];
          thread_amax = fmaxf(thread_amax, fabsf(elt));
          in_compute_colwise[i] = elt;
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);

      const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const size_t global_scales_offset_X = scales_offset_X_colwise;

      size_t scale_idx = 0;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = gemm_swizzled_scale_idx(global_scales_offset_X, global_scales_offset_Y,
                                            DIVUP(total_rows, static_cast<size_t>(128)));
      } else {
        scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      }
      scales_colwise_ptr[scale_idx] = biased_exponent;

      const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);

      // 3. Scale elements and store to colwise output shmem
#pragma unroll
      for (int i = 0; i < SCALE_DIM_Y; ++i) {
        const float scaled_out = in_compute_colwise[i] * block_scale_inverse;
        const size_t shmem_offset = shmem_offset_base_colwise + i * BUFF_DIM_X;
        out_colwise_data_sh[shmem_offset] = static_cast<OType>(scaled_out);
      }
    }

    // ===== ROWWISE SCALING =====
    // Rowwise: each thread handles one row (tid_Y_rowwise), iterates over SCALE_DIM_X=32 cols
    // in WAVES of PACK_SIZE=4 with bank-conflict-avoidance swizzling.
    if constexpr (ROWWISE_SCALING) {
      const size_t shmem_offset_base_rowwise =
          buff_offset + thread_offset_Y_rowwise * BUFF_DIM_X;
      float thread_amax = 0.0f;
      float in_compute_rowwise[SCALE_DIM_X];

      // Used as IType container for BF16/FP16 fast path
      Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

      // 1. Read elements. Find MXFP8-block AMAX
      if constexpr (!std::is_same_v<IType, float>) {
        IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;
          in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 2; ++e) {
            ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
          }
        }
        thread_amax =
            static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
      } else {
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;
          Vec<IType, PACK_SIZE> in;
          in.load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
          for (int e = 0; e < PACK_SIZE; ++e) {
            const int j = w * PACK_SIZE + e;
            float elt = in.data.elt[e];
            thread_amax = fmaxf(thread_amax, fabsf(elt));
            in_compute_rowwise[j] = elt;
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
      const size_t stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
      const size_t stage_scales_offset_X = scales_offset_X_rowwise;

      size_t scale_idx = 0;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = gemm_swizzled_scale_idx(stage_scales_offset_Y, stage_scales_offset_X,
                                            DIVUP(total_cols, static_cast<size_t>(128)));
      } else {
        scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
      }
      scales_rowwise_ptr[scale_idx] = biased_exponent;

      const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      const ptx::floatx2 block_scale_inverse_2x = {block_scale_inverse, block_scale_inverse};

      // 3. Scale elements and store to rowwise output shmem
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        Vec<OType2, PACK_SIZE / 2> out;
#pragma unroll
        for (int e = 0; e < PACK_SIZE / 2; ++e) {
          OType2 &out_pair = reinterpret_cast<OType2 &>(out.data.elt[e]);
          if constexpr (!std::is_same_v<IType, float>) {
            IType2 in = in_IType[w].data.elt[e];
            ptx::mul_cvt_2x(out_pair, in, block_scale_inverse_2x);
          } else {
            const int j = w * PACK_SIZE + 2 * e;
            ptx::floatx2 in_f2 = {in_compute_rowwise[j], in_compute_rowwise[j + 1]};
            ptx::mul_cvt_2x(out_pair, in_f2, block_scale_inverse_2x);
          }
        }
        const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
        const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
        const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_idx;
        out.store_to(&out_rowwise_data_sh[shmem_offset_rowwise]);
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    // Initiate TMA transfer to copy shared memory to global memory
    if (leading_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_rowwise_data_sh[buff_offset]));
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_colwise_data_sh[buff_offset]));
      }

      ptx::cp_async_bulk_commit_group();
    }

    // Wait for output TMA writes to complete before reusing the buffer
    // For the last stage, we still need to wait to ensure writes are flushed
    ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
  }

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace sbhd_kernel

//! Host-side launcher for the fused SBHD->BHSD MXFP8 quantization kernel.
//! This handles TMA descriptor creation and kernel launch.
template <sbhd_kernel::SrcLayout LAYOUT>
void quantize_mxfp8_sbhd(
    const Tensor &input,
    Tensor *output,
    size_t S, size_t B, size_t H, size_t D,
    bool use_rowwise,
    bool use_colwise,
    bool with_gemm_swizzled_scales,
    cudaStream_t stream) {
  using namespace sbhd_kernel;

  checkCuDriverContext(stream);

  const size_t total_rows = B * H * S;
  const size_t total_cols = D;

  NVTE_CHECK(total_rows % 128 == 0,
             "B*H*S must be divisible by 128 for MXFP8 quantization. Got ", total_rows);
  NVTE_CHECK(total_cols % 32 == 0,
             "D must be divisible by 32 for MXFP8 quantization. Got ", total_cols);
  NVTE_CHECK(use_rowwise || use_colwise,
             "Either rowwise or columnwise scaling must be enabled.");

  e8m0_t *scales_rowwise_ptr = reinterpret_cast<e8m0_t *>(output->scale_inv.dptr);
  e8m0_t *scales_colwise_ptr = reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr);

  if (use_rowwise) {
    NVTE_CHECK(scales_rowwise_ptr != nullptr, "Rowwise scaling tensor must be allocated");
  }
  if (use_colwise) {
    NVTE_CHECK(scales_colwise_ptr != nullptr, "Columnwise scaling tensor must be allocated");
  }

  const size_t blocks_Y = DIVUP(total_rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(total_cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_PER_CHUNK;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,

              constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
              constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

              constexpr size_t buff_elems_total = BUFFS_NUM * BUFF_DIM_Y * BUFF_DIM_X;
              constexpr size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
              constexpr size_t output_buff_size = (buff_elems_total * output_type_bit_size) / 8;
              constexpr size_t buff_size_aligned_in =
                  DIVUP_TO_MULTIPLE(input_buff_size, TMA_SHMEM_ALIGNMENT);
              constexpr size_t buff_size_aligned_out =
                  DIVUP_TO_MULTIPLE(output_buff_size, TMA_SHMEM_ALIGNMENT);

              constexpr size_t in_mem = buff_size_aligned_in;
              const size_t out_rowwise_mem = use_rowwise ? buff_size_aligned_out : 0;
              const size_t out_colwise_mem = use_colwise ? buff_size_aligned_out : 0;
              const size_t dshmem_size = in_mem + out_rowwise_mem + out_colwise_mem + TMA_SHMEM_ALIGNMENT;

              // Create TMA descriptors for output
              alignas(64) CUtensorMap tensor_map_output_rowwise{};
              alignas(64) CUtensorMap tensor_map_output_colwise{};

              if (use_rowwise) {
                create_2D_tensor_map(tensor_map_output_rowwise, output->data,
                                     total_rows, total_cols, BUFF_DIM_Y, BUFF_DIM_X,
                                     total_cols, 0, output_type_bit_size);
              }
              if (use_colwise) {
                create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data,
                                     total_rows, total_cols, BUFF_DIM_Y, BUFF_DIM_X,
                                     total_cols, 0, output_type_bit_size);
              }

              const IType *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);

              ScalingType scaling_type = ScalingType::BIDIMENSIONAL;
              if (!use_colwise) scaling_type = ScalingType::ROWWISE;
              else if (!use_rowwise) scaling_type = ScalingType::COLWISE;

              // Select kernel variant
              auto kernel = quantize_mxfp8_sbhd_kernel<IType, OType, true, true,
                                                        WITH_GEMM_SWIZZLED_SCALES, LAYOUT>;
              switch (scaling_type) {
                case ScalingType::ROWWISE:
                  kernel = quantize_mxfp8_sbhd_kernel<IType, OType, true, false,
                                                       WITH_GEMM_SWIZZLED_SCALES, LAYOUT>;
                  break;
                case ScalingType::COLWISE:
                  kernel = quantize_mxfp8_sbhd_kernel<IType, OType, false, true,
                                                       WITH_GEMM_SWIZZLED_SCALES, LAYOUT>;
                  break;
                case ScalingType::BIDIMENSIONAL:
                  kernel = quantize_mxfp8_sbhd_kernel<IType, OType, true, true,
                                                       WITH_GEMM_SWIZZLED_SCALES, LAYOUT>;
                  break;
              }

              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  input_ptr,
                  tensor_map_output_rowwise,
                  tensor_map_output_colwise,
                  S, B, H, D,
                  scales_rowwise_ptr,
                  scales_colwise_ptr);

              NVTE_CHECK_CUDA(cudaGetLastError());
          );  // NOLINT(*)
      );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_SBHD_CUH_
