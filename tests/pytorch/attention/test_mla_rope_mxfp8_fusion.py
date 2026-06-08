# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Prototype tests for fused MLA RoPE + MXFP8 fprop quantization."""

import os
import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    mxfp8_quantize_fast_path,
)

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent)] + sys.path
from mla_rope_utils import (  # noqa: E402
    HEAD_DIM_NOPE,
    HEAD_DIM_ROPE,
    HEAD_DIM_V,
    HAVE_TRITON,
    apply_mla_rope,
    apply_mla_rope_mxfp8_quantize,
    build_rope_tables,
)

NUM_HEADS = 128
HEAD_DIM_QK = HEAD_DIM_NOPE + HEAD_DIM_ROPE
SEED = 1234

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)


def _set_seed() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def _make_inputs(seq_len: int, batch_size: int = 1):
    q = torch.randn(
        seq_len,
        batch_size,
        NUM_HEADS,
        HEAD_DIM_QK,
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv = torch.randn(
        seq_len,
        batch_size,
        NUM_HEADS,
        HEAD_DIM_NOPE + HEAD_DIM_V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_pos_emb = torch.randn(
        seq_len,
        batch_size,
        1,
        HEAD_DIM_ROPE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rope_tables = build_rope_tables(seq_len, device=q.device)
    return q, kv, k_pos_emb, rope_tables


def _make_fprop_quantizers():
    q_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    k_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
    v_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    return q_quantizer, k_quantizer, v_quantizer


def _reference_rope_quantize(q, kv, k_pos_emb, rope_tables):
    q_rope, k_rope, v_rope = apply_mla_rope(
        q,
        kv,
        k_pos_emb,
        cos_table=rope_tables[0],
        sin_table=rope_tables[1],
    )
    quantizers = _make_fprop_quantizers()
    (q_fp8, k_fp8, v_fp8), scale_inv_format = mxfp8_quantize_fast_path(
        [(q_rope, quantizers[0]), (k_rope, quantizers[1]), (v_rope, quantizers[2])],
        "sbhd",
    )
    return q_fp8, k_fp8, v_fp8, scale_inv_format


def _fused_rope_quantize(q, kv, k_pos_emb, rope_tables):
    return apply_mla_rope_mxfp8_quantize(
        q,
        kv,
        k_pos_emb,
        cos_table=rope_tables[0],
        sin_table=rope_tables[1],
    )


def _fused_rope_quantize_for_dequant(q, kv, k_pos_emb, rope_tables):
    return apply_mla_rope_mxfp8_quantize(
        q,
        kv,
        k_pos_emb,
        cos_table=rope_tables[0],
        sin_table=rope_tables[1],
        swizzle_scales=False,
    )


def _assert_dequant_close(name: str, fused, ref) -> None:
    diff = (fused.float() - ref.float()).abs()
    print(
        f"\n[{name}] fused-vs-reference:"
        f" max_abs={diff.max().item():.6f}"
        f" rms={diff.pow(2).mean().sqrt().item():.6f}"
    )
    torch.testing.assert_close(fused, ref, atol=1.0, rtol=0.1)


def _dequantize_attention_mxfp8(tensor, *, columnwise: bool = False) -> torch.Tensor:
    if columnwise:
        data = tensor._columnwise_data.view(dtype=torch.float8_e4m3fn).float()
        scale = tensor._columnwise_scale_inv.view(dtype=torch.float8_e8m0fnu).float()
        scale = scale.view(1, NUM_HEADS, -1, HEAD_DIM_V).permute(2, 0, 1, 3)
        scale = scale.repeat_interleave(32, dim=0)[: data.shape[0]]
    else:
        data = tensor._rowwise_data.view(dtype=torch.float8_e4m3fn).float()
        scale = tensor._rowwise_scale_inv.view(dtype=torch.float8_e8m0fnu).float()
        scale = scale.view(1, NUM_HEADS, data.shape[0], -1).permute(2, 0, 1, 3)
        scale = scale.repeat_interleave(32, dim=-1)[..., : data.shape[-1]]
    return data * scale


def _time_cuda_ms(fn, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_fused_mla_rope_mxfp8_quantize_matches_reference() -> None:
    _set_seed()
    q, kv, k_pos_emb, rope_tables = _make_inputs(seq_len=128)

    q_ref, k_ref, v_ref = apply_mla_rope(
        q.clone(),
        kv,
        k_pos_emb,
        cos_table=rope_tables[0],
        sin_table=rope_tables[1],
    )
    q_fused, k_fused, v_fused, fused_format = _fused_rope_quantize_for_dequant(
        q,
        kv,
        k_pos_emb,
        rope_tables,
    )

    assert fused_format == "bhsd"
    assert not q_fused._with_gemm_swizzled_scales
    assert not k_fused._with_gemm_swizzled_scales
    assert not v_fused._with_gemm_swizzled_scales
    assert q_fused._rowwise_data is not None and q_fused._columnwise_data is None
    assert k_fused._rowwise_data is not None and k_fused._columnwise_data is None
    assert v_fused._rowwise_data is None and v_fused._columnwise_data is not None

    _assert_dequant_close("Q", _dequantize_attention_mxfp8(q_fused), q_ref.float())
    _assert_dequant_close("K", _dequantize_attention_mxfp8(k_fused), k_ref.float())
    _assert_dequant_close("V", _dequantize_attention_mxfp8(v_fused, columnwise=True), v_ref.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.skipif(
    os.getenv("NVTE_RUN_MLA_ROPE_MXFP8_FUSION_PERF", "0") != "1",
    reason="Set NVTE_RUN_MLA_ROPE_MXFP8_FUSION_PERF=1 to run the DSv3-sized timing check.",
)
def test_fused_mla_rope_mxfp8_quantize_is_faster() -> None:
    _set_seed()
    q, kv, k_pos_emb, rope_tables = _make_inputs(seq_len=4096)
    q_ref_work = q.clone()

    ref_ms = _time_cuda_ms(lambda: _reference_rope_quantize(q_ref_work, kv, k_pos_emb, rope_tables))
    fused_ms = _time_cuda_ms(lambda: _fused_rope_quantize(q, kv, k_pos_emb, rope_tables))
    speedup = ref_ms / fused_ms
    print(
        f"\n[PERF] MLA RoPE + MXFP8 quantize:"
        f"\n  reference: {ref_ms:.3f} ms"
        f"\n  fused:     {fused_ms:.3f} ms"
        f"\n  speedup:   {speedup:.2f}x"
    )
    assert fused_ms < ref_ms
