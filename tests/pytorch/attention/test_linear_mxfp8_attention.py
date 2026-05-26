# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 end-to-end attention unit test - DSv3 671B MLA dimensions.

Path: Linear(QKV, MXFP8) -> MLA-RoPE (Triton) -> DotProductAttention(MXFP8) -> Linear(out, MXFP8).
Tensor layout: sbhd (seq-first) throughout.

Run:
    python3 -m pytest tests/pytorch/attention/test_linear_mxfp8_attention.py -v -s

Expected benchmark output (GB200, b=1, s=4096):
    [PERF] b=1 s=4096:
      BF16 fprop:   8.582 ms  (477274 tok/s)
      BF16 bprop:  14.006 ms  (292445 tok/s)
      MXFP8 fprop:  5.210 ms  (786180 tok/s)
      MXFP8 bprop:  8.763 ms  (467428 tok/s)
      Fprop speedup: 1.65x
      Bprop speedup: 1.60x
"""

import os
import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.utils import get_cudnn_version

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import ModelConfig, get_available_attention_backends
from mla_rope_utils import apply_mla_rope


try:
    from transformer_engine.common.recipe import MXFP8BlockScaling

    mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
except (ImportError, AttributeError):
    mxfp8_available = False
    reason_for_no_mxfp8 = "MXFP8BlockScaling not available in this build"

# DSv3 671B MLA dims (micro_batch=1, seq_len=4096)
NUM_HEADS = 128
HEAD_DIM_ROPE = 64
HEAD_DIM_NOPE = 128
HEAD_DIM_QK = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 192
HEAD_DIM_V = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM_V  # 16384
QKV_SIZE = NUM_HEADS * (2 * HEAD_DIM_QK + HEAD_DIM_V)  # 65536
SEED = 42

WARMUP_ITERS = 10
TIMED_ITERS = 100
_DETERMINISTIC = (
    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
    or torch.are_deterministic_algorithms_enabled()
)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _build_modules(dtype: torch.dtype = torch.bfloat16):
    def _make_triple():
        qkv = te.Linear(HIDDEN_SIZE, QKV_SIZE, bias=True).to(dtype=dtype, device="cuda")
        dpa = te.DotProductAttention(
            num_attention_heads=NUM_HEADS,
            kv_channels=(HEAD_DIM_QK, HEAD_DIM_V),
            attention_dropout=0.0,
            qkv_format="sbhd",
        ).to(device="cuda")
        out = te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).to(dtype=dtype, device="cuda")
        return qkv, dpa, out

    base = _make_triple()
    mxfp8 = _make_triple()

    with torch.no_grad():
        for p_dst, p_src in zip(mxfp8[0].parameters(), base[0].parameters()):
            p_dst.copy_(p_src)
        for p_dst, p_src in zip(mxfp8[2].parameters(), base[2].parameters()):
            p_dst.copy_(p_src)

    for m in base + mxfp8:
        m.train()

    return base, mxfp8


def _require_attention_backends(
    batch_size: int,
    seq_len: int,
    fp8_recipe,
) -> None:
    if get_cudnn_version() < (9, 2, 1):
        pytest.skip("cuDNN 9.2.1+ is required for FP8 fused attention.")

    config = ModelConfig(
        batch_size,
        seq_len,
        NUM_HEADS,
        HEAD_DIM_QK,
        head_dim_v=HEAD_DIM_V,
    )
    fp8_meta = {"recipe": fp8_recipe}
    fp8_backends, _, _ = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout="sbhd_sbhd_sbhd",
        fp8=True,
        fp8_meta=fp8_meta,
        is_training=True,
        deterministic=_DETERMINISTIC,
    )
    flash_attn_supported, fused_attn_supported_fp8, _ = fp8_backends
    if flash_attn_supported + fused_attn_supported_fp8 < 1:
        pytest.skip("No FP8 attention backend available for DSv3 MLA shape.")

    bf16_backends, _, _ = get_available_attention_backends(
        config,
        qkv_dtype=torch.bfloat16,
        qkv_layout="sbhd_sbhd_sbhd",
        is_training=True,
        deterministic=_DETERMINISTIC,
    )
    if sum(bf16_backends) < 1:
        pytest.skip("No BF16 attention backend available for DSv3 MLA shape.")

    _attention_backends["backend_selection_requires_update"] = True


def _split_qkv(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKV [s, b, h*(2*dqk+dv)] -> Q/K [s,b,h,dqk], V [s,b,h,dv]."""
    s, b, _ = qkv.shape
    q = qkv[:, :, : NUM_HEADS * HEAD_DIM_QK].view(s, b, NUM_HEADS, HEAD_DIM_QK)
    k = qkv[:, :, NUM_HEADS * HEAD_DIM_QK : 2 * NUM_HEADS * HEAD_DIM_QK].view(
        s, b, NUM_HEADS, HEAD_DIM_QK
    )
    v = qkv[:, :, 2 * NUM_HEADS * HEAD_DIM_QK :].view(s, b, NUM_HEADS, HEAD_DIM_V)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _run_forward_bf16(modules: tuple, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qkv_linear, dpa, out_linear = modules
    qkv = qkv_linear(x)
    q, k, v = _split_qkv(qkv)
    q, k, v = apply_mla_rope(q, k, v)
    attn_out = dpa(q, k, v, qkv_format="sbhd")
    return qkv, out_linear(attn_out.view(x.shape[0], x.shape[1], HIDDEN_SIZE))


def _run_forward_mxfp8(
    modules: tuple,
    x: torch.Tensor,
    recipe,
    is_first_microbatch: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """is_first_microbatch=True caches quantized weights; False reuses cache; None re-quantizes."""
    qkv_linear, dpa, out_linear = modules

    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        qkv = qkv_linear(x, is_first_microbatch=is_first_microbatch)

    q, k, v = _split_qkv(qkv)
    q, k, v = apply_mla_rope(q, k, v)

    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        attn_out = dpa(q, k, v, qkv_format="sbhd")

    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = out_linear(
            attn_out.view(x.shape[0], x.shape[1], HIDDEN_SIZE),
            is_first_microbatch=is_first_microbatch,
        )

    return qkv, out


def _compute_errors(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.pow(2).mean().sqrt().item()


def _clear_training_step_grads(modules: tuple, x: torch.Tensor) -> None:
    x.grad = None
    for module in modules:
        for param in module.parameters():
            param.grad = None


def _benchmark_training_step(
    forward_fn,
    modules: tuple,
    x: torch.Tensor,
    *forward_args,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> tuple[float, float]:
    for _ in range(warmup):
        _clear_training_step_grads(modules, x)
        _, out = forward_fn(modules, x, *forward_args)
        out.sum().backward()
    torch.cuda.synchronize()

    fprop_ms = 0.0
    bprop_ms = 0.0
    for _ in range(iters):
        _clear_training_step_grads(modules, x)

        fprop_start = torch.cuda.Event(enable_timing=True)
        fprop_end = torch.cuda.Event(enable_timing=True)
        bprop_start = torch.cuda.Event(enable_timing=True)
        bprop_end = torch.cuda.Event(enable_timing=True)

        fprop_start.record()
        _, out = forward_fn(modules, x, *forward_args)
        fprop_end.record()

        bprop_start.record()
        out.sum().backward()
        bprop_end.record()

        torch.cuda.synchronize()
        fprop_ms += fprop_start.elapsed_time(fprop_end)
        bprop_ms += bprop_start.elapsed_time(bprop_end)

    return fprop_ms / iters, bprop_ms / iters


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [4096])
class TestLinearMXFP8Attention:

    def test_accuracy(self, batch_size: int, seq_len: int) -> None:
        """Validate MXFP8 against BF16 using loose tolerances."""
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        baseline_modules, mxfp8_modules = _build_modules()
        x = torch.randn(seq_len, batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

        qkv_bf16, out_bf16 = _run_forward_bf16(baseline_modules, x)
        qkv_mxfp8, out_mxfp8 = _run_forward_mxfp8(mxfp8_modules, x, fp8_recipe)

        assert not torch.isnan(qkv_mxfp8).any(), "MXFP8 QKV contains NaN"
        assert not torch.isinf(qkv_mxfp8).any(), "MXFP8 QKV contains Inf"
        assert qkv_mxfp8.float().abs().max() > 0, "MXFP8 QKV is all zeros"

        assert not torch.isnan(out_mxfp8).any(), "MXFP8 output contains NaN"
        assert not torch.isinf(out_mxfp8).any(), "MXFP8 output contains Inf"
        assert out_mxfp8.float().abs().max() > 0, "MXFP8 output is all zeros"

        max_abs_qkv, rms_qkv = _compute_errors(qkv_bf16, qkv_mxfp8)
        print(f"\n[QKV] b={batch_size} s={seq_len}: max_abs={max_abs_qkv:.6f}  rms={rms_qkv:.6f}")
        torch.testing.assert_close(
            qkv_mxfp8,
            qkv_bf16,
            atol=2.0,
            rtol=0.5,
            msg=f"QKV mismatch: max_abs={max_abs_qkv:.6f} rms={rms_qkv:.6f}",
        )

        max_abs_out, rms_out = _compute_errors(out_bf16, out_mxfp8)
        print(f"[OUT] b={batch_size} s={seq_len}: max_abs={max_abs_out:.6f}  rms={rms_out:.6f}")
        torch.testing.assert_close(
            out_mxfp8,
            out_bf16,
            atol=8.0,
            rtol=2.0,
            msg=f"Output mismatch: max_abs={max_abs_out:.6f} rms={rms_out:.6f}",
        )

    def test_backward(self, batch_size: int, seq_len: int) -> None:
        """Gradients must flow end-to-end without NaN/Inf."""
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        _, mxfp8_modules = _build_modules()

        x = torch.randn(
            seq_len,
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        _, out_mxfp8 = _run_forward_mxfp8(mxfp8_modules, x, fp8_recipe)
        out_mxfp8.sum().backward()

        assert x.grad is not None, "MXFP8 path: input grad is None"
        assert not torch.isnan(x.grad).any(), "MXFP8 path: input grad NaN"
        assert not torch.isinf(x.grad).any(), "MXFP8 path: input grad Inf"

        qkv_fp8, _, out_fp8 = mxfp8_modules
        for name, mod in [("qkv_linear", qkv_fp8), ("out_linear", out_fp8)]:
            for p in mod.parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"MXFP8 {name} param grad NaN"
                    assert not torch.isinf(p.grad).any(), f"MXFP8 {name} param grad Inf"

        dx_rms = x.grad.float().pow(2).mean().sqrt().item()
        print(f"\n[BPROP] b={batch_size} s={seq_len}: dx rms={dx_rms:.6f}")
        assert dx_rms > 0.0, "MXFP8 path: input grad is all zeros (no gradient flow)"

    def test_performance(self, batch_size: int, seq_len: int) -> None:
        """Benchmark MXFP8 against BF16.

        Weights are pre-cached via is_first_microbatch=True so pre-quantized
        weights are reused each iteration without per-iteration weight quantization.
        """
        fp8_recipe = MXFP8BlockScaling(fp8_dpa=True)
        _require_attention_backends(batch_size, seq_len, fp8_recipe)
        _set_seed()
        baseline_modules, mxfp8_modules = _build_modules()
        x = torch.randn(
            seq_len,
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )

        with torch.no_grad():
            _run_forward_mxfp8(mxfp8_modules, x, fp8_recipe, is_first_microbatch=True)

        mxfp8_fprop_ms, mxfp8_bprop_ms = _benchmark_training_step(
            _run_forward_mxfp8, mxfp8_modules, x, fp8_recipe, False
        )
        mxfp8_fprop_tok = (batch_size * seq_len) / (mxfp8_fprop_ms / 1000.0)
        mxfp8_bprop_tok = (batch_size * seq_len) / (mxfp8_bprop_ms / 1000.0)

        bf16_fprop_ms, bf16_bprop_ms = _benchmark_training_step(
            _run_forward_bf16, baseline_modules, x
        )
        bf16_fprop_tok = (batch_size * seq_len) / (bf16_fprop_ms / 1000.0)
        bf16_bprop_tok = (batch_size * seq_len) / (bf16_bprop_ms / 1000.0)
        fprop_speedup = bf16_fprop_ms / mxfp8_fprop_ms
        bprop_speedup = bf16_bprop_ms / mxfp8_bprop_ms
        print(
            f"\n[PERF] b={batch_size} s={seq_len}:"
            f"\n  BF16 fprop:  {bf16_fprop_ms:.3f} ms  ({bf16_fprop_tok:.0f} tok/s)"
            f"\n  BF16 bprop:  {bf16_bprop_ms:.3f} ms  ({bf16_bprop_tok:.0f} tok/s)"
            f"\n  MXFP8 fprop: {mxfp8_fprop_ms:.3f} ms  ({mxfp8_fprop_tok:.0f} tok/s)"
            f"\n  MXFP8 bprop: {mxfp8_bprop_ms:.3f} ms  ({mxfp8_bprop_tok:.0f} tok/s)"
            f"\n  Fprop speedup: {fprop_speedup:.2f}x"
            f"\n  Bprop speedup: {bprop_speedup:.2f}x"
        )

        assert fprop_speedup > 1.0, (
            "MXFP8 fprop should be faster than BF16: "
            f"got {mxfp8_fprop_ms:.3f} ms vs BF16 {bf16_fprop_ms:.3f} ms "
            f"(speedup={fprop_speedup:.2f}x)"
        )
        assert bprop_speedup > 1.0, (
            "MXFP8 bprop should be faster than BF16: "
            f"got {mxfp8_bprop_ms:.3f} ms vs BF16 {bf16_bprop_ms:.3f} ms "
            f"(speedup={bprop_speedup:.2f}x)"
        )
