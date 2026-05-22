# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MLA RoPE for DSv3 671B - Triton forward and backward kernels.

Source: Megatron-LM megatron/core/fusions/fused_mla_yarn_rope_apply.py
Falls back to pure PyTorch when Triton is unavailable.

Note: DSv3 uses YaRN-scaled RoPE for long-context extrapolation. This test
intentionally uses plain RoPE (base=10000) because it only validates MXFP8
attention path wiring, tensor shapes, forward/backward flow, and relative BF16
vs MXFP8 behavior. Both reference and MXFP8 paths use the same RoPE tables.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

HEAD_DIM_ROPE = 64
HEAD_DIM_NOPE = 128
HEAD_DIM_V = 128
ROTARY_BASE = 10000


def build_rope_tables(
    seq_len: int,
    emb_dim: int = HEAD_DIM_ROPE,
    base: int = ROTARY_BASE,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, emb_dim, 2, dtype=torch.float32, device=device) / emb_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs).contiguous(), torch.sin(freqs).contiguous()


if HAVE_TRITON:

    # Not used for non-packed batches; kept for THD compatibility.
    @triton.jit
    def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
        token_idx = -1
        this_seq_len = 0
        seq_idx = 0
        last_cum_seqlen = tl.load(cu_seqlens) // cp_size
        while seq_idx < seq_num:
            cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1) // cp_size
            if token_idx == -1 and cur_cum_seqlen > pid_m:
                token_idx = pid_m - last_cum_seqlen
                this_seq_len = cur_cum_seqlen - last_cum_seqlen
            last_cum_seqlen = cur_cum_seqlen
            seq_idx += 1
        if cp_size > 1:
            if token_idx < this_seq_len // 2:
                token_idx = token_idx + cp_rank * this_seq_len // 2
            else:
                token_idx = (token_idx - this_seq_len // 2) + (
                    2 * cp_size - cp_rank - 1
                ) * this_seq_len // 2
        return token_idx

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "head_num"],
        restore_value=["Q"],
    )
    @triton.jit
    def rotary_fwd_q_kernel(
        Q,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
        mask = x_off < head_num * stride_x_nheads
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_1 = tl.load(Q + x_1_off, mask=mask)
        x_2 = tl.load(Q + x_2_off, mask=mask)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        tl.store(Q + x_left_off, x_left, mask=mask)
        tl.store(Q + x_right_off, x_right, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "head_num"],
        restore_value=["DO"],
    )
    @triton.jit
    def rotary_bwd_q_kernel(
        DO,
        COS,
        SIN,
        qk_head_dim,
        emb_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_q,
        stride_x_seq,
        stride_x_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
        mask = x_off < head_num * stride_x_nheads
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        x_left = tl.load(DO + x_left_off, mask=mask)
        x_right = tl.load(DO + x_right_off, mask=mask)
        x_1 = x_left * cos_left + x_right * sin_right
        x_2 = -x_left * sin_left + x_right * cos_right
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        tl.store(DO + x_1_off, x_1, mask=mask)
        tl.store(DO + x_2_off, x_2, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "k_dim", "v_dim", "head_num"],
    )
    @triton.jit
    def rotary_fwd_kv_kernel(
        KV,
        K_POS_EMB,
        O_KEY,
        O_VALUE,
        COS,
        SIN,
        emb_dim: tl.constexpr,
        k_dim: tl.constexpr,
        v_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_kv,
        stride_kv_seq,
        stride_kv_nheads,
        stride_emb_seq,
        stride_k_seq,
        stride_k_nheads,
        stride_v_seq,
        stride_v_nheads,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_kv is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        sin_right = tl.load(SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2))
        KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
        kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
        mask = kv_off < head_num * stride_kv_nheads
        k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
        v_in_off = kv_off + k_dim + tl.arange(0, v_dim)[None, :]
        k = tl.load(KV_ptr + k_in_off, mask=mask)
        v = tl.load(KV_ptr + v_in_off, mask=mask)
        K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
        V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads
        k_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads + tl.arange(0, k_dim)[None, :]
        v_out_off = tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads + tl.arange(0, v_dim)[None, :]
        tl.store(K_ptr + k_out_off, k, mask=mask)
        tl.store(V_ptr + v_out_off, v, mask=mask)
        EMB = K_POS_EMB + pid_m * stride_emb_seq
        x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
        x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_left_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + k_dim
            + tl.arange(0, emb_dim // 2)[None, :]
        )
        x_right_off = x_left_off + emb_dim // 2
        tl.store(K_ptr + x_left_off, x_left, mask=mask)
        tl.store(K_ptr + x_right_off, x_right, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "k_dim", "v_dim", "head_num"],
    )
    @triton.jit
    def rotary_bwd_kv_kernel(
        dK,
        dV,
        dKV,
        dEMB,
        COS,
        SIN,
        emb_dim: tl.constexpr,
        k_dim: tl.constexpr,
        v_dim: tl.constexpr,
        head_num: tl.constexpr,
        batch_size,
        seq_num,
        cu_seqlens_kv,
        stride_dk_seq,
        stride_dk_nheads,
        stride_dv_seq,
        stride_dv_nheads,
        stride_dkv_seq,
        stride_dkv_nheads,
        stride_demb_seq,
        cp_rank,
        cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_kv is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _get_thd_token_idx(cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size)
        dKV_ptr = dKV + pid_m * stride_dkv_seq + pid_head * BLOCK_H * stride_dkv_nheads
        dkv_off = tl.arange(0, BLOCK_H)[:, None] * stride_dkv_nheads
        mask = dkv_off < head_num * stride_dkv_nheads
        dk_out_off = dkv_off + tl.arange(0, k_dim)[None, :]
        dv_out_off = dkv_off + k_dim + tl.arange(0, v_dim)[None, :]
        dK_ptr = dK + pid_m * stride_dk_seq + pid_head * BLOCK_H * stride_dk_nheads
        dV_ptr = dV + pid_m * stride_dv_seq + pid_head * BLOCK_H * stride_dv_nheads
        dk_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + tl.arange(0, k_dim)[None, :]
        dv_in_off = tl.arange(0, BLOCK_H)[:, None] * stride_dv_nheads + tl.arange(0, v_dim)[None, :]
        dk = tl.load(dK_ptr + dk_in_off, mask=mask)
        dv = tl.load(dV_ptr + dv_in_off, mask=mask)
        tl.store(dKV_ptr + dk_out_off, dk, mask=mask)
        tl.store(dKV_ptr + dv_out_off, dv, mask=mask)
        if pid_head == 0:
            x_left_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
            x_right_accum = tl.zeros((BLOCK_H, emb_dim // 2), dtype=tl.float32)
            for i in tl.static_range(triton.cdiv(head_num, BLOCK_H)):
                dK_ptr_i = dK + pid_m * stride_dk_seq + i * BLOCK_H * stride_dk_nheads
                x_off = tl.arange(0, BLOCK_H)[:, None] * stride_dk_nheads + k_dim
                mask_i = x_off < head_num * stride_dk_nheads
                x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
                x_right_off = x_left_off + emb_dim // 2
                x_left_accum += tl.load(dK_ptr_i + x_left_off, mask=mask_i)
                x_right_accum += tl.load(dK_ptr_i + x_right_off, mask=mask_i)
            x_left_accum = tl.sum(x_left_accum, axis=0)
            x_right_accum = tl.sum(x_right_accum, axis=0)
            x_left_accum = x_left_accum.to(dEMB.dtype.element_ty)
            x_right_accum = x_right_accum.to(dEMB.dtype.element_ty)
            cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
            sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
            cos_right = tl.load(
                COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
            )
            sin_right = tl.load(
                SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
            )
            x_1 = x_left_accum * cos_left + x_right_accum * sin_right
            x_2 = -x_left_accum * sin_left + x_right_accum * cos_right
            dEMB_ptr = dEMB + pid_m * stride_demb_seq
            tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2, x_1)
            tl.store(dEMB_ptr + tl.arange(0, emb_dim // 2) * 2 + 1, x_2)

    class _MLARoPETriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, cos, sin, head_dim_nope, head_dim_rope, head_dim_v):
            s, b, nheads, _ = q.shape
            total = s * b

            # Q forward in-place. q is a fresh contiguous tensor so no autograd aliasing.
            q_3d = q.contiguous().view(total, nheads, q.shape[-1])
            grid_q = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_q_kernel[grid_q](
                q_3d,
                cos,
                sin,
                head_dim_nope,
                head_dim_rope,
                nheads,
                b,
                None,
                None,
                q_3d.stride(0),
                q_3d.stride(1),
                0,
                1,
            )
            q_out = q_3d.view(s, b, nheads, q.shape[-1])

            # KV forward: pack [k_nope | v], rotate k_pos_emb (head-0's rope portion).
            k_nope = k[..., :head_dim_nope].contiguous()
            k_pos_emb = k[:, :, 0, head_dim_nope:].contiguous().view(total, head_dim_rope)
            kv = torch.cat([k_nope, v], dim=-1).contiguous()
            kv_3d = kv.view(total, nheads, kv.shape[-1])
            o_key = kv_3d.new_empty(total, nheads, head_dim_nope + head_dim_rope)
            o_value = kv_3d.new_empty(total, nheads, head_dim_v)
            grid_kv = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_fwd_kv_kernel[grid_kv](
                kv_3d,
                k_pos_emb,
                o_key,
                o_value,
                cos,
                sin,
                head_dim_rope,
                head_dim_nope,
                head_dim_v,
                nheads,
                b,
                None,
                None,
                kv_3d.stride(0),
                kv_3d.stride(1),
                k_pos_emb.stride(0),
                o_key.stride(0),
                o_key.stride(1),
                o_value.stride(0),
                o_value.stride(1),
                0,
                1,
            )
            k_out = o_key.view(s, b, nheads, head_dim_nope + head_dim_rope)
            v_out = o_value.view(s, b, nheads, head_dim_v)

            ctx.save_for_backward(cos, sin)
            ctx.head_dim_nope = head_dim_nope
            ctx.head_dim_rope = head_dim_rope
            ctx.head_dim_v = head_dim_v
            ctx.nheads = nheads
            ctx.s = s
            ctx.b = b
            return q_out, k_out, v_out

        @staticmethod
        def backward(ctx, dq, dk_out, dv_out):
            cos, sin = ctx.saved_tensors
            s, b, nheads = ctx.s, ctx.b, ctx.nheads
            ndp, ndr, ndv = ctx.head_dim_nope, ctx.head_dim_rope, ctx.head_dim_v
            total = s * b

            # Q backward in-place on dq.
            dq_3d = dq.contiguous().view(total, nheads, dq.shape[-1])
            grid_q = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_q_kernel[grid_q](
                dq_3d,
                cos,
                sin,
                ndp,
                ndr,
                nheads,
                b,
                None,
                None,
                dq_3d.stride(0),
                dq_3d.stride(1),
                0,
                1,
            )
            dq_in = dq_3d.view(s, b, nheads, dq.shape[-1])

            # KV backward.
            dk_3d = dk_out.contiguous().view(total, nheads, ndp + ndr)
            dv_3d = dv_out.contiguous().view(total, nheads, ndv)
            d_kv = dk_3d.new_empty(total, nheads, ndp + ndv)
            d_emb = dk_3d.new_empty(total, 1, ndr)
            grid_kv = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
            rotary_bwd_kv_kernel[grid_kv](
                dk_3d,
                dv_3d,
                d_kv,
                d_emb,
                cos,
                sin,
                ndr,
                ndp,
                ndv,
                nheads,
                b,
                None,
                None,
                dk_3d.stride(0),
                dk_3d.stride(1),
                dv_3d.stride(0),
                dv_3d.stride(1),
                d_kv.stride(0),
                d_kv.stride(1),
                d_emb.stride(0),
                0,
                1,
            )
            # d_kv[:,: ,:ndp] -> k_nope grad (all heads)
            # d_emb[:,0,:]   -> k_rope grad for head 0 only (k_pos_emb = k[:,:,0,ndp:])
            d_kv_4d = d_kv.view(s, b, nheads, ndp + ndv)
            d_emb_4d = d_emb.view(s, b, 1, ndr)
            dk_in = torch.zeros(s, b, nheads, ndp + ndr, dtype=dq.dtype, device=dq.device)
            dk_in[:, :, :, :ndp] = d_kv_4d[:, :, :, :ndp]
            dk_in[:, :, 0, ndp:] = d_emb_4d[:, :, 0, :]
            dv_in = d_kv_4d[:, :, :, ndp:].contiguous()

            return dq_in, dk_in, dv_in, None, None, None, None, None


def apply_mla_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim_nope: int = HEAD_DIM_NOPE,
    head_dim_rope: int = HEAD_DIM_ROPE,
    head_dim_v: int = HEAD_DIM_V,
    base: int = ROTARY_BASE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = q.shape[0]
    cos_table, sin_table = build_rope_tables(s, emb_dim=head_dim_rope, base=base, device=q.device)

    if HAVE_TRITON:
        return _MLARoPETriton.apply(
            q,
            k,
            v,
            cos_table,
            sin_table,
            head_dim_nope,
            head_dim_rope,
            head_dim_v,
        )
    return _apply_pytorch(q, k, v, cos_table, sin_table, head_dim_nope, head_dim_rope)


def _rotate_interleaved_to_neox(
    x: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor
) -> torch.Tensor:
    cos_ = cos_table[:, None, None, :].to(x.dtype)
    sin_ = sin_table[:, None, None, :].to(x.dtype)
    half_dim = x.shape[-1] // 2
    x_1 = x[..., 0::2]
    x_2 = x[..., 1::2]
    x_left = x_1 * cos_[..., :half_dim] - x_2 * sin_[..., :half_dim]
    x_right = x_2 * cos_[..., half_dim:] + x_1 * sin_[..., half_dim:]
    return torch.cat((x_left, x_right), dim=-1)


def _apply_pytorch(q, k, v, cos_table, sin_table, head_dim_nope, head_dim_rope):
    def _apply_q(t: torch.Tensor) -> torch.Tensor:
        t_nope = t[..., :head_dim_nope]
        t_rope = t[..., head_dim_nope : head_dim_nope + head_dim_rope]
        t_rope = _rotate_interleaved_to_neox(t_rope, cos_table, sin_table)
        return torch.cat((t_nope, t_rope), dim=-1)

    k_nope = k[..., :head_dim_nope]
    k_pos_emb = k[:, :, :1, head_dim_nope : head_dim_nope + head_dim_rope]
    k_rope = _rotate_interleaved_to_neox(k_pos_emb, cos_table, sin_table).expand(
        -1, -1, k.shape[2], -1
    )

    return _apply_q(q), torch.cat((k_nope, k_rope), dim=-1), v
