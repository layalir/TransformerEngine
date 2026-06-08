# Fused MLA RoPE + MXFP8 Quantize Handover

Branch: `layalir/fuse_fprop_attn_rope_quantize`

## Current State

This branch prototypes a fused fprop path for MLA RoPE plus MXFP8 quantization in unit-test utility code:

- `tests/pytorch/attention/mla_rope_utils.py`
  - Adds `apply_mla_rope_mxfp8_quantize`.
  - Adds prototype Triton kernels for fused Q RoPE+quantize, fused K RoPE+quantize, and V quantize.
  - Current Q/K fused launch uses `block_h = 8`.
- `tests/pytorch/attention/test_mla_rope_mxfp8_fusion.py`
  - Adds correctness coverage against the existing RoPE then MXFP8 quantize reference.
  - Adds a perf-gated DSv3-sized timing test under `NVTE_RUN_MLA_ROPE_MXFP8_FUSION_PERF=1`.

Latest dlcluster validation:

- Correctness job `1180665`: `1 passed, 1 skipped`.
- Nsight job `1180994`: baseline `0.528886 ms`, fused `0.464398 ms`, speedup `1.139x`.
- Perf pytest job `1181113`: reference `0.475 ms`, fused `0.465 ms`, speedup `1.02x`, `2 passed`.

The useful profile artifacts from the last run were copied locally to:

- `C:\tmp\te_mla_rope_mxfp8_fusion_1180994.sqlite`

## What HBM Floor Means

The HBM floor is the theoretical minimum time required to move the estimated bytes through HBM if the GPU sustained peak HBM bandwidth for this exact memory traffic.

For this estimate:

- Shape: `S=4096, B=1, H=128, Q/K=192, V=128`.
- Peak bandwidth assumption: `8 TB/s` per B200 GPU, derived from NVIDIA DGX B200's `64 TB/s` aggregate across 8 GPUs.
- Formula: `HBM floor ms = traffic GB / 8 TB/s`.

This is a lower bound, not an expected runtime. It ignores launch overhead, instruction overhead, FP8 conversion, E8M0 scale computation, reductions for amax, address-generation cost, cache behavior, non-coalesced accesses, redundant reads, and imperfect bandwidth utilization.

If silicon time is close to the HBM floor, memory movement is already efficient and little speedup remains. If silicon time is much larger than the HBM floor, either the kernel is not bandwidth efficient or it is limited by compute/control overhead rather than pure HBM traffic.

## Baseline Memory-Traffic Estimate

Silicon time is from Nsight job `1180994`, kernel totals divided by 5 profiled iterations. Traffic uses decimal GB.

| Path | Main Reads | Writes | Estimated Traffic | HBM Floor @ 8 TB/s | Silicon Time |
| --- | --- | --- | ---: | ---: | ---: |
| Q RoPE | BF16 Q RoPE dims, FP32 cos/sin | BF16 Q | 0.168 GB | 0.021 ms | 0.032 ms |
| KV RoPE | BF16 KV, BF16 K position embedding, FP32 cos/sin | BF16 K, BF16 V | 0.646 GB | 0.081 ms | 0.092 ms |
| Q+K MXFP8 quantize | BF16 Q/K | FP8 Q/K, uint8 E8M0 scales | 0.612 GB | 0.077 ms | 0.095 ms |
| V MXFP8 quantize | BF16 V | FP8 V, uint8 E8M0 scales | 0.203 GB | 0.025 ms | 0.035 ms |
| Core total | BF16 + FP32 table reads | BF16/FP8/scales | 1.629 GB | 0.204 ms | 0.254 ms |
| Core total plus scale post-processing, rough | BF16 + FP32 table reads + uint8 scales | BF16/FP8/scales | 1.667 GB | 0.208 ms | 0.281 ms |

Baseline is already fairly close to the bandwidth floor. The estimate suggests roughly `0.05-0.08 ms` remains in the current baseline kernel sequence, depending on whether scale post-processing is included.

## Fused Prototype Memory-Traffic Estimate

| Path | Main Reads | Writes | Estimated Traffic | HBM Floor @ 8 TB/s | Silicon Time |
| --- | --- | --- | ---: | ---: | ---: |
| Fused Q | BF16 Q, FP32 cos/sin; current kernel reloads RoPE pairs | FP8 Q, uint8 E8M0 scales | 0.407 GB | 0.051 ms | 0.182 ms |
| Fused K | BF16 KV noPE, BF16 K position embedding, FP32 cos/sin; current kernel duplicates K position embedding across head lanes | FP8 K, uint8 E8M0 scales | 0.407 GB | 0.051 ms | 0.143 ms |
| Fused V | BF16 V slice from KV | FP8 V, uint8 E8M0 scales | 0.203 GB | 0.025 ms | 0.128 ms |
| Core total | BF16 + FP32 table reads | FP8/scales | 1.017 GB | 0.127 ms | 0.453 ms |
| Core total plus scale post-processing, rough | BF16 + FP32 table reads + uint8 scales | FP8/scales | 1.038 GB | 0.130 ms | 0.462 ms |

The fused path reduces estimated traffic by about `38%` versus baseline: roughly `1.67 GB` to `1.04 GB`. However, the current Triton prototype uses the memory system much less efficiently than the existing TE kernels.

## Why Fused Bandwidth Looks Worse

The worse effective bandwidth is not only caused by duplicated or extra data loads.

Duplicate/extra traffic explains part of it:

- Fused Q currently reloads RoPE input pairs while producing the left and right rotated halves.
- Fused K currently duplicates `k_pos_emb` loads across the head lanes in the `BLOCK_H=8` program.

But the bigger signal is fused V:

- Fused V has the same logical memory traffic as the baseline TE V quantize path.
- Baseline TE V quantize takes about `0.035 ms`.
- Prototype fused V takes about `0.128 ms`.

That means the prototype kernels are not saturating memory bandwidth. Remaining issues are likely kernel implementation details: tiling, amax reductions, scalar scale stores, FP8 conversion, E8M0 scale math, address-generation overhead, and less mature scheduling than TE's production quantize kernels.

## TE Quantize Reuse Guidance

Use TE quantize kernels where doing so does not reintroduce the BF16 materialization that fusion is trying to remove.

For Q and K:

- Calling existing TE quantize after RoPE requires materialized BF16 Q/K RoPE outputs.
- That brings back the BF16 write from RoPE and BF16 read by quantize.
- Therefore, Q/K must remain fused or TE needs a new production fused entry point.

For V:

- V has no RoPE, so in principle TE V quantize is attractive because it is much faster than the prototype V kernel.
- In the current helper, `mxfp8_quantize_fast_path` reshapes the input with `.view(...)`.
- A direct V slice such as `kv[..., head_dim_nope:]` is not contiguous because the parent KV tensor has a larger head stride.
- Therefore, using the current TE V quantize helper would likely require `contiguous()` and a BF16 V copy, which would add another kernel and memory traffic.

Conclusion: do not switch to TE V quantize unless it can consume the strided V slice directly or TE exposes a quantize path that accepts this layout without materializing BF16 V. A reasonable hybrid design is Q/K fused plus TE-backed V only if that strided-input limitation is solved.

## Speedup Left On The Table

Using the rough scale-post-inclusive numbers:

- Current fused silicon time: about `0.462 ms`.
- HBM floor: about `0.130 ms`.
- Practical target at baseline-like bandwidth: `1.038 GB / 5.94 TB/s = about 0.175 ms`.

That means the current fused prototype has roughly:

- `2.6x` practical headroom versus baseline-like bandwidth.
- `3.6x` theoretical headroom versus the HBM floor.
- About `0.29-0.33 ms` absolute time left in this isolated RoPE+quantize stage.

## Recommended Next Steps

1. Optimize fused Q so each RoPE pair is loaded once and reused for both output halves.
2. Optimize fused K so `k_pos_emb` is loaded once per token/block and broadcast/reused, not duplicated across all head lanes.
3. Improve or replace fused V quantize. Reuse TE's V quantize only if it can operate on the strided V slice without a BF16 contiguous copy.
4. Reprofile after each change using the existing Nsight sbatch and compare against job `1180994`.
5. Once the prototype shows stable speedup, move the implementation out of test utility code toward a production TE entry point.

## Remote Test Notes

Remote staged workspace:

- `/home/lrashid/codex-te-mxfp8-pr3033-review/TransformerEngine`

Container:

- `gitlab-master.nvidia.com/dl/mlperf/optimized:deepseekv3_671b.pytorch.52025679`

Slurm:

- Account: `blackwell`
- Partition: `gb200nvl4`

Useful remote scripts:

- Correctness: `/home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_gb200.sbatch`
- Perf pytest: `/home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_perf_gb200.sbatch`
- Nsight: `/home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_nsys_gb200.sbatch`

Typical flow:

```bash
scp tests/pytorch/attention/mla_rope_utils.py tests/pytorch/attention/test_mla_rope_mxfp8_fusion.py dlcluster:/home/lrashid/codex-te-mxfp8-pr3033-review/TransformerEngine/tests/pytorch/attention/
ssh dlcluster sbatch /home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_gb200.sbatch
ssh dlcluster sbatch /home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_perf_gb200.sbatch
ssh dlcluster sbatch /home/lrashid/codex-te-mxfp8-pr3033-review/te_mla_rope_mxfp8_fusion_nsys_gb200.sbatch
```

