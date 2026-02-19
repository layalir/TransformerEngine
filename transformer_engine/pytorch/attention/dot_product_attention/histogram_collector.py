"""
Histogram collection for softmax analysis in attention mechanisms.

Environment variables:
- NVTE_HISTOGRAM_BINS: Number of histogram bins (default: 50)
- NVTE_HISTOGRAM_DISPLAY_BINS: Number of bins to display in table (default: 10, shows top N bins by count)
- NVTE_HISTOGRAM_OUTPUT_FREQ: Output every N forward calls on THIS rank (default: 1000)
- NVTE_HISTOGRAM_SAMPLE_STRIDE: Sample every Nth element (default: 10000), or if NVTE_HISTOGRAM_USE_MAXPOOL=1, this is the pool size
- NVTE_HISTOGRAM_USE_MAXPOOL: Use max pooling instead of strided sampling (default: 0, adds ~1-2% overhead)
- NVTE_HISTOGRAM_COLLECT_FORWARD: Enable forward collection (default: 1)
- NVTE_HISTOGRAM_COLLECT_BACKWARD: Enable backward collection (default: 0)
- NVTE_HISTOGRAM_DEBUG: Enable debug prints (default: 0)
- NVTE_HISTOGRAM_OUTPUT_RANK: Which rank(s) print tables (default: -1 = all ranks, or comma-separated list like "0,64,128,192")
- NVTE_HISTOGRAM_RESET_AFTER_OUTPUT: Clear buffers after output to bound memory (default: 1)

Features:
- Recompute detection: Automatically skips duplicate collections when activation
  checkpointing/recompute causes forward pass to run twice (within 100ms threshold)
- This prevents double-counting histograms when MoE/MLP recompute is enabled

Note: Only layers using UnfusedDotProductAttention will have histograms collected.
Use NVTE_FORCE_UNFUSED_LAYERS to control which layers are profiled.
"""

import os
import threading
import time
from typing import Dict, Optional, Set, Tuple
import torch


def _get_rank() -> int:
    """Get distributed rank if available, else 0."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


class SoftmaxHistogramCollector:
    """Thread-safe collector for softmax forward and backward pass histograms."""

    def __init__(self, num_bins: int = 50, output_freq: int = 1000):
        self.num_bins = num_bins
        self.output_freq = output_freq
        self.display_bins = int(os.getenv("NVTE_HISTOGRAM_DISPLAY_BINS", "10"))

        # Histogram storage (CPU tensors for memory efficiency)
        self.forward_histograms: Dict[int, torch.Tensor] = {}
        self.forward_bin_edges: Dict[int, torch.Tensor] = {}
        self.backward_histograms: Dict[int, torch.Tensor] = {}
        self.backward_ranges: Dict[int, Tuple[float, float]] = {}

        self.lock = threading.Lock()

        # Performance tuning
        self.sample_stride = int(os.getenv("NVTE_HISTOGRAM_SAMPLE_STRIDE", "10000"))
        self.use_maxpool = os.getenv("NVTE_HISTOGRAM_USE_MAXPOOL", "0") == "1"
        self.collect_forward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_FORWARD", "1") == "1"
        self.collect_backward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_BACKWARD", "0") == "1"
        self.debug = os.getenv("NVTE_HISTOGRAM_DEBUG", "0") == "1"

        # Output control: -1 = all ranks print, comma-separated list = specific ranks
        output_rank_str = os.getenv("NVTE_HISTOGRAM_OUTPUT_RANK", "-1")
        if output_rank_str == "-1":
            self.output_ranks = None  # None means all ranks
        else:
            # Parse comma-separated list: "0,64,128,192" or single "0"
            self.output_ranks = set(int(r.strip()) for r in output_rank_str.split(","))

        # Memory management: reset buffers after output to bound memory usage
        self.reset_after_output = os.getenv("NVTE_HISTOGRAM_RESET_AFTER_OUTPUT", "1") == "1"

        # Call counter
        self._total_fwd_calls = 0
        self._total_bwd_calls = 0
        self._last_output_at = 0

        # Track which layers we've logged (for debug, only log once per layer)
        self._logged_fwd_layers: Set[int] = set()
        self._logged_bwd_layers: Set[int] = set()

        # Recompute detection: track last collection time per layer
        # If called twice within threshold, skip second call (it's recompute)
        self._last_collection_time: Dict[int, float] = {}
        self._recompute_threshold_ms = 100.0  # 100ms threshold for detecting recompute

    def _can_output(self) -> bool:
        """Check if this rank is allowed to output."""
        if self.output_ranks is None:
            return True  # All ranks can output
        return _get_rank() in self.output_ranks

    def _should_collect_now(self) -> bool:
        """Check if we should collect based on total forward calls."""
        return (self._total_fwd_calls % self.output_freq) == 0

    def collect_forward(self, layer_id: int, probs: torch.Tensor) -> None:
        """Collect forward pass histogram (softmax output probabilities)."""
        self._total_fwd_calls += 1

        if not self.collect_forward_enabled:
            return

        # Recompute detection: Skip if we collected this layer very recently
        # This prevents double-counting when activation checkpointing recomputes forward
        current_time = time.time()
        last_time = self._last_collection_time.get(layer_id, 0)
        time_since_last_ms = (current_time - last_time) * 1000.0

        if time_since_last_ms < self._recompute_threshold_ms and last_time > 0:
            # Skip this collection - it's likely a recomputed forward pass
            if self.debug:
                rank = _get_rank()
                print(f"[HISTO r{rank}] Skipping recompute for layer {layer_id} "
                      f"(called again after {time_since_last_ms:.1f}ms)", flush=True)
            return

        # Update last collection time for this layer
        self._last_collection_time[layer_id] = current_time

        # Debug: log first time we collect each layer
        if self.debug and layer_id not in self._logged_fwd_layers:
            self._logged_fwd_layers.add(layer_id)
            rank = _get_rank()
            print(f"[HISTO r{rank}] Collecting forward for layer {layer_id}", flush=True)

        with torch.no_grad():
            # Sample and compute histogram on GPU, then move to CPU
            if self.use_maxpool:
                # Max pooling: find max of each pool_size chunk
                flat = probs.flatten().float()
                n_elements = flat.shape[0]
                pool_size = self.sample_stride  # Reuse stride variable as pool size
                n_pools = n_elements // pool_size

                if n_pools > 0:
                    # Truncate to multiple of pool_size and reshape
                    flat_truncated = flat[:n_pools * pool_size]
                    pooled = flat_truncated.view(n_pools, pool_size)
                    # Max along pool dimension
                    sampled_values = pooled.max(dim=1).values
                else:
                    # Fallback if tensor too small
                    sampled_values = flat
                del flat
            else:
                # Strided sampling (original approach)
                sampled_values = probs.flatten()[::self.sample_stride].float()

            hist = torch.histc(sampled_values, bins=self.num_bins, min=0.0, max=1.0)
            del sampled_values

            with self.lock:
                if layer_id not in self.forward_histograms:
                    self.forward_histograms[layer_id] = torch.zeros(
                        self.num_bins, dtype=torch.long
                    )
                    self.forward_bin_edges[layer_id] = torch.linspace(
                        0.0, 1.0, self.num_bins + 1
                    )
                self.forward_histograms[layer_id] += hist.cpu().long()

        self._do_output()

    def collect_backward(self, layer_id: int, grad: torch.Tensor) -> None:
        """Collect backward pass histogram (gradients through softmax)."""
        self._total_bwd_calls += 1

        if not self.collect_backward_enabled:
            return

        # Debug: log first time we collect each layer
        if self.debug and layer_id not in self._logged_bwd_layers:
            self._logged_bwd_layers.add(layer_id)
            rank = _get_rank()
            print(f"[HISTO r{rank}] Collecting backward for layer {layer_id}", flush=True)

        with torch.no_grad():
            flat_grad = grad.flatten()[::self.sample_stride].float()
            min_val = flat_grad.min().item()
            max_val = flat_grad.max().item()

            if abs(max_val - min_val) < 1e-10:
                center = (max_val + min_val) / 2
                min_val = center - 1e-5
                max_val = center + 1e-5

            hist = torch.histc(flat_grad, bins=self.num_bins, min=min_val, max=max_val)
            del flat_grad

            with self.lock:
                if layer_id not in self.backward_histograms:
                    self.backward_histograms[layer_id] = torch.zeros(
                        self.num_bins, dtype=torch.long
                    )
                    self.backward_ranges[layer_id] = (min_val, max_val)
                else:
                    old_min, old_max = self.backward_ranges[layer_id]
                    self.backward_ranges[layer_id] = (
                        min(old_min, min_val),
                        max(old_max, max_val)
                    )
                self.backward_histograms[layer_id] += hist.cpu().long()

    def _do_output(self) -> None:
        """Output histogram table and optionally reset buffers."""
        # Only output at the specified frequency
        if not self._should_collect_now():
            return

        if self._can_output():
            table = self.get_table()
            if table:
                print(table, flush=True)

        if self.reset_after_output:
            self._reset_histograms()
        self._last_output_at = self._total_fwd_calls

    def _format_number(self, num: int) -> str:
        return f"{num:,}"

    def _reset_histograms(self) -> None:
        """Clear all histogram buffers and release memory."""
        with self.lock:
            self.forward_histograms.clear()
            self.forward_bin_edges.clear()
            self.backward_histograms.clear()
            self.backward_ranges.clear()

    def get_table(self) -> str:
        """Generate table from local histogram data."""
        with self.lock:
            if not self.forward_histograms and not self.backward_histograms:
                return ""

            rank = _get_rank()
            lines = []
            lines.append("=" * 80)
            lines.append(f"SOFTMAX HISTOGRAM [RANK {rank}]")
            lines.append(f"(fwd_calls={self._total_fwd_calls} bwd_calls={self._total_bwd_calls})")
            lines.append("=" * 80)

            if self.forward_histograms:
                lines.append("")
                lines.append("--- FORWARD PASS (Softmax Output) ---")
                lines.append("")

                for layer_id in sorted(self.forward_histograms.keys()):
                    hist = self.forward_histograms[layer_id]
                    edges = self.forward_bin_edges[layer_id]
                    total_samples = hist.sum().item()

                    lines.append(f"Layer {layer_id}:")
                    lines.append(f"  Total samples: {self._format_number(total_samples)}")
                    lines.append(f"  Bin Range [0.000, 1.000]")
                    lines.append(f"  {'Bin Start':<12} | {'Bin End':<12} | {'Count':<15} | {'Percentage':<10}")
                    lines.append("  " + "-" * 60)

                    top_indices = torch.topk(hist, min(self.display_bins, self.num_bins)).indices
                    for idx in top_indices:
                        idx = idx.item()
                        bin_start = edges[idx].item()
                        bin_end = edges[idx + 1].item()
                        count = hist[idx].item()
                        percentage = 100.0 * count / total_samples if total_samples > 0 else 0.0
                        lines.append(
                            f"  {bin_start:<12.4f} | {bin_end:<12.4f} | "
                            f"{self._format_number(count):<15} | {percentage:>6.2f}%"
                        )
                    lines.append("")

            if self.backward_histograms:
                lines.append("")
                lines.append("--- BACKWARD PASS (dSoftmax Gradient) ---")
                lines.append("")

                for layer_id in sorted(self.backward_histograms.keys()):
                    hist = self.backward_histograms[layer_id]
                    min_val, max_val = self.backward_ranges[layer_id]
                    total_samples = hist.sum().item()
                    edges = torch.linspace(min_val, max_val, self.num_bins + 1)

                    lines.append(f"Layer {layer_id}:")
                    lines.append(f"  Total samples: {self._format_number(total_samples)}")
                    lines.append(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")
                    lines.append(f"  {'Bin Start':<12} | {'Bin End':<12} | {'Count':<15} | {'Percentage':<10}")
                    lines.append("  " + "-" * 60)

                    top_indices = torch.topk(hist, min(self.display_bins, self.num_bins)).indices
                    for idx in top_indices:
                        idx = idx.item()
                        bin_start = edges[idx].item()
                        bin_end = edges[idx + 1].item()
                        count = hist[idx].item()
                        percentage = 100.0 * count / total_samples if total_samples > 0 else 0.0
                        lines.append(
                            f"  {bin_start:<12.6f} | {bin_end:<12.6f} | "
                            f"{self._format_number(count):<15} | {percentage:>6.2f}%"
                        )
                    lines.append("")

            lines.append("=" * 80)
            return "\n".join(lines)

    def reset(self) -> None:
        self._reset_histograms()
        self._total_fwd_calls = 0
        self._total_bwd_calls = 0
        self._last_output_at = 0
        self._logged_fwd_layers.clear()
        self._logged_bwd_layers.clear()
        self._last_collection_time.clear()


# Global singleton
_collector: Optional[SoftmaxHistogramCollector] = None
_collector_lock = threading.Lock()


def get_histogram_collector() -> Optional[SoftmaxHistogramCollector]:
    """Get the global histogram collector singleton."""
    global _collector

    if _collector is None:
        with _collector_lock:
            if _collector is None:
                num_bins = int(os.getenv("NVTE_HISTOGRAM_BINS", "50"))
                output_freq = int(os.getenv("NVTE_HISTOGRAM_OUTPUT_FREQ", "1000"))
                _collector = SoftmaxHistogramCollector(
                    num_bins=num_bins,
                    output_freq=output_freq
                )
                if _collector.debug:
                    rank = _get_rank()
                    output_ranks_str = "all" if _collector.output_ranks is None else str(sorted(_collector.output_ranks))
                    sampling_mode = f"maxpool={_collector.sample_stride}" if _collector.use_maxpool else f"stride={_collector.sample_stride}"
                    print(f"[HISTO r{rank}] Initialized: bins={num_bins} freq={output_freq} "
                          f"{sampling_mode} output_ranks={output_ranks_str}",
                          flush=True)
    return _collector
