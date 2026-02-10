"""
Histogram collection for softmax analysis in attention mechanisms.

Environment variables:
- NVTE_HISTOGRAM_BINS: Number of histogram bins (default: 50)
- NVTE_HISTOGRAM_OUTPUT_FREQ: Output every N forward calls on THIS rank (default: 1000)
- NVTE_HISTOGRAM_SAMPLE_STRIDE: Sample every Nth element (default: 10000)
- NVTE_HISTOGRAM_LAYER_FREQ: Collect every Nth layer (default: 1)
- NVTE_HISTOGRAM_COLLECT_FORWARD: Enable forward collection (default: 1)
- NVTE_HISTOGRAM_COLLECT_BACKWARD: Enable backward collection (default: 0)
- NVTE_HISTOGRAM_DEBUG: Enable debug prints (default: 0)
- NVTE_HISTOGRAM_OUTPUT_RANK: Which rank prints tables (default: -1 = all ranks print)
- NVTE_HISTOGRAM_RESET_AFTER_OUTPUT: Clear buffers after output to bound memory (default: 1)
"""

import os
import threading
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

        # Histogram storage (CPU tensors for memory efficiency)
        self.forward_histograms: Dict[int, torch.Tensor] = {}
        self.forward_bin_edges: Dict[int, torch.Tensor] = {}
        self.backward_histograms: Dict[int, torch.Tensor] = {}
        self.backward_ranges: Dict[int, Tuple[float, float]] = {}

        self.lock = threading.Lock()

        # Performance tuning
        self.sample_stride = int(os.getenv("NVTE_HISTOGRAM_SAMPLE_STRIDE", "10000"))
        self.layer_freq = int(os.getenv("NVTE_HISTOGRAM_LAYER_FREQ", "1"))
        self.collect_forward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_FORWARD", "1") == "1"
        self.collect_backward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_BACKWARD", "0") == "1"
        self.debug = os.getenv("NVTE_HISTOGRAM_DEBUG", "0") == "1"

        # Output control: -1 = all ranks print, otherwise only specified rank prints
        self.output_rank = int(os.getenv("NVTE_HISTOGRAM_OUTPUT_RANK", "-1"))

        # Memory management: reset buffers after output to bound memory usage
        self.reset_after_output = os.getenv("NVTE_HISTOGRAM_RESET_AFTER_OUTPUT", "1") == "1"

        # Call counter
        self._total_fwd_calls = 0
        self._total_bwd_calls = 0
        self._last_output_at = 0

        # Track which layers we've logged (for debug, only log once per layer)
        self._logged_fwd_layers: Set[int] = set()
        self._logged_bwd_layers: Set[int] = set()

    def _can_output(self) -> bool:
        """Check if this rank is allowed to output."""
        if self.output_rank == -1:
            return True
        return _get_rank() == self.output_rank

    def _should_collect_layer(self, layer_id: int) -> bool:
        return (layer_id % self.layer_freq) == 0

    def _should_collect_now(self) -> bool:
        """Check if we should collect based on total forward calls."""
        return (self._total_fwd_calls % self.output_freq) == 0

    def collect_forward(self, layer_id: int, probs: torch.Tensor) -> None:
        """Collect forward pass histogram (softmax output probabilities)."""
        self._total_fwd_calls += 1

        if not self.collect_forward_enabled:
            return

        if not self._should_collect_now():
            return

        if not self._should_collect_layer(layer_id):
            return

        # Debug: log first time we collect each layer
        if self.debug and layer_id not in self._logged_fwd_layers:
            self._logged_fwd_layers.add(layer_id)
            rank = _get_rank()
            print(f"[HISTO r{rank}] Collecting forward for layer {layer_id}", flush=True)

        with torch.no_grad():
            # Sample and compute histogram on GPU, then move to CPU
            flat_probs = probs.flatten()[::self.sample_stride].float()
            hist = torch.histc(flat_probs, bins=self.num_bins, min=0.0, max=1.0)
            del flat_probs

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

        if not self._should_collect_now():
            return

        if not self._should_collect_layer(layer_id):
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

                    top_indices = torch.topk(hist, min(10, self.num_bins)).indices
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

                    top_indices = torch.topk(hist, min(10, self.num_bins)).indices
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
                    print(f"[HISTO r{rank}] Initialized: bins={num_bins} freq={output_freq} "
                          f"stride={_collector.sample_stride} output_rank={_collector.output_rank}",
                          flush=True)
    return _collector
