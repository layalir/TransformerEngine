"""
Histogram collection for softmax analysis in attention mechanisms.

Environment variables:
- NVTE_HISTOGRAM_BINS: Number of histogram bins (default: 50)
- NVTE_HISTOGRAM_OUTPUT_FREQ: Output every N forward calls (default: 1000)
- NVTE_HISTOGRAM_SAMPLE_STRIDE: Sample every Nth element (default: 10000)
- NVTE_HISTOGRAM_LAYER_FREQ: Collect every Nth layer (default: 1)
- NVTE_HISTOGRAM_COLLECT_FORWARD: Enable forward collection (default: 1)
- NVTE_HISTOGRAM_COLLECT_BACKWARD: Enable backward collection (default: 1)
- NVTE_HISTOGRAM_DEBUG: Enable debug prints (default: 0)

Note: OUTPUT_FREQ counts forward pass calls, not training steps.
With micro-batches and pipeline parallelism, one training step has many forward calls.
"""

import os
import threading
from typing import Dict, Optional, Tuple
import torch


def _get_rank() -> int:
    """Get distributed rank if available, else -1."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


class SoftmaxHistogramCollector:
    """Thread-safe collector for softmax forward and backward pass histograms."""

    def __init__(self, num_bins: int = 50, output_freq: int = 1000):
        self.num_bins = num_bins
        self.output_freq = output_freq

        # Histogram storage
        self.forward_histograms: Dict[int, torch.Tensor] = {}
        self.forward_bin_edges: Dict[int, torch.Tensor] = {}
        self.backward_histograms: Dict[int, torch.Tensor] = {}
        self.backward_ranges: Dict[int, Tuple[float, float]] = {}

        self.lock = threading.Lock()

        # Performance tuning
        self.sample_stride = int(os.getenv("NVTE_HISTOGRAM_SAMPLE_STRIDE", "10000"))
        self.layer_freq = int(os.getenv("NVTE_HISTOGRAM_LAYER_FREQ", "1"))
        self.collect_forward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_FORWARD", "1") == "1"
        self.collect_backward_enabled = os.getenv("NVTE_HISTOGRAM_COLLECT_BACKWARD", "1") == "1"
        self.debug = os.getenv("NVTE_HISTOGRAM_DEBUG", "0") == "1"

        # Simple call counter (no step detection - just count forward calls)
        self._total_fwd_calls = 0
        self._total_bwd_calls = 0
        self._last_output_at = 0

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            print(f"[HISTO] {msg}", flush=True)

    def _get_tensor_info(self, tensor: torch.Tensor) -> str:
        """Get tensor shape and device info for debug prints."""
        shape_str = list(tensor.shape)
        device = str(tensor.device)
        rank = _get_rank()
        numel = tensor.numel()
        sampled_size = numel // self.sample_stride
        mem_kb = (sampled_size * 4) / 1024
        return f"shape={shape_str} device={device} rank={rank} mem_est={mem_kb:.1f}KB"

    def _should_collect_layer(self, layer_id: int) -> bool:
        return (layer_id % self.layer_freq) == 0

    def _should_collect_now(self) -> bool:
        """Check if we should collect based on total forward calls."""
        return (self._total_fwd_calls % self.output_freq) == 0

    def _should_output_now(self) -> bool:
        """Check if we should output (after collecting enough data)."""
        return self._total_fwd_calls > 0 and self._total_fwd_calls > self._last_output_at

    def collect_forward(self, layer_id: int, probs: torch.Tensor) -> None:
        """Collect forward pass histogram (softmax output probabilities)."""
        self._total_fwd_calls += 1
        call_num = self._total_fwd_calls

        if self.debug:
            tensor_info = self._get_tensor_info(probs)
            self._debug_print(
                f"collect_forward layer={layer_id} call={call_num} {tensor_info}"
            )

        # Check if forward collection is enabled
        if not self.collect_forward_enabled:
            return

        # Only collect every output_freq calls
        if not self._should_collect_now():
            return

        # Skip layers based on layer_freq
        if not self._should_collect_layer(layer_id):
            return

        self._debug_print(f"COLLECTING forward layer={layer_id} call={call_num}")

        with torch.no_grad():
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

    def collect_backward(self, layer_id: int, grad: torch.Tensor) -> None:
        """Collect backward pass histogram (gradients through softmax)."""
        self._total_bwd_calls += 1
        call_num = self._total_bwd_calls

        if self.debug:
            tensor_info = self._get_tensor_info(grad)
            self._debug_print(
                f"collect_backward layer={layer_id} call={call_num} {tensor_info}"
            )

        # Check if backward collection is enabled
        if not self.collect_backward_enabled:
            # Check for output trigger
            if layer_id == 1 and self._should_output_now() and self.forward_histograms:
                self._do_output()
            return

        # Only collect when forward is also collecting
        if not self._should_collect_now():
            return

        # Skip layers based on layer_freq
        if not self._should_collect_layer(layer_id):
            return

        self._debug_print(f"COLLECTING backward layer={layer_id} call={call_num}")

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

        # Output after layer 1 backward (last layer in backward order)
        if layer_id == 1 and self._should_output_now():
            self._do_output()

    def _do_output(self) -> None:
        """Output histogram table and reset."""
        self._debug_print(f"OUTPUT at fwd_call={self._total_fwd_calls}")
        self._output_table()
        self._reset_histograms()
        self._last_output_at = self._total_fwd_calls

    def _format_number(self, num: int) -> str:
        return f"{num:,}"

    def _output_table(self) -> None:
        table = self.get_table()
        if table:
            print(table, flush=True)

    def _reset_histograms(self) -> None:
        with self.lock:
            self.forward_histograms.clear()
            self.forward_bin_edges.clear()
            self.backward_histograms.clear()
            self.backward_ranges.clear()

    def get_table(self) -> str:
        with self.lock:
            if not self.forward_histograms and not self.backward_histograms:
                return ""

            lines = []
            lines.append("=" * 80)
            lines.append(f"SOFTMAX HISTOGRAM (fwd_calls={self._total_fwd_calls} bwd_calls={self._total_bwd_calls})")
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


# Global singleton
_collector: Optional[SoftmaxHistogramCollector] = None
_collector_lock = threading.Lock()


def get_histogram_collector() -> Optional[SoftmaxHistogramCollector]:
    """Get the global histogram collector singleton."""
    global _collector

    # DEBUG: Always enable (remove env check)
    # To restore: if not int(os.getenv("NVTE_COLLECT_SOFTMAX_HISTOGRAM", "0")): return None

    if _collector is None:
        with _collector_lock:
            if _collector is None:
                num_bins = int(os.getenv("NVTE_HISTOGRAM_BINS", "50"))
                output_freq = int(os.getenv("NVTE_HISTOGRAM_OUTPUT_FREQ", "1000"))
                _collector = SoftmaxHistogramCollector(
                    num_bins=num_bins,
                    output_freq=output_freq
                )
    return _collector
