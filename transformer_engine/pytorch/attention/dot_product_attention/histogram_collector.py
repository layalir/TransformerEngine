"""
Histogram collection for softmax analysis in attention mechanisms.

Environment variables:
- NVTE_HISTOGRAM_BINS: Number of histogram bins (default: 50)
- NVTE_HISTOGRAM_OUTPUT_FREQ: Output every N forward calls on THIS rank (default: 1000)
- NVTE_HISTOGRAM_SAMPLE_STRIDE: Sample every Nth element (default: 10000)
- NVTE_HISTOGRAM_LAYER_FREQ: Collect every Nth layer (default: 1)
- NVTE_HISTOGRAM_COLLECT_FORWARD: Enable forward collection (default: 1)
- NVTE_HISTOGRAM_COLLECT_BACKWARD: Enable backward collection (default: 1)
- NVTE_HISTOGRAM_DEBUG: Enable debug prints (default: 0)
- NVTE_HISTOGRAM_OUTPUT_RANK: Only this rank outputs tables (default: 0, set -1 for all ranks)
- NVTE_HISTOGRAM_RESET_AFTER_OUTPUT: Clear buffers after output to bound memory (default: 1)
- NVTE_HISTOGRAM_GATHER_TO_RANK0: Gather histograms from all ranks to rank 0 (default: 1)
- NVTE_HISTOGRAM_MAX_LAYERS: Maximum number of layers to track (default: 256)
"""

import os
import threading
from typing import Dict, List, Optional, Tuple
import torch


def _get_rank() -> int:
    """Get distributed rank if available, else -1."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return -1


def _get_world_size() -> int:
    """Get distributed world size if available, else 1."""
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
    except Exception:
        pass
    return 1


def _is_distributed() -> bool:
    """Check if distributed is initialized."""
    try:
        return torch.distributed.is_initialized()
    except Exception:
        return False


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

        # Only allow specific rank to output (default: rank 0, set -1 for all ranks)
        self.output_rank = int(os.getenv("NVTE_HISTOGRAM_OUTPUT_RANK", "0"))

        # Memory management: reset buffers after output to bound memory usage
        self.reset_after_output = os.getenv("NVTE_HISTOGRAM_RESET_AFTER_OUTPUT", "1") == "1"

        # Distributed gathering: collect histograms from all ranks to rank 0
        self.gather_to_rank0 = os.getenv("NVTE_HISTOGRAM_GATHER_TO_RANK0", "1") == "1"

        # Maximum layers to track (for distributed gather tensor sizing)
        self.max_layers = int(os.getenv("NVTE_HISTOGRAM_MAX_LAYERS", "256"))

        # Simple call counter (no step detection - just count forward calls)
        self._total_fwd_calls = 0
        self._total_bwd_calls = 0
        self._last_output_at = 0

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

        if not self.collect_forward_enabled:
            return

        if not self._should_collect_now():
            return

        if not self._should_collect_layer(layer_id):
            return

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

    def _can_output(self) -> bool:
        """Check if this rank is allowed to output."""
        if self.output_rank == -1:
            return True
        current_rank = _get_rank()
        if current_rank == -1:
            return True
        return current_rank == self.output_rank

    def _gather_histograms_to_rank0(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Gather histograms from all ranks to rank 0.

        Returns aggregated forward and backward histograms.
        Only rank 0 gets meaningful data; other ranks get empty dicts.
        """
        if not _is_distributed():
            return self.forward_histograms.copy(), self.backward_histograms.copy()

        world_size = _get_world_size()
        rank = _get_rank()

        # Pack local forward histograms into a tensor
        # Format: [layer_id, hist_bin_0, hist_bin_1, ..., hist_bin_N-1] for each layer
        # Use -1 as layer_id to indicate empty slots
        local_fwd_data = torch.full(
            (self.max_layers, 1 + self.num_bins), -1, dtype=torch.long
        )
        with self.lock:
            for i, (layer_id, hist) in enumerate(self.forward_histograms.items()):
                if i >= self.max_layers:
                    break
                local_fwd_data[i, 0] = layer_id
                local_fwd_data[i, 1:] = hist

        # Pack local backward histograms similarly
        # Format: [layer_id, hist_bin_0, ..., hist_bin_N-1, min_val_bits, max_val_bits]
        local_bwd_data = torch.full(
            (self.max_layers, 1 + self.num_bins + 2), -1, dtype=torch.long
        )
        with self.lock:
            for i, (layer_id, hist) in enumerate(self.backward_histograms.items()):
                if i >= self.max_layers:
                    break
                local_bwd_data[i, 0] = layer_id
                local_bwd_data[i, 1:1+self.num_bins] = hist
                if layer_id in self.backward_ranges:
                    min_val, max_val = self.backward_ranges[layer_id]
                    # Store floats as int bits
                    local_bwd_data[i, -2] = torch.tensor(min_val).view(torch.long).item()
                    local_bwd_data[i, -1] = torch.tensor(max_val).view(torch.long).item()

        # Gather all data to rank 0
        if rank == 0:
            gathered_fwd = [torch.zeros_like(local_fwd_data) for _ in range(world_size)]
            gathered_bwd = [torch.zeros_like(local_bwd_data) for _ in range(world_size)]
        else:
            gathered_fwd = None
            gathered_bwd = None

        torch.distributed.gather(local_fwd_data, gathered_fwd, dst=0)
        torch.distributed.gather(local_bwd_data, gathered_bwd, dst=0)

        # Aggregate on rank 0
        aggregated_fwd: Dict[int, torch.Tensor] = {}
        aggregated_bwd: Dict[int, torch.Tensor] = {}
        aggregated_bwd_ranges: Dict[int, Tuple[float, float]] = {}

        if rank == 0 and gathered_fwd is not None and gathered_bwd is not None:
            # Aggregate forward histograms
            for rank_data in gathered_fwd:
                for row in rank_data:
                    layer_id = row[0].item()
                    if layer_id < 0:
                        continue
                    hist = row[1:]
                    if layer_id not in aggregated_fwd:
                        aggregated_fwd[layer_id] = torch.zeros(self.num_bins, dtype=torch.long)
                    aggregated_fwd[layer_id] += hist

            # Aggregate backward histograms
            for rank_data in gathered_bwd:
                for row in rank_data:
                    layer_id = row[0].item()
                    if layer_id < 0:
                        continue
                    hist = row[1:1+self.num_bins]
                    min_bits = row[-2].item()
                    max_bits = row[-1].item()
                    min_val = torch.tensor(min_bits, dtype=torch.long).view(torch.float64).item()
                    max_val = torch.tensor(max_bits, dtype=torch.long).view(torch.float64).item()

                    if layer_id not in aggregated_bwd:
                        aggregated_bwd[layer_id] = torch.zeros(self.num_bins, dtype=torch.long)
                        aggregated_bwd_ranges[layer_id] = (min_val, max_val)
                    else:
                        old_min, old_max = aggregated_bwd_ranges[layer_id]
                        aggregated_bwd_ranges[layer_id] = (
                            min(old_min, min_val),
                            max(old_max, max_val)
                        )
                    aggregated_bwd[layer_id] += hist

            # Store ranges for table generation
            self._aggregated_bwd_ranges = aggregated_bwd_ranges

        return aggregated_fwd, aggregated_bwd

    def _do_output(self) -> None:
        """Output histogram table and optionally reset buffers."""
        current_rank = _get_rank()

        if self.gather_to_rank0 and _is_distributed():
            # Gather from all ranks, only rank 0 prints
            fwd_histograms, bwd_histograms = self._gather_histograms_to_rank0()

            if current_rank == 0:
                table = self._get_table_from_data(fwd_histograms, bwd_histograms)
                if table:
                    print(table, flush=True)
        else:
            # Local-only mode: each rank handles its own data
            if self._can_output():
                table = self.get_table()
                if table:
                    print(table, flush=True)

        # Reset buffers to bound memory usage (default behavior)
        if self.reset_after_output:
            self._reset_histograms()
        self._last_output_at = self._total_fwd_calls

    def _format_number(self, num: int) -> str:
        return f"{num:,}"

    def _reset_histograms(self) -> None:
        """Clear all histogram buffers and release memory."""
        with self.lock:
            for tensor in self.forward_histograms.values():
                del tensor
            for tensor in self.forward_bin_edges.values():
                del tensor
            for tensor in self.backward_histograms.values():
                del tensor
            self.forward_histograms.clear()
            self.forward_bin_edges.clear()
            self.backward_histograms.clear()
            self.backward_ranges.clear()

    def _get_table_from_data(
        self,
        forward_histograms: Dict[int, torch.Tensor],
        backward_histograms: Dict[int, torch.Tensor],
    ) -> str:
        """Generate table from provided histogram data (used after gathering)."""
        if not forward_histograms and not backward_histograms:
            return ""

        lines = []
        lines.append("=" * 80)
        lines.append(f"SOFTMAX HISTOGRAM [AGGREGATED FROM ALL RANKS]")
        lines.append(f"(fwd_calls={self._total_fwd_calls} bwd_calls={self._total_bwd_calls})")
        lines.append("=" * 80)

        if forward_histograms:
            lines.append("")
            lines.append("--- FORWARD PASS (Softmax Output) ---")
            lines.append("")

            edges = torch.linspace(0.0, 1.0, self.num_bins + 1)
            for layer_id in sorted(forward_histograms.keys()):
                hist = forward_histograms[layer_id]
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

        if backward_histograms:
            lines.append("")
            lines.append("--- BACKWARD PASS (dSoftmax Gradient) ---")
            lines.append("")

            bwd_ranges = getattr(self, '_aggregated_bwd_ranges', {})
            for layer_id in sorted(backward_histograms.keys()):
                hist = backward_histograms[layer_id]
                min_val, max_val = bwd_ranges.get(layer_id, (0.0, 1.0))
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

    def get_table(self) -> str:
        """Generate table from local histogram data."""
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
                    print(f"[HISTO] Initialized: rank={rank} bins={num_bins} freq={output_freq} "
                          f"stride={_collector.sample_stride} output_rank={_collector.output_rank} "
                          f"reset={_collector.reset_after_output} gather={_collector.gather_to_rank0}",
                          flush=True)
    return _collector
