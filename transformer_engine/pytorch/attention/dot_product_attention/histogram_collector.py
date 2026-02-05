"""
Histogram collection infrastructure for softmax analysis in attention mechanisms.

This module provides thread-safe histogram collection for forward and backward passes
through the softmax operation in UnfusedDotProductAttention. Controlled via environment
variables, it has zero overhead when disabled.

Environment variables:
- NVTE_COLLECT_SOFTMAX_HISTOGRAM: Set to "1" to enable collection
- NVTE_HISTOGRAM_BINS: Number of histogram bins (default: 50)
- NVTE_HISTOGRAM_OUTPUT_FREQ: Output frequency in steps (default: 100)
"""

import os
import threading
from typing import Dict, Optional, Tuple
import torch


class SoftmaxHistogramCollector:
    """
    Thread-safe collector for softmax forward and backward pass histograms.

    Accumulates per-layer histograms over time and provides table formatting
    for analysis output.
    """

    def __init__(self, num_bins: int = 50, output_freq: int = 100):
        """
        Initialize the histogram collector.

        Args:
            num_bins: Number of bins for histogram (default: 50)
            output_freq: Output frequency in steps (default: 100)
        """
        self.num_bins = num_bins
        self.output_freq = output_freq

        # Per-layer histogram storage
        # forward_histograms[layer_id] = tensor of shape (num_bins,)
        self.forward_histograms: Dict[int, torch.Tensor] = {}
        self.forward_bin_edges: Dict[int, torch.Tensor] = {}

        # backward_histograms[layer_id] = tensor of shape (num_bins,)
        self.backward_histograms: Dict[int, torch.Tensor] = {}
        self.backward_ranges: Dict[int, Tuple[float, float]] = {}

        # Thread safety
        self.lock = threading.Lock()

        # Step counter for periodic output
        self.step_count = 0

    def _compute_histogram_bf16_compatible(
        self, data: torch.Tensor, num_bins: int, min_val: float, max_val: float
    ) -> torch.Tensor:
        """
        Compute histogram using BFloat16-compatible operations.

        Uses torch.bucketize for bin assignment, which supports BFloat16,
        then counts elements per bin using torch.bincount.

        Args:
            data: Flattened input tensor (any dtype including BFloat16)
            num_bins: Number of histogram bins
            min_val: Minimum value for histogram range
            max_val: Maximum value for histogram range

        Returns:
            Histogram counts tensor of shape (num_bins,)
        """
        # Create bin edges on the same device as data
        # Use float32 for bin edges for precision, bucketize handles dtype conversion
        bin_edges = torch.linspace(
            min_val, max_val, num_bins + 1, device=data.device, dtype=torch.float32
        )

        # Clamp data to range to handle edge cases
        # These operations work with BFloat16
        clamped_data = data.clamp(min=min_val, max=max_val)

        # bucketize returns bin indices (0 to num_bins)
        # right=False means bins are [edge[i], edge[i+1])
        # We need to convert to float32 for bucketize comparison precision
        bin_indices = torch.bucketize(clamped_data.float(), bin_edges, right=False)

        # Adjust indices: bucketize can return num_bins for values == max_val
        # We want these in the last bin (index num_bins - 1)
        bin_indices = bin_indices.clamp(max=num_bins - 1)

        # For values at exactly min_val, bucketize with right=False returns 0,
        # but we want them in bin 0, so subtract 1 for indices > 0 where value < edge
        # Actually, bucketize returns the index where value would be inserted,
        # so for [0, 0.02, 0.04, ...], value 0.0 returns 0, value 0.01 returns 1
        # We need to subtract 1 for indices > 0 to get the correct bin
        bin_indices = (bin_indices - 1).clamp(min=0)

        # Count elements in each bin
        hist = torch.bincount(bin_indices, minlength=num_bins)

        return hist[:num_bins]  # Ensure exactly num_bins elements

    def collect_forward(self, layer_id: int, probs: torch.Tensor) -> None:
        """
        Collect forward pass histogram (softmax output probabilities).

        Args:
            layer_id: Layer identifier for per-layer tracking
            probs: Softmax output tensor (values in [0, 1])
        """
        with torch.no_grad():
            # Flatten tensor - works with BFloat16
            flat_probs = probs.flatten()

            # Use BFloat16-compatible histogram computation
            # Fixed range [0, 1] for softmax outputs
            hist = self._compute_histogram_bf16_compatible(
                flat_probs, self.num_bins, min_val=0.0, max_val=1.0
            )

            with self.lock:
                # Initialize storage for this layer if needed
                if layer_id not in self.forward_histograms:
                    self.forward_histograms[layer_id] = torch.zeros(
                        self.num_bins, dtype=torch.long
                    )
                    # Store bin edges for later display
                    self.forward_bin_edges[layer_id] = torch.linspace(
                        0.0, 1.0, self.num_bins + 1
                    )

                # Accumulate counts
                self.forward_histograms[layer_id] += hist.cpu().long()

                # Increment step counter and check for output
                self.step_count += 1
                if self.step_count % self.output_freq == 0:
                    self._output_table()

    def collect_backward(self, layer_id: int, grad: torch.Tensor) -> None:
        """
        Collect backward pass histogram (gradients through softmax).

        Args:
            layer_id: Layer identifier for per-layer tracking
            grad: Gradient tensor flowing through softmax
        """
        with torch.no_grad():
            # Flatten gradients - works with BFloat16
            flat_grad = grad.flatten()

            # Compute dynamic range (min/max work with BFloat16)
            min_val = flat_grad.min().item()
            max_val = flat_grad.max().item()

            # Avoid degenerate case where all gradients are the same
            if abs(max_val - min_val) < 1e-10:
                # Use a small range around the value
                center = (max_val + min_val) / 2
                min_val = center - 1e-5
                max_val = center + 1e-5

            # Use BFloat16-compatible histogram computation
            hist = self._compute_histogram_bf16_compatible(
                flat_grad, self.num_bins, min_val=min_val, max_val=max_val
            )

            with self.lock:
                # Initialize storage for this layer if needed
                if layer_id not in self.backward_histograms:
                    self.backward_histograms[layer_id] = torch.zeros(
                        self.num_bins, dtype=torch.long
                    )
                    self.backward_ranges[layer_id] = (min_val, max_val)
                else:
                    # Update range to encompass all observed values
                    old_min, old_max = self.backward_ranges[layer_id]
                    self.backward_ranges[layer_id] = (
                        min(old_min, min_val),
                        max(old_max, max_val)
                    )

                # Accumulate counts
                self.backward_histograms[layer_id] += hist.cpu().long()

    def _format_number(self, num: int) -> str:
        """Format large numbers with comma separators."""
        return f"{num:,}"

    def _output_table(self) -> None:
        """Print histogram table to stdout."""
        table = self.get_table()
        if table:
            print(table, flush=True)

    def get_table(self) -> str:
        """
        Generate formatted table string for all collected histograms.

        Returns:
            Formatted string containing histogram tables
        """
        with self.lock:
            if not self.forward_histograms and not self.backward_histograms:
                return ""

            lines = []
            lines.append("=" * 80)
            lines.append(f"SOFTMAX HISTOGRAM ANALYSIS (Step {self.step_count})")
            lines.append("=" * 80)

            # Forward pass histograms
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

                    # Show top 10 bins by count
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

            # Backward pass histograms
            if self.backward_histograms:
                lines.append("")
                lines.append("--- BACKWARD PASS (dSoftmax Gradient) ---")
                lines.append("")

                for layer_id in sorted(self.backward_histograms.keys()):
                    hist = self.backward_histograms[layer_id]
                    min_val, max_val = self.backward_ranges[layer_id]

                    total_samples = hist.sum().item()

                    # Compute bin edges for this range
                    edges = torch.linspace(min_val, max_val, self.num_bins + 1)

                    lines.append(f"Layer {layer_id}:")
                    lines.append(f"  Total samples: {self._format_number(total_samples)}")
                    lines.append(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")
                    lines.append(f"  {'Bin Start':<12} | {'Bin End':<12} | {'Count':<15} | {'Percentage':<10}")
                    lines.append("  " + "-" * 60)

                    # Show top 10 bins by count
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
        """Clear all collected histogram data."""
        with self.lock:
            self.forward_histograms.clear()
            self.forward_bin_edges.clear()
            self.backward_histograms.clear()
            self.backward_ranges.clear()
            self.step_count = 0


# Global singleton instance
_collector: Optional[SoftmaxHistogramCollector] = None
_collector_lock = threading.Lock()


def get_histogram_collector() -> Optional[SoftmaxHistogramCollector]:
    """
    Get the global histogram collector singleton.

    Returns None if histogram collection is disabled via environment variable.
    Lazy initialization on first call.

    Returns:
        SoftmaxHistogramCollector instance if enabled, None otherwise
    """
    global _collector

    # ==========================================================================
    # DEBUG MODIFICATION: Always enable histogram collection
    # Original code checked: if not int(os.getenv("NVTE_COLLECT_SOFTMAX_HISTOGRAM", "0")): return None
    # To restore original behavior, uncomment the following two lines:
    # if not int(os.getenv("NVTE_COLLECT_SOFTMAX_HISTOGRAM", "0")):
    #     return None
    # ==========================================================================

    # Lazy initialization with double-checked locking
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                num_bins = int(os.getenv("NVTE_HISTOGRAM_BINS", "50"))
                output_freq = int(os.getenv("NVTE_HISTOGRAM_OUTPUT_FREQ", "100"))
                _collector = SoftmaxHistogramCollector(
                    num_bins=num_bins,
                    output_freq=output_freq
                )

    return _collector
