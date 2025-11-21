# Copyright (c) 2024, Adaptive Pipeline Parallelism Research
# Network monitoring infrastructure for adaptive pipeline parallelism

import random
import time
import weakref
from collections import deque
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist

from .p2p_communication import P2PCommunicator
from .monitoring_stats_collector import get_monitoring_collector


class NetworkMonitor:
    """Lightweight network monitoring with async CUDA events.

    This class tracks bandwidth and latency of P2P communications using
    asynchronous CUDA events to minimize overhead. Measurements are collected
    lazily without forcing synchronization.

    Args:
        history_size: Maximum number of measurements to keep in history
    """

    def __init__(self, history_size: int = 100):
        self.pending_measurements = []
        self.bandwidth_history = deque(maxlen=history_size)
        self.latency_history = deque(maxlen=history_size)

        # Current estimates (exponential moving average)
        self.current_bandwidth_gbps: Optional[float] = None
        self.current_latency_ms: Optional[float] = None

        # Statistics
        self.total_bytes_sent = 0
        self.total_transfers = 0
        self.alpha = 0.3  # EMA smoothing factor

    def record_async(
        self,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        send_next_size: int,
        send_prev_size: int,
        recv_prev: bool,
        recv_next: bool
    ):
        """Record a measurement using CUDA events (non-blocking).

        Args:
            start_event: CUDA event recorded before communication
            end_event: CUDA event recorded after communication
            send_next_size: Bytes sent to next rank
            send_prev_size: Bytes sent to previous rank
            recv_prev: Whether receiving from previous rank
            recv_next: Whether receiving from next rank
        """
        self.pending_measurements.append({
            'start_event': start_event,
            'end_event': end_event,
            'send_next_size': send_next_size,
            'send_prev_size': send_prev_size,
            'recv_prev': recv_prev,
            'recv_next': recv_next,
            'timestamp': time.time()
        })

        # Periodically collect completed measurements
        if len(self.pending_measurements) > 10:
            self._collect_completed()

    def _collect_completed(self):
        """Collect completed measurements without blocking.

        This method checks which CUDA events have completed and processes
        their measurements. It does not block on incomplete events.
        """
        completed_indices = []

        for i, measurement in enumerate(self.pending_measurements):
            # Non-blocking check if event is complete
            if measurement['end_event'].query():
                # Compute elapsed time in milliseconds
                duration_ms = measurement['start_event'].elapsed_time(
                    measurement['end_event']
                )

                # Compute total data transferred
                total_bytes = measurement['send_next_size'] + measurement['send_prev_size']

                if duration_ms > 0 and total_bytes > 0:
                    # Compute bandwidth in GB/s
                    bandwidth_gbps = (total_bytes / (duration_ms / 1000)) / 1e9

                    # Update statistics
                    self._update_stats(bandwidth_gbps, duration_ms, total_bytes)

                completed_indices.append(i)

        # Remove completed measurements (reverse order to maintain indices)
        for i in reversed(completed_indices):
            self.pending_measurements.pop(i)

    def _update_stats(self, bandwidth_gbps: float, latency_ms: float, bytes_transferred: int):
        """Update running statistics.

        Args:
            bandwidth_gbps: Measured bandwidth in GB/s
            latency_ms: Measured latency in milliseconds
            bytes_transferred: Number of bytes transferred
        """
        self.bandwidth_history.append(bandwidth_gbps)
        self.latency_history.append(latency_ms)
        self.total_bytes_sent += bytes_transferred
        self.total_transfers += 1

        # Exponential moving average for current estimates
        if self.current_bandwidth_gbps is None:
            self.current_bandwidth_gbps = bandwidth_gbps
            self.current_latency_ms = latency_ms
        else:
            self.current_bandwidth_gbps = (
                self.alpha * bandwidth_gbps +
                (1 - self.alpha) * self.current_bandwidth_gbps
            )
            self.current_latency_ms = (
                self.alpha * latency_ms +
                (1 - self.alpha) * self.current_latency_ms
            )

    def get_bandwidth(self) -> Optional[float]:
        """Get current bandwidth estimate in GB/s.

        Returns:
            Current bandwidth estimate or None if no data
        """
        self._collect_completed()  # Collect any new data
        return self.current_bandwidth_gbps

    def get_latency(self) -> Optional[float]:
        """Get current latency estimate in milliseconds.

        Returns:
            Current latency estimate or None if no data
        """
        self._collect_completed()
        return self.current_latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dictionary with all statistics
        """
        self._collect_completed()

        return {
            'bandwidth_gbps': self.current_bandwidth_gbps,
            'latency_ms': self.current_latency_ms,
            'total_bytes_sent': self.total_bytes_sent,
            'total_transfers': self.total_transfers,
            'pending_measurements': len(self.pending_measurements),
            'avg_bandwidth_gbps': (
                sum(self.bandwidth_history) / len(self.bandwidth_history)
                if self.bandwidth_history else None
            ),
            'avg_latency_ms': (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history else None
            ),
            'timestamp': time.time()
        }


# Global registry for P2P communicators (for stats collection)
_global_communicators = []


class MonitoredP2PCommunicator(P2PCommunicator):
    """P2P Communicator with lightweight network monitoring.

    This class extends the standard P2PCommunicator to add network
    monitoring capabilities for adaptive pipeline parallelism. It uses
    CUDA events and probabilistic sampling to keep overhead minimal.

    Args:
        pp_group: Pipeline parallel process group
        config: Model parallel configuration
        enable_monitoring: Whether to enable monitoring (default: True)
        sample_rate: Fraction of operations to monitor (default: 0.1)
    """

    def __init__(
        self,
        pp_group: dist.ProcessGroup,
        config,
        enable_monitoring: bool = True,
        sample_rate: float = 0.1
    ):
        super().__init__(pp_group, config)

        self.enable_monitoring = enable_monitoring
        self.sample_rate = sample_rate
        self.monitor = NetworkMonitor() if enable_monitoring else None

        # Debug counters
        self.total_calls = 0
        self.monitored_calls = 0

        # Register this communicator globally using weak reference
        # This prevents memory leaks when communicators are recreated
        if enable_monitoring:
            _global_communicators.append(weakref.ref(self))

    def _communicate(
        self,
        *,
        tensor_send_next,
        tensor_send_prev,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape,
        wait_on_reqs: bool = True
    ):
        """Monitored version of _communicate.

        This method wraps the parent _communicate() with lightweight
        monitoring using CUDA events and probabilistic sampling.

        Args:
            tensor_send_next: Tensor to send to next rank
            tensor_send_prev: Tensor to send to previous rank
            recv_prev: Whether to receive from previous rank
            recv_next: Whether to receive from next rank
            tensor_shape: Shape of tensors to receive
            wait_on_reqs: Whether to wait on requests

        Returns:
            Same as parent _communicate: (recv_prev_tensor, recv_next_tensor, reqs)
        """
        self.total_calls += 1

        # Fast path: monitoring disabled or not sampled
        if not self.enable_monitoring or random.random() >= self.sample_rate:
            return super()._communicate(
                tensor_send_next=tensor_send_next,
                tensor_send_prev=tensor_send_prev,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                wait_on_reqs=wait_on_reqs
            )

        # Monitoring path
        self.monitored_calls += 1

        # Compute sizes for monitoring
        send_next_size = (
            tensor_send_next.numel() * tensor_send_next.element_size()
            if tensor_send_next is not None else 0
        )
        send_prev_size = (
            tensor_send_prev.numel() * tensor_send_prev.element_size()
            if tensor_send_prev is not None else 0
        )

        # Create CUDA events for async timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Record start
        start_event.record()

        # Call parent implementation
        result = super()._communicate(
            tensor_send_next=tensor_send_next,
            tensor_send_prev=tensor_send_prev,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=wait_on_reqs
        )

        # Record end (non-blocking)
        end_event.record()

        # Store for async collection
        self.monitor.record_async(
            start_event=start_event,
            end_event=end_event,
            send_next_size=send_next_size,
            send_prev_size=send_prev_size,
            recv_prev=recv_prev,
            recv_next=recv_next
        )

        return result

    def get_network_stats(self) -> Optional[Dict[str, Any]]:
        """Get current network statistics.

        Returns:
            Dictionary with network statistics or None if monitoring disabled
        """
        if not self.enable_monitoring:
            return None

        stats = self.monitor.get_stats()
        stats.update({
            'total_calls': self.total_calls,
            'monitored_calls': self.monitored_calls,
            'sample_rate': self.sample_rate
        })
        return stats


def get_all_communicator_stats() -> List[Dict[str, Any]]:
    """Get stats from all registered P2P communicators.

    Returns:
        List of stats dictionaries from all communicators
    """
    stats_list = []
    # Clean up dead weak references and collect stats from alive communicators
    alive_communicators = []
    for comm_ref in _global_communicators:
        comm = comm_ref()  # Dereference the weak reference
        if comm is not None:  # Communicator still alive
            alive_communicators.append(comm_ref)
            stats = comm.get_network_stats()
            if stats is not None:
                stats_list.append(stats)

    # Update global list to only contain alive communicators
    _global_communicators[:] = alive_communicators

    return stats_list


def get_aggregated_stats() -> Optional[Dict[str, Any]]:
    """Get aggregated stats across all communicators.

    Returns:
        Dictionary with averaged stats or None if no data
    """
    all_stats = get_all_communicator_stats()
    if not all_stats:
        return None

    # Filter out None values and aggregate
    bandwidths = [s['bandwidth_gbps'] for s in all_stats if s.get('bandwidth_gbps') is not None]
    latencies = [s['latency_ms'] for s in all_stats if s.get('latency_ms') is not None]
    total_calls = sum(s.get('total_calls', 0) for s in all_stats)
    monitored_calls = sum(s.get('monitored_calls', 0) for s in all_stats)

    return {
        'bandwidth_gbps': sum(bandwidths) / len(bandwidths) if bandwidths else None,
        'latency_ms': sum(latencies) / len(latencies) if latencies else None,
        'total_calls': total_calls,
        'monitored_calls': monitored_calls,
        'num_communicators': len(all_stats),
        'timestamp': time.time()
    }
