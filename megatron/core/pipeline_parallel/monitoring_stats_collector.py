# Copyright (c) 2024, Adaptive Pipeline Parallelism Research
# Global monitoring stats collector for runtime P2P monitoring

"""
Global statistics collector for P2P network monitoring.

This module provides a singleton pattern for collecting and accessing
P2P monitoring statistics across the training loop.
"""

from typing import Optional, Dict, Any, List
import threading


class MonitoringStatsCollector:
    """Global singleton for collecting P2P monitoring stats."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.bandwidth_history = []
        self.latency_history = []
        self.iteration_stats = {}  # {iteration: {bandwidth, latency, ...}}
        self._initialized = True

    def record_stats(self, iteration: int, stats: Dict[str, Any]):
        """Record stats for a specific iteration.

        Args:
            iteration: Training iteration number
            stats: Dictionary with bandwidth_gbps, latency_ms, etc.
        """
        self.iteration_stats[iteration] = stats

        if stats.get('bandwidth_gbps') is not None:
            self.bandwidth_history.append({
                'iteration': iteration,
                'value': stats['bandwidth_gbps']
            })

        if stats.get('latency_ms') is not None:
            self.latency_history.append({
                'iteration': iteration,
                'value': stats['latency_ms']
            })

    def get_latest_stats(self) -> Optional[Dict[str, Any]]:
        """Get the most recent stats.

        Returns:
            Dictionary with latest stats or None if no stats available
        """
        if not self.iteration_stats:
            return None

        latest_iter = max(self.iteration_stats.keys())
        return self.iteration_stats[latest_iter]

    def get_stats_for_iteration(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Get stats for a specific iteration.

        Args:
            iteration: Training iteration number

        Returns:
            Dictionary with stats or None if not available
        """
        return self.iteration_stats.get(iteration)

    def get_bandwidth_history(self) -> List[Dict[str, Any]]:
        """Get full bandwidth history.

        Returns:
            List of {iteration, value} dictionaries
        """
        return self.bandwidth_history.copy()

    def get_latency_history(self) -> List[Dict[str, Any]]:
        """Get full latency history.

        Returns:
            List of {iteration, value} dictionaries
        """
        return self.latency_history.copy()

    def clear(self):
        """Clear all collected stats."""
        self.bandwidth_history.clear()
        self.latency_history.clear()
        self.iteration_stats.clear()


# Global accessor functions
_collector = None


def get_monitoring_collector() -> MonitoringStatsCollector:
    """Get the global monitoring stats collector.

    Returns:
        Singleton instance of MonitoringStatsCollector
    """
    global _collector
    if _collector is None:
        _collector = MonitoringStatsCollector()
    return _collector


def record_monitoring_stats(iteration: int, stats: Dict[str, Any]):
    """Convenience function to record stats.

    Args:
        iteration: Training iteration number
        stats: Dictionary with monitoring stats
    """
    collector = get_monitoring_collector()
    collector.record_stats(iteration, stats)


def get_latest_monitoring_stats() -> Optional[Dict[str, Any]]:
    """Convenience function to get latest stats.

    Returns:
        Latest monitoring stats or None
    """
    collector = get_monitoring_collector()
    return collector.get_latest_stats()
