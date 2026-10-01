"""Common interface for experiment tracking.

Same idea as BaseBPModel: train.py talks to this interface only, so if a
a different backend is ever wanted, it can be
added as a new ExperimentTracker implementation without touching
train.py. CSVTracker is currently the only implementation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ExperimentTracker(ABC):
    @abstractmethod
    def start_run(self, experiment_id: str, run_name: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Dict, units: Optional[Dict] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_tags(self, tags: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_run(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def run_dir(self) -> str:
        """Local directory where this run's artifacts are staged/stored,
        used by train.py to know where to write predictions.csv etc.
        before handing them to log_artifact().
        """
        raise NotImplementedError
