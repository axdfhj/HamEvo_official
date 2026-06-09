import json

import numpy as np
import torch
from torchmetrics import MeanMetric, MinMetric

from .distributed_utils import gather_dict


class MetricsCollector:
    def __init__(self):
        self.data = {}

    def add(self, key: str, value: torch.Tensor):
        """Add a single value to a tracked key."""
        if key not in self.data:
            self.data[key] = []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        else:
            raise TypeError(f"value must be a Tensor, got {type(value)}")
        self.data[key].append(value)

    def get(self, key: str) -> torch.Tensor:
        return torch.stack(self.data[key])

    def all(self) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in self.data.items():
            if isinstance(v, list):
                if len(v) == 0:
                    out[k] = torch.empty(0)
                elif len(v) == 1:
                    out[k] = v[0].unsqueeze(0) if v[0].dim() == 0 else v[0]
                else:
                    # 确保所有张量至少是一维的
                    tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in v]
                    out[k] = torch.cat(tensors)
            elif isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0) if v.dim() == 0 else v
            else:
                raise RuntimeError(f"Invalid data type for key '{k}': {type(v)}")
        return out

    def reset(self):
        self.data.clear()

    def gather(self, dst: int = 0) -> dict:
        """Gather data from all processes to the destination process.

        Args:
            dst: Destination rank (default: 0)

        Returns:
            Combined data dictionary if on destination rank, None otherwise
        """
        # Gather the raw data dictionaries
        gathered_data = gather_dict(self.data, dst=dst)

        if gathered_data is None:
            return None

        # Create a new MetricsCollector with the gathered data
        gathered_collector = MetricsCollector()
        gathered_collector.data = gathered_data

        return gathered_collector

    def save(self, path: str, format: str = "pt"):
        all_data = self.all()
        if format == "pt":
            torch.save(all_data, path)
        elif format == "npy":
            np_data = {k: v.numpy() for k, v in all_data.items()}
            np.save(path, np_data)
        elif format == "json":
            json_data = {k: v.numpy().tolist() for k, v in all_data.items()}
            with open(path, "w") as f:
                json.dump(json_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MetricRegistry:
    def __init__(self):
        self.metrics = {}

    def register(self, name: str, metric_type="mean"):
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already registered.")
        if metric_type == "mean":
            self.metrics[name] = MeanMetric()
        elif metric_type == "min":
            self.metrics[name] = MinMetric()
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        return self.metrics[name]

    def get(self, name: str):
        return self.metrics[name]

    def log_all(self, module, prefix="", sync_dist=True, prog_bar=True):
        for name, metric in self.metrics.items():
            val = metric.compute()
            module.log(f"{prefix}{name}", val, sync_dist=sync_dist, prog_bar=prog_bar)

    def reset_all(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, name: str, value):
        self.get(name).update(value)

    def __getitem__(self, name: str):
        return self.get(name)

    def __contains__(self, name: str):
        return name in self.metrics

    def to(self, device):
        for metric in self.metrics.values():
            metric.to(device)
