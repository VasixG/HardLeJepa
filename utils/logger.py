import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

Number = Union[int, float]

@dataclass
class Logger:
    logdir: str
    csv_path: str
    use_tb: bool = True

    def __post_init__(self):
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.tb: Optional["SummaryWriter"] = None
        if self.use_tb and SummaryWriter is not None:
            self.tb = SummaryWriter(self.logdir)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = None

    def log(self, step: int, metrics: Dict[str, Number]) -> None:
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        if self._writer is None:
            fieldnames = ["step"] + sorted(metrics.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._writer.writeheader()
        row = {"step": step}
        row.update(metrics)
        self._writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        self._csv_file.close()
