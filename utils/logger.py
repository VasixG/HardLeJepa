import csv
import json
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
        self.jsonl_path = str(Path(self.csv_path).with_suffix(".jsonl"))
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames = ["step"]

    def _rewrite_csv_with_new_fields(self, new_fieldnames):
        self._csv_file.close()
        rows = []
        src = Path(self.csv_path)
        if src.exists() and src.stat().st_size > 0:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        tmp = src.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in new_fieldnames})
        tmp.replace(src)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=new_fieldnames)
        self._fieldnames = list(new_fieldnames)

    def log(self, step: int, metrics: Dict[str, Number]) -> None:
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            payload = {"step": int(step)}
            payload.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})
            jf.write(json.dumps(payload, ensure_ascii=False) + "\n")
        keys = ["step"] + sorted(metrics.keys())
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=keys)
            self._writer.writeheader()
            self._fieldnames = list(keys)
        else:
            missing = [k for k in keys if k not in self._fieldnames]
            if missing:
                new_fields = ["step"] + sorted(set(self._fieldnames[1:] + missing))
                self._rewrite_csv_with_new_fields(new_fields)
        row = {"step": step}
        row.update(metrics)
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._csv_file.flush()

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        self._csv_file.close()
