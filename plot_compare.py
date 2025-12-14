import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def read_jsonl_metrics(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_series(rows: List[Dict], metric: str, x_key: str = "step") -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        if metric in r and x_key in r:
            v = r[metric]
            if isinstance(v, (int, float)):
                xs.append(float(r[x_key]))
                ys.append(float(v))
    return xs, ys


def moving_average(y: List[float], window: int) -> List[float]:
    if window <= 1:
        return y
    out = []
    s = 0.0
    q = []
    for v in y:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def find_runs(run_paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in run_paths:
        pp = Path(p)
        if pp.is_dir():
            out.append(pp)
        else:
            out.extend(sorted(Path().glob(p)))
    out = [r for r in out if (r / "metrics.jsonl").exists()]
    return out


def plot_metric(
    runs: List[Path],
    metric: str,
    x_key: str,
    smooth: int,
    title: Optional[str],
    out_path: Optional[str],
):
    plt.figure(figsize=(9, 5))
    for r in runs:
        rows = read_jsonl_metrics(r / "metrics.jsonl")
        xs, ys = extract_series(rows, metric=metric, x_key=x_key)
        if not xs:
            continue
        ys_s = moving_average(ys, smooth)
        plt.plot(xs, ys_s, label=r.name)

    plt.xlabel(x_key)
    plt.ylabel(metric)
    plt.title(title or f"{metric} vs {x_key}")
    plt.legend()
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    else:
        plt.show()

    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Folders runs/<exp> or glob patterns.")
    ap.add_argument("--metric", required=True, help="Metric key in jsonl, e.g. eval_student/knn_acc1")
    ap.add_argument("--x", default="step", help="X axis key: step or epoch")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (1 = none)")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="Output png path. If omitted -> show()")
    args = ap.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit("No runs with metrics.jsonl found. Check --runs paths/patterns.")

    plot_metric(
        runs=runs,
        metric=args.metric,
        x_key=args.x,
        smooth=args.smooth,
        title=args.title,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
