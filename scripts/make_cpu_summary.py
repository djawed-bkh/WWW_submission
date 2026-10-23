#!/usr/bin/env python3
"""Generate the CPU table from supplied values or benchmark_cpu JSON output."""
import argparse
import json
import re
from pathlib import Path

# Values supplied with the requested table; not taken from temps_cpu.txt.
SUPPLIED = {
    "Q6256": {"oracle": 33.3, "times": [136.37, 117.69, 82.10, 79.28, 77.28, 77.86]},
    "Q215380": {"oracle": 14.4, "times": [26.74, 24.71, 24.26, 24.31, 24.71, 24.17]},
    "Q82955": {"oracle": 184.0, "times": [542.0, 489.9, 478.0, 475.4, 473.4, 473.6]},
}
DATASETS = [
    ("Q6256", "Country", [0, 10, 59, 110, 160, 190]),
    ("Q215380", "Musical group", [0, 10, 30, 60, 100, 150]),
    ("Q82955", "Politician", [0, 14, 42, 128, 389, 1185]),
]


def load_benchmark(path):
    """Select occurrence=0 and reduced composition, without averaging configs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for qid, _, taus in DATASETS:
        entry = raw[qid]
        times = {}
        for config, seconds in entry["propagation_mean_cpu_s"].items():
            parts = config.replace("\\", "/").split("/")
            match = next((m for part in parts if (m := re.fullmatch(
                r"occ_([0-9p.]+)_min_(\d+)_timeout_\d+", part))), None)
            reduced = "reduced_composition" in parts or (
                parts[-1].endswith("_reduced_composition.json")
                and not parts[-1].endswith("_non_reduced_composition.json")
            )
            if match is None or float(match[1].replace("p", ".")) != 0 or not reduced:
                continue
            tau = int(match[2])
            if tau in times:
                raise ValueError(f"Configuration ambiguë : {qid}, tau={tau}")
            times[tau] = seconds
        result[qid] = {"oracle": entry["oracle"]["mean_cpu_s"],
                       "times": [times[tau] for tau in taus]}
    return result


def render(data):
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{5pt}", r"\scalebox{0.9}{%",
             r"\begin{tabular}{@{}l r l rrrrrr@{}}", r"\toprule",
             r"Dataset & Orac. & & \multicolumn{6}{c}{} \\"]
    for qid, name, taus in DATASETS:
        entry = data[qid]
        lines.extend([
            r"\midrule",
            rf"\multirow{{2}}{{*}}{{{name}}} & \multirow{{2}}{{*}}{{{entry['oracle']:.1f}}}",
            r"  & $\tau$ & " + " & ".join(map(str, taus)) + r" \\",
            r"& & Inf. & " + " & ".join(f"{t:.2f}" for t in entry["times"]) + r" \\",
        ])
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}",
                  r"\caption{Computation time in seconds. The oracle phase does not depend",
                  r"on $\tau$; Inf. reports propagation time only.}",
                  r"\label{tab:cpu}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="benchmark_cpu JSON; defaults to the values supplied in the table")
    parser.add_argument("--output", type=Path, default=Path("latex_tables/tab_cpu.tex"))
    args = parser.parse_args()
    data = load_benchmark(args.input) if args.input else SUPPLIED
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(f"Tableau écrit dans {args.output}")


if __name__ == "__main__":
    main()
