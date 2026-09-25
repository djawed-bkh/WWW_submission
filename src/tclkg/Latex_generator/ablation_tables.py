#!/usr/bin/env python3
"""Parse weighted-MRR ablation results and emit a LaTeX table per dataset.

Input format (one result file, free-form lines mixed with section separators):

    <composition>/<dataset>/occ_0p<X>_min_<N>_timeout_86400/<file>.json: weighted MRR = <value>
    Skipping ablation for <composition>/<dataset>/occ_0p<X>_min_<N>_.../<file>.json: <reason>

Any other line (blank lines, "----" separators, mislabeled section headers)
is ignored.

Usage:
    python generate_ablation_table.py Ablation_results.txt > ablation_tables.tex
"""

import argparse
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"(?P<composition>reduced_composition|non_reduced_composition)/"
    r"(?P<dataset>[^/]+)/"
    r"occ_0p(?P<occ>\d+)_min_(?P<min>\d+)_timeout_\d+/"
    r"[^:]+:\s*"
    r"(?:.*?weighted MRR = (?P<mrr>[-\d.eE]+)"
    r"|.*?\(at least (?P<required>\d+) required\))"
)


def parse_results(path):
    """Return {dataset: {(occ, min): {composition: mrr_or_None}}}."""
    data = defaultdict(lambda: defaultdict(dict))

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                # Skip separators / mislabeled section headers / anything else.
                continue

            composition = m.group("composition")
            dataset = m.group("dataset")
            occ = float(f"0.{m.group('occ')}")
            min_support = int(m.group("min"))
            mrr = float(m.group("mrr")) if m.group("mrr") is not None else None

            counts = re.search(
                r"observed relations = (\d+).*?missing relations = (\d+)/(\d+)", line
            )
            data[dataset][(occ, min_support)][composition] = (
                {"mrr": mrr, "observed": int(counts[1]),
                 "missing": int(counts[2]), "masked": int(counts[3])}
                if counts else mrr
            )

    return data


def format_value(value):
    if isinstance(value, dict):
        return (f"{value['mrr']:.3f} & {value['observed']} & "
                f"{value['missing']}/{value['masked']}")
    if value is None:
        return "-- & -- & --"
    return f"{value:.3f} & -- & --"


def build_latex_table(dataset, entries, caption=None, label=None):
    """entries: {(occ, min): {composition: mrr_or_None}}"""
    occs = sorted({k[0] for k in entries})
    caption = caption or (f"Weighted MRR ablation results for {dataset}. "
                          "Obs.: eligible observed Allen relations; Miss./Mask.: missing "
                          "predictions over masked relations, summed over ten rounds. "
                          "Directed pairs are counted separately; -- means unavailable.")
    label = label or f"tab:ablation_{dataset.lower()}"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{cc ccc ccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$\tau$ & min & \multicolumn{3}{c}{non-reduced comp.} & \multicolumn{3}{c}{reduced comp.} \\"
    )
    lines.append(r" & & MRR & Obs. & Miss./Mask. & MRR & Obs. & Miss./Mask. \\")
    lines.append(r"\midrule")

    for occ in occs:
        mins = sorted(k[1] for k in entries if k[0] == occ)
        n = len(mins)
        for i, min_support in enumerate(mins):
            values = entries[(occ, min_support)]
            non_reduced = format_value(values.get("non_reduced_composition"))
            reduced = format_value(values.get("reduced_composition"))

            if i == 0:
                occ_cell = f"\\multirow{{{n}}}{{*}}{{{occ:.2f}}}"
            else:
                occ_cell = ""

            lines.append(
                f"{occ_cell} & {min_support} & {non_reduced} & {reduced} \\\\"
            )
        if occ != occs[-1]:
            lines.append(r"\cmidrule{1-8}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the ablation results text file")
    parser.add_argument(
        "-o", "--output", help="Output .tex file (defaults to stdout)"
    )
    args = parser.parse_args()

    data = parse_results(args.input)

    if not data:
        print(
            "No results parsed -- check that the input format matches the "
            "expected pattern.",
            file=sys.stderr,
        )
        sys.exit(1)

    tables = []
    for dataset in sorted(data.keys()):
        tables.append(build_latex_table(dataset, data[dataset]))

    output_text = "\n\n".join(tables) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()