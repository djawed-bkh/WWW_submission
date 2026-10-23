"""
Fusionne les deux CSV (domain sizes + inference scores) sur les clés communes
(kg, version, occurence_thresh, support_thresh) et génère, pour chaque dataset,
un unique tableau LaTeX avec deux groupes de colonnes ("Domain size" /
"Inference score"), Version et tau_c mutualisés via \\multirow imbriqué.

Usage:
    python generate_combined_tables.py <sizes.csv> <scores.csv> <output_dir>
"""

import sys
from pathlib import Path

import pandas as pd

SIZE_COLS = [
    "n_inferred_domains",
    "%inferred from none",
    "%domaines_informatifs",
    "min_domain_sizes",
    "avg_domain_sizes",
    "median_domain_sizes",
    "max_domain_sizes",
]
SIZE_HEADERS = [
    "N inf.",
    "\\% inf. from none",
    "\\% informative",
    "Min",
    "Avg.",
    "Med.",
    "Max",
]

SCORE_COLS = [
    "min_inference_score",
    "avg_best_candidate_score",
    "avg_domain_mean_score",
    "max_inference_score",
]
SCORE_HEADERS = [
    "Min",
    "Avg. best cand.",
    "Avg. domain mean",
    "Max",
]

VERSIONS_ORDER = ["non_reduced_composition", "reduced_composition"]
VERSION_LABELS = {
    "non_reduced_composition": "all comp. allowed",
    "reduced_composition": "only $\\leq 13$ comp.",
}

KEYS = ["kg", "version", "occurence_thresh", "support_thresh"]


def format_combined_table(df: pd.DataFrame, kg: str) -> str:
    n_metric_cols = len(SIZE_COLS) + len(SCORE_COLS)
    col_spec = "ll" + "r" * (1 + n_metric_cols)  # version, tau_c, tau, then metrics

    n_poss = df[df["kg"] == kg]["n_domains (possible inferences)"].iloc[0]
    n_poss_str = f"{int(n_poss)}" if float(n_poss).is_integer() else f"{n_poss}"

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Inference power \\& score evaluation -- {kg} "
        f"($N$ possible domains $= {n_poss_str}$, constant across all settings)}}"
    )
    lines.append(f"\\label{{tab:combined_{kg}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(
        f" & & & \\multicolumn{{{len(SIZE_COLS)}}}{{c}}{{Domain size}} & "
        f"\\multicolumn{{{len(SCORE_COLS)}}}{{c}}{{Inference score}} \\\\"
    )
    lines.append(f"\\cmidrule(lr){{4-{3+len(SIZE_COLS)}}}\\cmidrule(l){{{4+len(SIZE_COLS)}-{3+n_metric_cols}}}")
    lines.append(
        "Version & $\\tau_c$ & $\\tau$ & "
        + " & ".join(SIZE_HEADERS)
        + " & "
        + " & ".join(SCORE_HEADERS)
        + " \\\\"
    )
    lines.append("\\midrule")

    for v in VERSIONS_ORDER:
        sub_v = df[(df["kg"] == kg) & (df["version"] == v)].sort_values(
            ["occurence_thresh", "support_thresh"]
        )
        occ_thresholds = sorted(sub_v["occurence_thresh"].unique())
        n_rows_version = len(sub_v)
        first_row_of_version = True
        for occ in occ_thresholds:
            sub_occ = sub_v[sub_v["occurence_thresh"] == occ]
            n_rows_occ = len(sub_occ)
            first_row_of_occ = True
            for _, r in sub_occ.iterrows():
                row = []
                if first_row_of_version:
                    row.append(f"\\multirow{{{n_rows_version}}}{{*}}{{{VERSION_LABELS[v]}}}")
                    first_row_of_version = False
                else:
                    row.append("")
                if first_row_of_occ:
                    row.append(f"\\multirow{{{n_rows_occ}}}{{*}}{{{occ:.2f}}}")
                    first_row_of_occ = False
                else:
                    row.append("")
                row.append(str(int(r["support_thresh"])))
                for c in SIZE_COLS + SCORE_COLS:
                    val = r[c]
                    if pd.isna(val):
                        row.append("--")
                    elif isinstance(val, float):
                        row.append(f"{val:.2f}")
                    else:
                        row.append(str(val))
                lines.append(" & ".join(row) + " \\\\")
            lines.append(f"\\cmidrule(l){{2-{3 + n_metric_cols}}}")
        lines.pop()
        lines.append("\\midrule")

    lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main(sizes_csv: str, scores_csv: str, output_dir: str):
    sizes = pd.read_csv(sizes_csv)
    scores = pd.read_csv(scores_csv)

    merged = pd.merge(
        sizes,
        scores,
        on=KEYS,
        how="outer",
        suffixes=("", "_scores"),
        validate="one_to_one",
    )

    unmatched = merged[merged[SCORE_COLS[0]].isna() | merged[SIZE_COLS[0]].isna()]
    if len(unmatched):
        print(f"ATTENTION: {len(unmatched)} ligne(s) sans correspondance entre les deux CSV:")
        print(unmatched[KEYS].to_string(index=False))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for kg in sorted(merged["kg"].unique()):
        tex = format_combined_table(merged, kg)
        fname = out / f"table_combined_{kg}.tex"
        fname.write_text(tex, encoding="utf-8")
        print(f"Généré: {fname}")
    print("Note: tableaux -> \\usepackage{multirow, booktabs} requis dans le préambule.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_combined_tables.py <sizes.csv> <scores.csv> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])