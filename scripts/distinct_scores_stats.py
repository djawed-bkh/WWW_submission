"""Descriptive statistics for inferred universal-domain distinct-score counts."""
import argparse
from pathlib import Path

import pandas as pd


# Edit these thresholds to select the high-tau populations in Table 1.
HIGH_TAUS = {"Country": 190, "Politician": 389, "Musical group": 60}


def load_histograms(input_path: str) -> pd.DataFrame:
    """Validate every population and calculate proportions from counts."""
    data = pd.read_csv(input_path, sep=None, engine="python", comment="#")
    required = {"dataset", "tau", "n_domains", "n_distinct_scores", "count"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data["dataset"].isna().any():
        raise ValueError("Empty input or missing dataset names")
    for column in ("tau", "n_domains", "n_distinct_scores", "count"):
        values = pd.to_numeric(data[column], errors="raise")
        if (values.isna().any() or values.isin([float("inf"), float("-inf")]).any()
                or values.lt(0).any() or values.mod(1).ne(0).any()):
            raise ValueError(f"{column} must contain nonnegative integers")
        data[column] = values.astype("int64")
    if not data["n_distinct_scores"].between(1, 13).all():
        raise ValueError("n_distinct_scores must lie in [1, 13]")
    if data.duplicated(["dataset", "tau", "n_distinct_scores"]).any():
        raise ValueError("Duplicate histogram bins")
    grouped = data.groupby(["dataset", "tau"], sort=False)
    totals = grouped["count"].transform("sum")
    invalid = grouped["n_domains"].transform("nunique").ne(1) | totals.ne(data["n_domains"])
    if invalid.any():
        populations = data.loc[invalid, ["dataset", "tau"]].drop_duplicates()
        raise ValueError(f"Counts do not match n_domains, or n_domains is inconsistent:\n{populations.to_string(index=False)}")
    # Zero populations have undefined statistics, displayed as N/A.
    data["proportion"] = data["count"].div(totals.where(totals.gt(0)))
    return data.sort_values(["dataset", "tau", "n_distinct_scores"])


def generate_distinct_score_tables(input_path, output_dir="distinct_score_tables"):
    """Écrire les tableaux d'effectifs et de pourcentages CSV/LaTeX par BDD.

    Entrée : histogramme dataset, tau, n_domains, n_distinct_scores, count.
    Lignes : 1 à 13 scores distincts ; colonnes : seuils tau triés.
    Chaque cellule est un nombre de domaines (count), pas un nombre de lignes.
    Les catégories absentes valent zéro. L'entrée accepte virgules ou tabulations.
    Les pourcentages sont calculés par colonne ; une population vide affiche --.
    Retourner {BDD: DataFrame des effectifs}.
    """
    data = load_histograms(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {}
    for dataset, population in data.groupby("dataset", sort=True):
        table = population.pivot(index="n_distinct_scores", columns="tau", values="count")
        table = table.reindex(index=range(1, 14), columns=sorted(table.columns)).fillna(0).astype("int64")
        tables[dataset] = table
        slug = "".join(c if c.isalnum() else "-" for c in dataset).strip("-")
        table.to_csv(destination / f"distinct_scores_{slug}.csv")
        display_name = {"Q6256": "Country", "Q215380": "Musical group", "Q82955": "Politician"}.get(dataset, dataset)
        # Escape dataset names so the generated table is valid LaTeX.
        replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                        "$": r"\$", "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}",
                        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
        name = "".join(replacements.get(c, c) for c in display_name)
        lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
                 r"\begin{tabular}{l " + " ".join("r" for _ in table.columns) + "}",
                 r"\hline", "Distinct scores / $\\tau$ & " + " & ".join(str(t) for t in table.columns) + r" \\", r"\hline"]
        for n, row in table.iterrows():
            lines.append(" & ".join([str(n), *(str(int(v)) for v in row)]) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}",
                      rf"\caption{{Number of domains by distinct score count and threshold for {name}.}}",
                      rf"\label{{tab:distinct-scores-{slug}}}", r"\end{table}"])
        (destination / f"distinct_scores_{slug}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        totals = table.sum(axis=0)
        percentages = table.div(totals.where(totals.gt(0)), axis="columns") * 100
        percentages.to_csv(destination / f"distinct_scores_percentages_{slug}.csv",
                           float_format="%.2f", na_rep="--")
        percent_lines = lines[:7]
        for n, row in percentages.iterrows():
            cells = ["--" if pd.isna(value) else f"{value:.2f}" for value in row]
            percent_lines.append(" & ".join([str(n), *cells]) + r" \\")
        percent_lines.extend([r"\hline", r"\end{tabular}",
                             rf"\caption{{Percentage of domains by distinct score count and threshold for {name} (in \%). Each column is normalized by its total number of domains.}}",
                             rf"\label{{tab:distinct-scores-percentages-{slug}}}", r"\end{table}"])
        (destination / f"distinct_scores_percentages_{slug}.tex").write_text(
            "\n".join(percent_lines) + "\n", encoding="utf-8")
    return tables


def calculate_statistics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected-threshold statistics and the stability table for all tau.

    Missing score bins contribute zero. Ties for the mode use the smallest
    n_distinct_scores. Empty populations produce undefined statistics (N/A).
    """
    rows = []
    datasets = list(data["dataset"].drop_duplicates())
    unknown = set(datasets) - set(HIGH_TAUS)
    if unknown:
        raise ValueError(f"Configure HIGH_TAUS for datasets: {sorted(unknown)}")
    for dataset in HIGH_TAUS:
        if dataset not in datasets:
            continue
        subset = data.loc[data["dataset"].eq(dataset)]
        for tau in (0, HIGH_TAUS[dataset]):
            if not subset["tau"].eq(tau).any():
                raise ValueError(f"Missing required population: {dataset}, tau={tau}")
        for tau, population in subset.groupby("tau", sort=True):
            n = int(population["n_domains"].iloc[0])
            distinct = population["n_distinct_scores"]
            undefined = float("nan")
            rows.append({
                "dataset": dataset, "tau": int(tau), "n_domains": n,
                "prop_uniform": population.loc[distinct.eq(1), "proportion"].sum() if n else undefined,
                "prop_4to6": population.loc[distinct.between(4, 6), "proportion"].sum() if n else undefined,
                "prop_3to5": population.loc[distinct.between(3, 5), "proportion"].sum() if n else undefined,
                "mean_distinct": distinct.mul(population["count"]).sum() / n if n else undefined,
                "mode_distinct": int(population.loc[population["count"].eq(population["count"].max()),
                                                    "n_distinct_scores"].min()) if n else pd.NA,
            })
    all_stats = pd.DataFrame(rows)
    all_stats["mode_distinct"] = all_stats["mode_distinct"].astype("Int64")
    selected = all_stats.loc[
        all_stats["tau"].eq(0) | all_stats["tau"].eq(all_stats["dataset"].map(HIGH_TAUS))
    ].reset_index(drop=True)
    stability = all_stats[["dataset", "tau", "n_domains", "prop_uniform", "mean_distinct"]].copy()
    return selected, stability


def format_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """Use percentages with two decimals and means with two decimals."""
    formatted = table.copy()
    for column in table.columns:
        if column.startswith("prop_"):
            formatted[column] = table[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2%}")
        elif column == "mean_distinct":
            formatted[column] = table[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2f}")
        elif column == "mode_distinct":
            formatted[column] = table[column].map(lambda value: "N/A" if pd.isna(value) else str(int(value)))
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="distinct_score_histograms/distinct_score_histograms.txt")
    parser.add_argument("--output", default="distinct_scores_stats.csv")
    parser.add_argument("--tables-dir", help="Generate domain-count tables for each dataset instead of summary statistics.")
    args = parser.parse_args()
    if args.tables_dir:
        try:
            tables = generate_distinct_score_tables(args.input, args.tables_dir)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        for dataset, table in tables.items():
            print(f"\n{dataset}\n{table.to_string()}")
        print(f"\nCSV and LaTeX tables saved: {args.tables_dir}")
        return
    try:
        selected, stability = calculate_statistics(load_histograms(args.input))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    formatted = format_statistics(selected)
    print("Table 1: tau=0 and selected high tau")
    print(formatted.to_string(index=False))
    print("\nTable 2: all thresholds")
    print(format_statistics(stability).to_string(index=False))
    formatted.to_csv(args.output, index=False)
    print(f"\nCSV saved: {args.output}")


if __name__ == "__main__":
    main()
