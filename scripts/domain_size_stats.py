"""Descriptive statistics from aggregated eligible observed-domain sizes.

Run from the project root:
    .venv/bin/python scripts/domain_size_stats.py eligible_observed_domain_sizes.txt
Only pandas is required; distributions are never expanded into individual rows.
"""
import argparse

import pandas as pd


TAU_GRIDS = {
    "Country": [0, 10, 59, 110, 160, 190],
    "Politician": [0, 14, 42, 128, 389, 1185],
    "Musical group": [0, 10, 30, 60, 100, 150],
}


def load_domain_distribution(input_path: str) -> pd.DataFrame:
    """Validate every population and recompute proportions from counts."""
    data = pd.read_csv(input_path, sep="\t", comment="#")
    required = {"dataset", "tau", "n_eligible", "size", "count", "proportion"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data["dataset"].isna().any():
        raise ValueError("Empty input or missing dataset names")
    for column in ("tau", "n_eligible", "size", "count"):
        values = pd.to_numeric(data[column], errors="raise")
        if (values.isna().any() or values.isin([float("inf"), float("-inf")]).any()
                or values.lt(0).any() or values.mod(1).ne(0).any()):
            raise ValueError(f"{column} must contain nonnegative integers")
        data[column] = values.astype("int64")
    if not data["size"].between(1, 13).all():
        raise ValueError("Sizes must lie in [1, 13]")
    if data.duplicated(["dataset", "tau", "size"]).any():
        raise ValueError("Duplicate (dataset, tau, size) bins")
    grouped = data.groupby(["dataset", "tau"], sort=False)
    totals = grouped["count"].transform("sum")
    consistent = grouped["n_eligible"].transform("nunique").eq(1)
    invalid = ~consistent | totals.ne(data["n_eligible"])
    if invalid.any():
        populations = data.loc[invalid, ["dataset", "tau"]].drop_duplicates()
        raise ValueError(f"Counts do not match n_eligible, or n_eligible is inconsistent:\n{populations.to_string(index=False)}")
    if totals.eq(0).any():
        raise ValueError("Empty populations have undefined proportions and medians")
    # Overwrite the supplied proportion column: only counts determine the result.
    data["proportion"] = data["count"].div(totals)
    return data.sort_values(["dataset", "tau", "size"])


def weighted_median(distribution: pd.DataFrame) -> tuple[int, float, float]:
    """Return the first size reaching 50%, and cumulative mass before/at it."""
    distribution = distribution.sort_values("size")
    cumulative_counts = distribution["count"].cumsum()
    total = int(distribution["count"].sum())
    # Integer counts avoid rounding near the 50% boundary.
    median_row = distribution.loc[cumulative_counts.mul(2).ge(total)].iloc[0]
    size = int(median_row["size"])
    before = distribution.loc[distribution["size"].lt(size), "count"].sum() / total
    at = distribution.loc[distribution["size"].le(size), "count"].sum() / total
    return size, float(before), float(at)


def calculate_domain_size_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one summary row per dataset plus median verification diagnostics."""
    rows, diagnostics = [], []
    datasets = [name for name in TAU_GRIDS if name in data["dataset"].unique()]
    unknown = set(data["dataset"]) - set(TAU_GRIDS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    for dataset in datasets:
        subset = data.loc[data["dataset"].eq(dataset)]
        tau_1, tau_max = TAU_GRIDS[dataset][1], TAU_GRIDS[dataset][-1]
        distributions = {}
        for tau in (0, tau_1, tau_max):
            distributions[tau] = subset.loc[subset["tau"].eq(tau)]
            if distributions[tau].empty:
                raise ValueError(f"Missing required population: {dataset}, tau={tau}")
        zero, first, last = (distributions[tau] for tau in (0, tau_1, tau_max))
        medians = {}
        for tau in (0, tau_max):
            median, before, at = weighted_median(distributions[tau])
            medians[tau] = median
            diagnostics.append(dict(dataset=dataset, tau=tau, median=median,
                                    cumulative_before=before, cumulative_at=at))
        rows.append({
            "dataset": dataset,
            "n(τ0)": int(zero["count"].sum()),
            "prop_le3(τ0)": zero.loc[zero["size"].le(3), "proportion"].sum(),
            "prop_le2(τ0)": zero.loc[zero["size"].le(2), "proportion"].sum(),
            "median(τ0)": medians[0], "median(τmax)": medians[tau_max],
            "tau_max": tau_max,
            "prop_13(τ0)": zero.loc[zero["size"].eq(13), "proportion"].sum(),
            "prop_13(τmax)": last.loc[last["size"].eq(13), "proportion"].sum(),
            "prop_1(τ0)": zero.loc[zero["size"].eq(1), "proportion"].sum(),
            "prop_1(τ1)": first.loc[first["size"].eq(1), "proportion"].sum(),
            "tau_1": tau_1,
        })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="eligible_observed_domain_sizes.txt")
    parser.add_argument("--output", default="domain_size_stats.csv")
    args = parser.parse_args()
    try:
        summary, diagnostics = calculate_domain_size_stats(load_domain_distribution(args.input))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    formatted = summary.copy()
    for column in formatted.columns:
        if column.startswith("prop_"):
            formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
    print(formatted.to_string(index=False))
    print("\nWeighted median verification (cumulative proportions):")
    for column in ("cumulative_before", "cumulative_at"):
        diagnostics[column] = diagnostics[column].map(lambda value: f"{value:.1%}")
    print(diagnostics.to_string(index=False))
    formatted.to_csv(args.output, index=False)
    print(f"\nCSV saved: {args.output}")


if __name__ == "__main__":
    main()
