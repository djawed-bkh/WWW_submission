"""Calculate weighted means per dataset and threshold from histogram counts."""
import argparse
from pathlib import Path

import pandas as pd


def calculate_mean_sizes(input_path, size_column="size"):
    """Return sum(size * count) / sum(count) for every dataset and tau.

    size_column may be n_distinct_scores for distinct-score histograms.
    Zero populations have an undefined mean (NaN).
    """
    data = pd.read_csv(input_path, sep=None, engine="python", comment="#")
    required = {"dataset", "tau", size_column, "count"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data['dataset'].isna().any():
        raise ValueError('Empty input or missing dataset')
    for column in ('tau', size_column, 'count'):
        values = pd.to_numeric(data[column], errors='raise')
        if (values.isna().any() or values.isin([float('inf'), float('-inf')]).any()
                or values.lt(0).any() or values.mod(1).ne(0).any()):
            raise ValueError(f'{column} must contain nonnegative integers')
        data[column] = values.astype('int64')
    if not data[size_column].between(1, 13).all():
        raise ValueError('Sizes must lie in [1, 13]')
    if data.duplicated(['dataset', 'tau', size_column]).any():
        raise ValueError('Duplicate histogram bins')
    data['weighted_sum'] = data[size_column] * data['count']
    grouped = data.groupby(['dataset', 'tau'], sort=True)
    result = grouped.agg(population=('count', 'sum'), weighted_sum=('weighted_sum', 'sum'))
    for column in ('n_domains', 'n_eligible'):
        if column in data:
            declared = pd.to_numeric(data[column], errors='raise')
            data[column] = declared
            if (grouped[column].nunique().ne(1).any()
                    or grouped[column].first().ne(result.population).any()):
                raise ValueError(f'Histogram totals do not match {column}')
    result['mean_size'] = result.weighted_sum / result.population.where(result.population.gt(0))
    return result.reset_index()


def generate_mean_size_tables(input_path, output_dir='mean_domain_sizes', size_column='size'):
    """Write full-precision CSV and English LaTeX tables (two decimals)."""
    result = calculate_mean_sizes(input_path, size_column)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'mean_sizes.csv', index=False)
    for dataset, rows in result.groupby('dataset', sort=True):
        slug = ''.join(c if c.isalnum() else '-' for c in dataset).strip('-')
        metric = 'Mean distinct score count' if size_column == 'n_distinct_scores' else 'Mean domain size'
        lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
                 r'\begin{tabular}{r r r}', r'\hline',
                 rf'$\tau$ & Population & {metric} \\', r'\hline']
        for row in rows.itertuples():
            mean = '--' if pd.isna(row.mean_size) else f'{row.mean_size:.2f}'
            lines.append(f'{row.tau} & {row.population} & {mean}' + r' \\')
        name = str(dataset).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
        lines.extend([r'\hline', r'\end{tabular}',
                      rf'\caption{{{metric} by threshold for {name}.}}',
                      rf'\label{{tab:mean-size-{slug}}}', r'\end{table}'])
        (output / f'mean_sizes_{slug}.tex').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--size-column', choices=('size', 'n_distinct_scores'), default='size')
    parser.add_argument('--output-dir', default='mean_domain_sizes')
    args = parser.parse_args()
    print(generate_mean_size_tables(args.input, args.output_dir, args.size_column).to_string(index=False))
