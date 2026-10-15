"""Status-aware domain populations and sizes used in the publication plots."""
import json

import numpy as np

from tclkg import random_experiments as experiments


def test_json_populations_use_status_and_positive_support(tmp_path):
    for qid, dataset in experiments.DOMAIN_SIZE_QIDS.items():
        for index, tau in enumerate(experiments.DOMAIN_SIZE_TAUS[dataset]):
            directory = tmp_path / qid / 'reduced_composition' / f'occ_0_min_{tau}_timeout_86400'
            directory.mkdir(parents=True)
            domains = {
                'observed': {'status': 'Observed', 'relations': {
                    'before': {'support': [3, 10]},
                    'after': {'support': [0, 10]},
                    'meets': {'support': [1, 10]},
                }},
                'inferred': {'status': 'Inferred', 'relations': {
                    'before': {'inferred_score': 0.7},
                    'after': {'inferred_score': 0.1},
                    'meets': {'inferred_score': 0},
                    'during': {'inferred_score': -0.1},
                    'equals': {},
                    'starts': {'inferred_score': None},
                }},
                'unknown': {'status': None, 'relations': {str(i): {} for i in range(13)}},
                'changing': {'status': 'Observed' if index == 0 else 'Inferred',
                             'relations': {'before': {'support': [1, 10], 'inferred_score': 0.5}}},
            }
            (directory / f'qcn2_{qid}_reduced_composition.json').write_text(
                json.dumps({'after_propagation': domains})
            )
    rows = experiments.load_domain_size_rows(tmp_path)
    assert not any(row['domain_id'] == 'unknown' for row in rows)
    observed = next(row for row in rows if row['domain_id'] == 'observed')
    assert observed['size_observed'] == 2
    assert observed['size_inferred'] is None
    inferred = next(row for row in rows if row['domain_id'] == 'inferred')
    assert inferred['size_inferred'] == 2
    assert inferred['size_observed'] is None
    variable = experiments.calculate_domain_size_proportions(rows)
    for dataset in experiments.DOMAIN_SIZE_TAUS:
        assert variable[dataset]['counts']['size_observed'] == [2, 1, 1, 1, 1, 1]
        assert variable[dataset]['counts']['size_inferred'] == [1, 2, 2, 2, 2, 2]
        for field in ('size_observed', 'size_inferred'):
            assert np.allclose(variable[dataset][field].sum(axis=0), 1)


def test_empty_status_population_is_undefined(tmp_path):
    rows = [dict(dataset=dataset, tau=tau, domain_id='observed',
                 size_observed=1, size_inferred=None)
            for dataset, taus in experiments.DOMAIN_SIZE_TAUS.items() for tau in taus]
    summary = experiments.calculate_domain_size_proportions(rows)
    for dataset in experiments.DOMAIN_SIZE_TAUS:
        assert summary[dataset]['counts']['size_inferred'] == [0] * 6
        assert np.isnan(summary[dataset]['size_inferred']).all()
    output = experiments.plot_domain_size_heatmap(summary, tmp_path / 'figure.pdf')
    assert output.read_bytes().startswith(b'%PDF')
