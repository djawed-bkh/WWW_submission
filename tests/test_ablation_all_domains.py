import json

import pytest

from tclkg import random_experiments as rexp
from tclkg import qcn_generator2 as qgen


@pytest.mark.parametrize('count', [0, 7, 23, 120])
def test_all_domains_balanced_folds(tmp_path, monkeypatch, count):
    monkeypatch.setattr(rexp, 'results_root', tmp_path)
    folder = tmp_path / 'conservative/Q42/reduced_composition/occ_0p5_min_10_timeout_600'
    folder.mkdir(parents=True)
    source = folder / 'qcn2_Q42_reduced_composition.json'
    domain = {'status': 'Observed', 'relations': {'before': {'support': [8, 10]}}}
    original = {'after_oracle': {f'p{i}___q{i}': domain for i in range(count)}}
    original['after_oracle']['invalid___pair'] = {
        'status': 'Observed', 'relations': {'before': {'support': [1, 10]}}
    }
    source.write_text(json.dumps(original))
    calls = []

    def propagate(qcn, **kwargs):
        masked = [pair for pair, value in qcn.items() if value['status'] is None]
        calls.append(masked)
        assert qcn[('invalid', 'pair')]['status'] == 'Observed'
        for pair in masked:
            assert qcn[pair]['relations']['before']['support'] == [0, 0]
            qcn[pair] = {'status': 'Inferred', 'relations': {'before': {'inferred_score': 1.0}}}
        return qcn

    monkeypatch.setattr(qgen, 'propagateAndFilterConservative', propagate)
    details = rexp.ablation_study_of_inference_all_domains(str(source), seed=42, return_details=True)
    assert json.loads(source.read_text()) == original
    assert not (tmp_path / 'MRR').exists()
    if count == 0:
        assert details is None
        assert calls == []
        return
    assert details['n_observed'] == details['n_masked'] == count
    assert details['mrr'] == 1.0
    assert details['n_unreached'] == 0
    archive = tmp_path / 'MRR_all_domains/Q42/reduced_composition' / folder.name
    rounds = json.loads((archive / 'masked_rounds.json').read_text())
    assert len(rounds) == 10
    sizes = [len(r['selected_pairs']) for r in rounds]
    assert max(sizes) - min(sizes) <= 1
    selected = [pair for r in rounds for pair in r['selected_pairs']]
    assert len(selected) == len(set(selected)) == count
    rexp.ablation_study_of_inference_all_domains(str(source), seed=42)
    assert json.loads((archive / 'masked_rounds.json').read_text()) == rounds


def test_run_all_domains_selects_only_requested_conservative_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(rexp, 'results_root', tmp_path)
    for mode, qid in [('conservative', 'Q6256'), ('conservative', 'Q42'), ('two_stage', 'Q6256')]:
        for minimum in (10, 20):
            folder = tmp_path / mode / qid / 'reduced_composition' / f'occ_0p5_min_{minimum}_timeout_600'
            folder.mkdir(parents=True)
            (folder / f'qcn2_{qid}_reduced_composition.json').write_text('{}')
    calls = []

    def ablation(source, **kwargs):
        calls.append((source, kwargs))
        return {'mrr': 0.5}

    monkeypatch.setattr(rexp, 'ablation_study_of_inference_all_domains', ablation)
    reports = rexp.run_ablation_studies_of_inference_all_domains('Q6256', seed=17)
    assert len(reports) == len(calls) == 2
    assert all(key.startswith('conservative/Q6256/') for key in reports)
    assert all(kwargs == {'seed': 17, 'return_details': True} for _, kwargs in calls)
    with pytest.raises(ValueError):
        rexp.run_ablation_studies_of_inference_all_domains('../Q6256')
    with pytest.raises(FileNotFoundError):
        rexp.run_ablation_studies_of_inference_all_domains('Q999')


@pytest.mark.parametrize('converse_valid', [True, False])
def test_converses_are_always_masked_together(tmp_path, monkeypatch, converse_valid):
    monkeypatch.setattr(rexp, 'results_root', tmp_path)
    folder = tmp_path / 'conservative/Q42/reduced_composition/occ_0p5_min_10_timeout_600'
    folder.mkdir(parents=True)
    source = folder / 'qcn2_Q42_reduced_composition.json'
    domains = {}
    for i in range(23):
        domains[f'p{i}___q{i}'] = {'status': 'Observed', 'relations': {'before': {'support': [8, 10]}}}
        domains[f'q{i}___p{i}'] = {'status': 'Observed', 'relations': {'after': {
            'support': [8 if converse_valid else 1, 10], 'inferred_score': 0.9,
        }}}
    source.write_text(json.dumps({'after_oracle': domains}))

    def propagate(qcn, **kwargs):
        for (p1, p2), domain in qcn.items():
            if domain['status'] is None:
                assert qcn[(p2, p1)]['status'] is None
                for evidence in domain['relations'].values():
                    assert evidence['support'] == [0, 0]
                    assert 'inferred_score' not in evidence
        return qcn

    monkeypatch.setattr(qgen, 'propagateAndFilterConservative', propagate)
    result = rexp.ablation_study_of_inference_all_domains(str(source), seed=42, return_details=True)
    assert result['n_masked'] == (46 if converse_valid else 23)
    archive = tmp_path / 'MRR_all_domains/Q42/reduced_composition' / folder.name
    rounds = json.loads((archive / 'masked_rounds.json').read_text())
    masked = [pair for r in rounds for pair in r['masked_pairs']]
    assert len(masked) == len(set(masked)) == 46
    assert sorted(len(r['masked_pairs']) for r in rounds) == [4] * 7 + [6] * 3
    assert json.loads(source.read_text())['after_oracle'] == domains
