import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('fact_checking', Path(__file__).resolve().parents[1] / 'fact_checking.py')
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def test_validation_votes_abstentions_and_no_leakage(tmp_path):
    data = tmp_path / 'data' / 'Q1'
    data.mkdir(parents=True)
    p, q = 'http://www.wikidata.org/prop/P1', 'http://www.wikidata.org/prop/P2'
    (data / 'train_cst_knowledge.quintuplet').write_text(f'e\t{q}\tv\t2005-01-01\t2006-01-01\n')
    (data / 'valid.quintuplet').write_text(
        f'e\t{p}\tv\t2000-01-01\t2001-01-01\n'
        f'e\t{p}\tv\t2007-01-01\t2008-01-01\n'
        f'unknown\t{p}\tv\t2000-01-01\t2001-01-01\n'
        f'e\t{p}\tv\tNone\tNone\n'
        f'e\tP3\tv\t2000-01-01\t2001-01-01\n')
    (data / 'valid.gt').write_text('True\nFalse\nTrue\nTrue\nFalse\n')
    output = tmp_path / 'output'
    rule = output / 'result_train/tcl/Q1/min_total_1/Q1/conservative/qcn2_Q1_reduced_composition.json'
    rule.parent.mkdir(parents=True)
    rule.write_text(json.dumps({'after_propagation': {
        f'{p}___{q}': {'status': 'Observed', 'relations': {'before': {'support': [1, 1]}}},
        f'{q}___{p}': {'status': 'Observed', 'relations': {'after': {'support': [1, 1]}}}}}))
    metrics = fc.run_validation(data.parent, output, ['Q1'])[0]
    assert metrics['accuracy'] == 1
    assert metrics['coverage'] == 0.4
    assert [metrics[f'code_{i}_count'] for i in range(5)] == [1, 1, 1, 1, 1]
    breakdown = metrics['abstention_breakdown']
    assert {row['code'] for row in breakdown} == {2, 4, '3a', '3b'}
    for row in breakdown:
        if row['code'] == '3b':
            assert row['count'] == row['percent_of_facts'] == row['percent_of_abstentions'] == 0
            continue
        assert row['count'] == 1
        assert row['percent_of_facts'] == 20
        assert row['percent_of_abstentions'] == pytest.approx(100 / 3)
    assert (output / 'result_valid/tcl/summary.csv').is_file()
    # Strict threshold, matching Soulard.
    fact = fc.load_facts(data / 'valid.quintuplet')[0]
    known = fc.load_facts(data / 'train_cst_knowledge.quintuplet')[0]
    assert fc.predict_fact(fact, {'e': {q: [known]}}, fc.load_constraint_index(rule), 1)[0] == 1
    (data / 'valid.gt').write_text('True\n')
    with pytest.raises(ValueError, match='étiquette'):
        fc.run_validation(data.parent, output, ['Q1'])


def test_validation_without_propagation_uses_oracle_and_returns_all_metrics(tmp_path, monkeypatch):
    data = tmp_path / 'data' / 'Q1'
    data.mkdir(parents=True)
    (data / 'train_cst_knowledge.quintuplet').write_text(
        'e\tP2\tv\t2005-01-01\t2006-01-01\n')
    (data / 'valid.quintuplet').write_text(
        'e\tP1\tv\t2000-01-01\t2001-01-01\n'
        'e\tP1\tv\t2007-01-01\t2008-01-01\n'
        'unknown\tP1\tv\t2000-01-01\t2001-01-01\n')
    (data / 'valid.gt').write_text('True\nFalse\nTrue\n')
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({
        'after_oracle': {'P1___P2': {'status': 'Observed', 'relations': {
            'before': {'support': [2, 2]}}}},
        'after_propagation': {'P1___P2': {'status': 'Inferred', 'relations': {
            'after': {'support': [0, 0], 'inferred_score': 1}}}},
    }))
    original = rule.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail('La validation ne doit lancer ni apprentissage ni propagation')

    monkeypatch.setattr(fc.qcn_generator2, 'main', forbidden)
    monkeypatch.setattr(fc.qcn_generator2, 'propagateAndFilterConservative', forbidden)
    destination = tmp_path / 'evaluation'
    metrics = fc.validate_without_propagation(
        'Q1', rule, data.parent, destination)
    assert metrics['stage'] == 'after_oracle'
    assert metrics['method'] == 'tcl_without_propagation'
    assert metrics['total'] == 3
    assert metrics['answered'] == metrics['correct'] == 2
    assert metrics['accuracy'] == 1
    assert metrics['coverage'] == pytest.approx(2 / 3)
    assert metrics['abstentions'] == 1
    percentages = metrics['confusion_matrix_percentages']
    assert percentages['TP'] == percentages['TN'] == pytest.approx(100 / 3)
    assert percentages['FP'] == percentages['FN'] == 0
    assert percentages['abstentions'] == pytest.approx(100 / 3)
    for cls in ('positive', 'negative'):
        for measure in ('precision', 'recall', 'f1'):
            assert percentages[f'{measure}_{cls}'] == 100
    assert json.loads((destination / 'metrics.json').read_text()) == metrics
    assert rule.read_bytes() == original
    output = tmp_path / 'batch_output'
    existing = output / 'result_valid' / 'tcl' / 'metrics.json'
    existing.parent.mkdir(parents=True)
    existing.write_text('existing results')
    for minimum in (1, 2):
        source = (output / 'result_train' / 'tcl' / 'Q1' /
                  f'min_total_{minimum}' / 'Q1' / 'conservative' / 'qcn2_Q1.json')
        source.parent.mkdir(parents=True)
        source.write_bytes(original)
    summaries = fc.validate_without_propagation('Q1', data_dir=data.parent, output_dir=output)
    assert len(summaries) == 2
    assert all(item['accuracy'] == 1 for item in summaries)
    target = output / 'result_valid' / 'tcl_without_propagation' / 'Q1'
    assert len(list(target.rglob('predictions.tsv'))) == 2
    assert json.loads((target / 'summary.json').read_text()) == summaries
    assert existing.read_text() == 'existing results'
    rule.write_text(json.dumps({'after_propagation': {}}))
    with pytest.raises(ValueError, match='after_oracle'):
        fc.validate_without_propagation('Q1', rule, data.parent, destination)


def test_unconstrained_and_empty_domains(tmp_path):
    from tclkg.allen_list import ALLEN_RELATIONS
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {'status': 'Observed', 'relations': dict.fromkeys(ALLEN_RELATIONS, {'support': [1, 1]})}}}))
    assert fc.load_constraint_index(rule) == {}
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {'status': 'Observed', 'relations': {}}}}))
    assert fc.load_constraint_index(rule) == {}


def test_same_property_domain_raises(tmp_path):
    rule = tmp_path / 'rules.json'
    p = 'http://www.wikidata.org/prop/P1'
    direct = 'http://www.wikidata.org/prop/direct/P1'
    rule.write_text(json.dumps({'after_propagation': {
        f'{direct}___{p}': {'status': 'Observed', 'relations': {'before': {}}}}}))
    with pytest.raises(ValueError, match='deux fois la même propriété'):
        fc.load_constraint_index(rule)


def test_relation_weight_strict_cutoff(tmp_path):
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {'status': 'Observed', 'relations': {
        'before': {'support': [1, 10]},
        'meets': {'support': [2, 10]},
        'overlaps': {'support': [0, 0], 'inferred_score': 0.11},
        'during': {'support': [0, 0], 'inferred_score': 0.1},
        'equals': {'support': [0, 0]}}}}}))
    assert fc.load_constraint_index(rule)['P1'] == [(('P1', 'P2'), {'meets', 'overlaps'})]


@pytest.mark.parametrize('status', [None, 'Unknown', 'unkown'])
def test_unknown_domain_skips_relation_inspection(tmp_path, status):
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {
        'P1___P2': {'status': status, 'relations': 'not inspected'}}}))
    assert fc.load_constraint_index(rule) == {}


def test_direct_allen_checks_match_original_for_all_small_intervals():
    from itertools import combinations_with_replacement, product
    from types import SimpleNamespace
    from tclkg.allen_list import ALLEN_RELATIONS

    intervals = [SimpleNamespace(date=SimpleNamespace(start=a, end=b))
                 for a, b in combinations_with_replacement(range(4), 2)]
    for left, right in product(intervals, repeat=2):
        matches = dict.fromkeys(ALLEN_RELATIONS, 0)
        fc.qcn_generator2.add_allen_relation_matches(matches, left, right)
        for relation in ALLEN_RELATIONS:
            assert fc.matches_allowed_relations(left, right, {relation}) == bool(matches[relation])
        assert fc.matches_allowed_relations(left, right, {'before', 'meets'}) == bool(matches['before'] or matches['meets'])


def test_pair_stops_after_first_failure(monkeypatch):
    from types import SimpleNamespace

    def fact(property_):
        return SimpleNamespace(head='e', relation=property_, date=SimpleNamespace(start=1, end=2))

    candidate = fact('P1')
    calls = []

    def check(left, right, allowed):
        calls.append(right)
        return False

    monkeypatch.setattr(fc, 'matches_allowed_relations', check)
    result = fc.predict_fact(candidate, {'e': {'P2': [fact('P2'), fact('P2')]}},
                             {'P1': [(('P1', 'P2'), {'before'})]}, 0.9)
    assert result == (1, 0.0, 1)
    assert len(calls) == 1


def test_custom_relation_weight_threshold(tmp_path):
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {
        'status': 'Observed', 'relations': {
            'before': {'support': [3, 10]},
            'meets': {'support': [4, 10]}}}}}))
    assert fc.load_constraint_index(rule, relation_weight_threshold=0.3)['P1'] == [(('P1', 'P2'), {'meets'})]
    assert fc.load_constraint_index(rule, relation_weight_threshold=0)['P1'] == [(('P1', 'P2'), {'before', 'meets'})]
    assert fc.load_constraint_index(rule, relation_weight_threshold=1) == {}


@pytest.mark.parametrize('status', ['Inferred', 'Infered'])
def test_inferred_domain_uses_inferred_weight(tmp_path, status):
    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {
        'status': status, 'relations': {
            'before': {'support': [0, 100], 'inferred_score': 0.8},
            'meets': {'support': [90, 100], 'inferred_score': 0.05}}}}}))
    assert fc.load_constraint_index(rule)['P1'] == [(('P1', 'P2'), {'before'})]


def test_abstention_subcases_are_written_and_counted(tmp_path):
    import csv
    from types import SimpleNamespace

    rule = tmp_path / 'rules.json'
    rule.write_text(json.dumps({'after_propagation': {'P1___P2': {
        'status': 'Observed', 'relations': {'before': {'support': [1, 1]}}}}}))
    facts = [SimpleNamespace(head='e', relation=prop) for prop in ('P3', 'P1')]
    metrics = fc.evaluate_configuration(
        'Q1', rule, tmp_path / 'evaluation', 'valid', 0.9,
        'after_propagation', {'e': {}}, facts, [True, False])
    with (tmp_path / 'evaluation/predictions.tsv').open() as source:
        rows = list(csv.DictReader(source, delimiter='\t'))
    assert [row['prediction'] for row in rows] == ['3', '3']
    assert [row['abstention_subcode'] for row in rows] == ['3a', '3b']
    breakdown = {row['code']: row for row in metrics['abstention_breakdown']}
    assert 3 not in breakdown
    for code in ('3a', '3b'):
        assert breakdown[code]['count'] == 1
        assert breakdown[code]['percent_of_facts'] == 50
        assert breakdown[code]['percent_of_abstentions'] == 50
    assert fc.confusion_matrix_from_predictions(tmp_path / 'evaluation/predictions.tsv')['abstentions'] == 100


def test_legacy_abstentions_keep_unspecified_code_3(tmp_path):
    path = tmp_path / 'predictions.tsv'
    path.write_text('prediction\n3\n')
    breakdown = {row['code']: row for row in fc.abstention_breakdown_from_predictions(path)}
    assert breakdown[3]['count'] == 1
    assert breakdown[3]['percent_of_abstentions'] == 100
    assert '3a' not in breakdown
