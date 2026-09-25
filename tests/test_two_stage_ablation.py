import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tclkg import random_experiments as rexp
from tclkg import qcn_generator2 as qgen


class TwoStageAblationTests(unittest.TestCase):
    def test_metrics_include_failures_and_separate_stage_populations(self):
        observed = {key: {'relations': {'before': {'support': [1, 1]}}}
                    for key in 'abcd'}
        def prediction(stage, relations):
            return {'status': 'Inferred', 'inference_stage': stage,
                    'relations': {r: {'inferred_score': v} for r, v in relations.items()}}
        predictions = {'a': prediction(1, {'before': 1}),
                       'b': prediction(2, {'before': 0.5, 'after': 1}),
                       'c': prediction(2, {'after': 1}), 'd': {'status': None}}
        result = rexp.evaluate_two_stage_inference(predictions, observed, 0, 0)
        self.assertEqual(result, dict(n_masked=4, n_stage_1=1, n_stage_2=2,
                                     n_uninferred=1, mrr_global=0.5,
                                     mrr_stage_1=1, mrr_stage_2=0.25))
        empty = rexp.evaluate_two_stage_inference({}, observed, 0, 0)
        self.assertIsNone(empty['mrr_global'])
        self.assertIsNone(empty['mrr_stage_1'])
        self.assertIsNone(empty['mrr_stage_2'])

    def test_mrr_ignores_missing_predictions_but_counts_wrong_predictions(self):
        observed = {key: {'relations': {'before': {'support': [1, 1]}}}
                    for key in 'abcd'}
        predicted = {
            'a': {'relations': {'before': {'inferred_score': 1.0}}},
            'b': {'relations': {'after': {'inferred_score': 1.0}}},
            'c': {'relations': {'before': {'support': [0, 0]}}},
            'd': {'relations': {'before': {'inferred_score': float('nan')}}},
        }
        self.assertEqual(rexp.weighted_mean_reciprocal_rank_evaluation(
            predicted, observed, 0, 0), 0.5)

    def test_rounds_restore_evidence_mask_converses_and_save_reports(self):
        qcn = {(a, b): qgen.with_relation_data({'before' if a < b else 'after': (1, 1)})
               for a in 'abcdef' for b in 'abcdef' if a != b}
        seen = set()
        def propagate(masked, total, occurrence, reduced):
            hidden = [pair for pair, domain in masked.items() if domain['status'] is None]
            self.assertEqual(len(hidden), 2)
            self.assertEqual(hidden[0], hidden[1][::-1])
            canonical = tuple(sorted(hidden[0]))
            self.assertNotIn(canonical, seen)
            seen.add(canonical)
            for pair, domain in masked.items():
                if pair not in hidden:
                    self.assertEqual(domain, json.loads(json.dumps(qcn[pair])))
                else:
                    self.assertTrue(all(p['support'] == (0, 0) for p in domain['relations'].values()))
            result = copy.deepcopy(masked)
            for pair in hidden:
                result[pair] = {'status': 'Inferred', 'inference_stage': 1 if len(seen) <= 3 else 2,
                                'relations': {'before' if pair[0] < pair[1] else 'after':
                                              {'support': (0, 0), 'inferred_score': 1}}}
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'occ_0p3_min_1_timeout_600'
            folder.mkdir()
            path = folder / 'qcn2_Q6256_reduced_composition.json'
            path.write_text(json.dumps({'after_oracle': {'___'.join(k): v for k, v in qcn.items()}}))
            with patch.object(qgen, 'propagateAndFilterTwoStage', side_effect=propagate), patch.object(rexp, 'results_root', root):
                results = rexp.run_two_stage_ablation_studies(n=1)
            summary = next(iter(results.values()))
            self.assertEqual(summary['n_masked'], 10)
            self.assertEqual(summary['n_stage_1'], 3)
            self.assertEqual(summary['n_stage_2'], 7)
            self.assertEqual(summary['mrr_global'], 1)
            self.assertEqual(len(summary['rounds']), 10)
            self.assertTrue((root / 'two_stage_ablation.csv').is_file())
            saved = json.loads((root / 'two_stage_ablation.json').read_text())
            self.assertEqual(saved, results)
