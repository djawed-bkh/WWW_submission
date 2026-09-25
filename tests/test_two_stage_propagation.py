import copy
import unittest

from tclkg import qcn_generator2 as qgen


class TwoStagePropagationTests(unittest.TestCase):
    def chain(self):
        properties = "abcdefz"
        qcn = {
            (a, b): qgen.with_relation_data({r: (0, 0) for r in qgen.ALLEN_RELATIONS})
            for a in properties for b in properties if a != b
        }
        for a, b in zip("abcde", "bcdef"):
            qcn[a, b] = qgen.with_relation_data({"before": (8, 10)})
            qcn[b, a] = qgen.fast_converse_domain(qcn[a, b])
        return qcn

    def test_two_stages_preserve_first_stage_and_use_both_source_types(self):
        for reduced in (False, True):
            with self.subTest(reduced=reduced):
                qcn = self.chain()
                original = copy.deepcopy(qcn)
                first = qgen.propagateAndFilterConservative(qcn, 10, 0.7, reduced)
                second = qgen.propagateAndFilterTwoStage(qcn, 10, 0.7, reduced)
                self.assertEqual(qcn, original)
                for pair, domain in first.items():
                    self.assertNotIn("inference_stage", domain)
                    if domain["status"] == "Inferred":
                        self.assertEqual(second[pair], {**domain, "inference_stage": 1})
                    elif domain["status"] is not None:
                        self.assertEqual(second[pair], domain)
                    if second[pair]["status"] != "Inferred":
                        self.assertNotIn("inference_stage", second[pair])
                # Three edges require observed + inferred, four inferred + inferred.
                for target in ("d", "e"):
                    self.assertIsNone(first["a", target]["status"])
                    self.assertEqual(second["a", target], {
                        "status": "Inferred",
                        "inference_stage": 2,
                        "relations": {"before": {"support": (0, 0), "inferred_score": 0.8}},
                    })
                    self.assertEqual(second[target, "a"], {
                        **qgen.fast_converse_domain(second["a", target]), "inference_stage": 2,
                    })
                # No recursive reuse of second-stage results; disconnected stays unknown.
                self.assertIsNone(second["a", "f"]["status"])
                self.assertIsNone(second["a", "z"]["status"])
                reverse_order = qgen.propagateAndFilterTwoStage(
                    dict(reversed(list(qcn.items()))), 10, 0.7, reduced
                )
                self.assertEqual(second, reverse_order)

    def test_observed_evidence_thresholds_still_apply(self):
        for reliability, occurrence in ((11, 0.7), (10, 0.9)):
            result = qgen.propagateAndFilterTwoStage(
                self.chain(), reliability, occurrence
            )
            self.assertIsNone(result["a", "c"]["status"])
            self.assertIsNone(result["a", "d"]["status"])

    def test_strass_freezes_domains_and_advances_in_stages(self):
        qcn = self.chain()
        original = copy.deepcopy(qcn)
        for reduced in (False, True):
            first = qgen.propagateAndFilterStrass(qcn, 10, 0.7, reduced, max_stages=1)
            second = qgen.propagateAndFilterStrass(qcn, 10, 0.7, reduced, max_stages=2)
            final = qgen.propagateAndFilterStrass(qcn, 10, 0.7, reduced)
            self.assertEqual(first['a', 'c']['inference_stage'], 1)
            self.assertIsNone(first['a', 'd']['status'])
            self.assertEqual(second, qgen.propagateAndFilterTwoStage(qcn, 10, 0.7, reduced))
            self.assertEqual(second['a', 'e']['inference_stage'], 2)
            self.assertIsNone(second['a', 'f']['status'])
            self.assertEqual(final['a', 'f']['inference_stage'], 3)
            self.assertEqual(final['f', 'a']['inference_stage'], 3)
            self.assertIsNone(final['a', 'z']['status'])
            for previous in (first, second):
                for pair, domain in previous.items():
                    if domain['status'] is not None:
                        self.assertEqual(final[pair], domain)
            self.assertEqual(final, qgen.propagateAndFilterStrass(
                dict(reversed(list(qcn.items()))), 10, 0.7, reduced,
            ))
        self.assertEqual(qcn, original)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                qgen.propagateAndFilterStrass(qcn, max_stages=invalid)

    def test_iterative_reuses_inferences_beyond_two_stages(self):
        qcn = self.chain()
        original = copy.deepcopy(qcn)
        for reduced in (False, True):
            result = qgen.propagateAndFilterIterative(qcn, 10, 0.7, reduced)
            self.assertTrue(all("inference_stage" not in domain for domain in result.values()))
            self.assertEqual(result["a", "f"]["status"], "Inferred")
            self.assertEqual(result["a", "f"]["relations"]["before"]["inferred_score"], 0.8)
            self.assertIsNone(result["a", "z"]["status"])
            for pair, domain in qcn.items():
                if domain["status"] == "Observed":
                    self.assertEqual(result[pair], domain)
                self.assertEqual(result[pair[1], pair[0]], qgen.fast_converse_domain(result[pair]))
            self.assertEqual(result, qgen.propagateAndFilterIterative(
                dict(reversed(list(qcn.items()))), 10, 0.7, reduced,
            ))
        self.assertEqual(qcn, original)
        blocked = qgen.propagateAndFilterIterative(qcn, 11, 0.7)
        self.assertIsNone(blocked["a", "f"]["status"])

    def test_iterative_target_receives_multiple_paths(self):
        qcn = {
            (a, b): qgen.with_relation_data({})
            for a in "abcd" for b in "abcd" if a != b
        }
        for pair, support in ((('a', 'b'), (4, 10)), (('b', 'd'), (4, 10)),
                              (('a', 'c'), (9, 10)), (('c', 'd'), (9, 10))):
            qcn[pair] = qgen.with_relation_data({"before": support})
            qcn[pair[1], pair[0]] = qgen.fast_converse_domain(qcn[pair])
        result = qgen.propagateAndFilterIterative(qcn, reduced_composition=True)
        self.assertEqual(result['a', 'd']['relations']['before']['inferred_score'], 0.9)

    def test_iterative_replays_path_after_source_score_improves(self):
        # Empty observed domains block alternative paths to a -> d.
        qcn = {
            (a, b): {"status": "Observed", "relations": {}}
            for a in "abcd" for b in "abcd" if a != b
        }
        for pair in (("a", "c"), ("c", "b"), ("b", "d")):
            qcn[pair] = qgen.with_relation_data({"before": (9, 10)})
            qcn[pair[1], pair[0]] = qgen.fast_converse_domain(qcn[pair])
        qcn['a', 'b'] = {
            "status": "Inferred",
            "relations": {"before": {"support": (0, 0), "inferred_score": 0.2}},
        }
        qcn['b', 'a'] = qgen.fast_converse_domain(qcn['a', 'b'])
        qcn['a', 'd'] = qgen.with_relation_data({})
        qcn['d', 'a'] = qgen.with_relation_data({})
        # a -> b -> d first yields 0.2, then must run again after a -> c -> b.
        result = qgen.propagateAndFilterIterative(qcn)
        self.assertEqual(result['a', 'b']['relations']['before']['inferred_score'], 0.9)
        self.assertEqual(result['a', 'd']['relations']['before']['inferred_score'], 0.9)
        self.assertEqual(result['d', 'a'], qgen.fast_converse_domain(result['a', 'd']))
        self.assertEqual(qgen.propagateAndFilterIterative(result), result)



if __name__ == "__main__":
    unittest.main()
