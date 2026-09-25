from __future__ import annotations

import numpy as np

from tclkg import ground_truth
from tclkg import qcn_generator2 as qgen
from tclkg.allen_list import ALLEN_RELATIONS
from tclkg.generator_common import compose_allen_scores, converse_domain_scores
from tclkg.time_package import Entity, Interval, Triple


def _triple(head: str, relation: str, start: str, end: str) -> Triple:
    return Triple(
        head,
        relation,
        "value",
        Interval(np.datetime64(start, "D"), np.datetime64(end, "D")),
    )


def _entity(name: str, triples: list[Triple]) -> Entity:
    entity = Entity(name, np.datetime64("2023-12-31", "D"), "D")
    for triple in triples:
        entity.add_triple(triple)
    return entity


def _empty_domain() -> dict[str, object]:
    return qgen.with_relation_data({})


def _observed_domain(
    relation: str, support: tuple[int, int] = (1, 1)
) -> dict[str, object]:
    return qgen.with_relation_data({relation: support})


def test_query_computes_expected_support_ratios() -> None:
    p1 = "http://www.wikidata.org/prop/P1"
    p2 = "http://www.wikidata.org/prop/P2"
    entities = {
        "e1": _entity(
            "e1",
            [
                _triple("e1", p1, "2000-01-01", "2000-01-02"),
                _triple("e1", p2, "2000-01-03", "2000-01-04"),
            ],
        ),
        "e2": _entity(
            "e2",
            [
                _triple("e2", p1, "2001-01-02", "2001-01-03"),
                _triple("e2", p2, "2001-01-01", "2001-01-02"),
            ],
        ),
    }

    result = qgen.Query((p1, p2), entities)
    scores = qgen.relation_scores(result)

    assert result["relations"]["before"]["support"] == (1, 2)
    assert result["relations"]["met_by"]["support"] == (1, 2)
    assert result["relations"]["after"]["support"] == (0, 2)
    assert scores["before"] == 0.5
    assert scores["met_by"] == 0.5
    assert scores["after"] == 0.0
    assert qgen.domain_status(result) == qgen.STATUS_OBSERVED


def test_query_without_common_entities_returns_universal_unknown_domain() -> None:
    p1 = "http://www.wikidata.org/prop/P1"
    p2 = "http://www.wikidata.org/prop/P2"
    entities = {
        "e1": _entity("e1", [_triple("e1", p1, "2000-01-01", "2000-01-02")]),
        "e2": _entity("e2", [_triple("e2", p2, "2000-01-03", "2000-01-04")]),
    }

    result = qgen.Query((p1, p2), entities)

    assert qgen.domain_status(result) is None
    assert set(result["relations"]) == set(ALLEN_RELATIONS)
    assert all(
        payload["support"] == (0, 0)
        for payload in result["relations"].values()
    )


def test_converse_and_composition_preserve_expected_relations() -> None:
    domain = _observed_domain("before", (3, 4))
    converse = converse_domain_scores(domain)
    composition = compose_allen_scores({"before": 0.75}, {"meets": 0.5})

    assert qgen.relation_scores(converse) == {"after": 0.75}
    assert qgen.domain_status(converse) == qgen.STATUS_OBSERVED
    assert composition == {"before": 0.5}


def test_propagate_and_filter_infers_missing_composition() -> None:
    a = "http://www.wikidata.org/prop/PA"
    b = "http://www.wikidata.org/prop/PB"
    c = "http://www.wikidata.org/prop/PC"
    qcn = {
        (a, b): _observed_domain("before", (3, 4)),
        (b, a): _observed_domain("after", (3, 4)),
        (b, c): _observed_domain("before", (1, 2)),
        (c, b): _observed_domain("after", (1, 2)),
        (a, c): _empty_domain(),
        (c, a): _empty_domain(),
    }

    result = qgen.propagateAndFilterConservative(qcn)

    assert result[(a, c)]["relations"]["before"]["support"] == (0, 0)
    assert result[(a, c)]["relations"]["before"]["inferred_score"] == 0.5
    assert set(result[(a, c)]["relations"]) == {"before"}
    assert qgen.domain_status(result[(a, c)]) == qgen.STATUS_INFERRED
    assert result[(c, a)]["relations"]["after"]["support"] == (0, 0)
    assert result[(c, a)]["relations"]["after"]["inferred_score"] == 0.5
    assert set(result[(c, a)]["relations"]) == {"after"}
    assert qgen.domain_status(result[(c, a)]) == qgen.STATUS_INFERRED

    inferred_stats = qgen.compute_qcn_relation_stats({(a, c): result[(a, c)]})
    assert inferred_stats["count_inferred"] == 1
    assert inferred_stats["count_none_none"] == 0
    assert inferred_stats["min_inferred_score"] == 0.5
    assert inferred_stats["avg_inferred_score"] == 0.5
    assert inferred_stats["max_inferred_score"] == 0.5


def test_ground_truth_semantic_stats_and_transition_stats() -> None:
    after_oracle = {
        "a___b": _observed_domain("before"),
        "a___c": _empty_domain(),
    }
    after_propagation = {
        "a___b": _observed_domain("before"),
        "a___c": {
            "status": qgen.STATUS_INFERRED,
            "relations": {"before": {"support": (0, 0)}},
        },
    }

    stats = ground_truth.semantic_stats(after_oracle)
    transitions = ground_truth.transition_stats(after_oracle, after_propagation)

    assert stats["domains"] == 2
    assert stats["relations"] == 1
    assert stats["status_counts"] == {"Observed": 1}
    assert stats["support_counts"] == {
        "negative": 0,
        "none": 0,
        "positive": 1,
        "zero": 0,
    }
    assert transitions["none_after_oracle_total"] == 0
    assert transitions["none_to_inferred"] == 0
    assert transitions["domains_without_positive_to_positive"] == 0
