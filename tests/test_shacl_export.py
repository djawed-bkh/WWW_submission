from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import combinations_with_replacement, product
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdflib import RDF, SH, XSD, Graph, Literal, Namespace, URIRef
from rdflib.query import ResultRow

from tclkg import generator_common, qcn_generator2
from tclkg.allen_list import ALLEN_RELATIONS
from tclkg.qcn_generator2 import add_allen_relation_matches
from tclkg.shacl_export import (
    TCL,
    TemporalConstraint,
    build_shapes,
    load_constraints,
    load_data,
    load_quintuplets,
    main,
    validate_graph,
)

EX = Namespace("https://example.org/test#")


def _write_network(
    path: Path, relations: Iterable[str], stage: str = "after_propagation"
) -> Path:
    path.write_text(
        json.dumps(
            {
                stage: {
                    f"{EX.A}___{EX.B}": {
                        "status": "Inferred",
                        "relations": {
                            r: {"support": [0, 0], "inferred_score": 0.5}
                            for r in relations
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _fact(graph: Graph, subject: URIRef, prop: URIRef, start: int, end: int) -> URIRef:
    node = URIRef(f"urn:test:fact:{len(graph)}")
    for predicate, value in (
        (RDF.type, TCL.TemporalAssertion),
        (RDF.subject, subject),
        (RDF.predicate, prop),
        (RDF.object, EX.v),
        (
            TCL.start,
            Literal(f"2000-01-{start + 1:02d}T00:00:00Z", datatype=XSD.dateTime),
        ),
        (TCL.end, Literal(f"2000-01-{end + 1:02d}T00:00:00Z", datatype=XSD.dateTime)),
    ):
        graph.add((node, predicate, value))
    return node


def _shapes(relations: Iterable[str] = ("after",)) -> Graph:
    return build_shapes(
        [TemporalConstraint(str(EX.A), str(EX.B), frozenset(relations))], str(EX.C)
    )


def _violations(graph: Graph, query: str) -> set[URIRef]:
    subjects = set()
    for row in graph.query(query):
        assert isinstance(row, ResultRow)
        subject = row["this"]
        assert isinstance(subject, URIRef)
        subjects.add(subject)
    return subjects


def test_all_endpoint_filters_match_tcl_including_points() -> None:
    intervals = list(combinations_with_replacement(range(5), 2))
    graph = Graph()
    expected = {r: set() for r in ALLEN_RELATIONS}
    for index, ((a, b), (c, d)) in enumerate(product(intervals, repeat=2)):
        subject = EX[f"I{index}"]
        graph.add((subject, RDF.type, EX.C))
        _fact(graph, subject, EX.A, a, b)
        _fact(graph, subject, EX.B, c, d)
        counts: dict[str, float] = dict.fromkeys(ALLEN_RELATIONS, 0.0)
        add_allen_relation_matches(
            counts,
            SimpleNamespace(date=SimpleNamespace(start=a, end=b)),
            SimpleNamespace(date=SimpleNamespace(start=c, end=d)),
        )
        for relation, count in counts.items():
            if not count:
                expected[relation].add(subject)
    for relation in sorted(ALLEN_RELATIONS):
        query = next(_shapes([relation]).objects(None, SH.select))
        violations = _violations(graph, str(query))
        assert violations == expected[relation], relation
    query = next(_shapes(["after", "meets"]).objects(None, SH.select))
    assert _violations(graph, str(query)) == expected["after"] & expected["meets"]


@pytest.mark.parametrize(
    ("a_values", "b_values", "allowed", "conforms"),
    [
        ([(4, 5)], [(1, 2)], ["after"], True),
        ([(1, 3)], [(2, 4)], ["after"], False),
        ([(3, 4)], [(1, 3)], ["after"], False),
        ([(3, 4)], [(1, 3)], ["after", "met_by"], True),
        ([], [(1, 2)], ["after"], True),
        ([(1, 2)], [], ["after"], True),
        ([], [], ["after"], True),
        ([(3, 4), (5, 6)], [(0, 1), (1, 2)], ["after"], True),
        ([(3, 4), (5, 6)], [(0, 1), (4, 6)], ["after"], False),
        ([(2, 2)], [(2, 2)], ["equals"], True),
        ([(2, 2)], [(2, 2)], ["met_by"], True),
    ],
)
def test_shacl_validation_uses_every_pair(
    a_values: list[tuple[int, int]],
    b_values: list[tuple[int, int]],
    allowed: list[str],
    conforms: bool,
) -> None:
    graph = Graph()
    graph.add((EX.I, RDF.type, EX.C))
    for prop, values in ((EX.A, a_values), (EX.B, b_values)):
        for start, end in values:
            _fact(graph, EX.I, prop, start, end)
    assert validate_graph(graph, _shapes(allowed))[0] is conforms


def test_shapes_are_valid_shacl_and_target_only_the_class() -> None:
    from pyshacl import validate

    data = Graph()
    data.add((EX.I, RDF.type, EX.Other))
    _fact(data, EX.I, EX.A, 0, 1)
    _fact(data, EX.I, EX.B, 2, 3)
    assert validate(data, shacl_graph=_shapes(), meta_shacl=True)[0]


def test_quintuplet_normalization_matches_loader(tmp_path: Path) -> None:
    path = tmp_path / "facts.quintuplet"
    prop = "http://www.wikidata.org/prop/direct/P1"
    path.write_text(
        f"e1\t{prop}\tv\t2005-01-01\t2000-01-01\n"
        f"e1\t{prop}\tv\t2000-01-01\t2005-01-01\n"
        f"e1\t{EX.B}\tv\tNone\t2006-01-01\n",
        encoding="utf-8",
    )
    graph = load_quintuplets(path, str(EX.C))
    assertions = set(graph.subjects(RDF.type, TCL.TemporalAssertion))
    assert len(assertions) == 2
    assert (URIRef("urn:tcl:entity:e1"), RDF.type, EX.C) in graph
    a = next(graph.subjects(RDF.predicate, URIRef("http://www.wikidata.org/prop/P1")))
    assert str(graph.value(a, TCL.start)).startswith("2000-01-01")
    assert str(graph.value(a, TCL.end)).startswith("2005-01-01")
    b = next(graph.subjects(RDF.predicate, EX.B))
    assert graph.value(b, TCL.start) == graph.value(b, TCL.end)


def test_missing_both_endpoints_fails_structural_validation(tmp_path: Path) -> None:
    path = tmp_path / "facts.quintuplet"
    path.write_text(f"{EX.I}\t{EX.A}\tv\tNone\tNone\n", encoding="utf-8")
    conforms, report = validate_graph(load_data(path, str(EX.C)), _shapes())
    assert not conforms
    assert (
        None,
        SH.sourceConstraintComponent,
        SH.MinCountConstraintComponent,
    ) in report


def test_rdf_input_preserves_types_and_normalizes_predicates(tmp_path: Path) -> None:
    graph = Graph()
    graph.add((EX.I, RDF.type, EX.C))
    _fact(graph, EX.I, URIRef("http://www.wikidata.org/prop/direct/P1"), 2, 3)
    path = tmp_path / "facts.ttl"
    graph.serialize(destination=str(path), format="turtle")
    loaded = load_data(path, str(EX.Other))
    assert (EX.I, RDF.type, EX.C) in loaded
    assert (EX.I, RDF.type, EX.Other) not in loaded
    assert (None, RDF.predicate, URIRef("http://www.wikidata.org/prop/P1")) in loaded


def test_rdf_normalization_preserves_invalid_multiple_predicates(
    tmp_path: Path,
) -> None:
    graph = Graph()
    graph.add((EX.I, RDF.type, EX.C))
    assertion = _fact(graph, EX.I, EX.A, 2, 3)
    graph.add((assertion, RDF.predicate, EX.B))
    path = tmp_path / "invalid.ttl"
    graph.serialize(destination=str(path), format="turtle")
    data = load_data(path, str(EX.C))
    assert set(data.objects(assertion, RDF.predicate)) == {EX.A, EX.B}
    conforms, report = validate_graph(data, _shapes())
    assert not conforms
    assert (
        None,
        SH.sourceConstraintComponent,
        SH.MaxCountConstraintComponent,
    ) in report


def test_real_learner_output_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "data" / "Mini"
    cohort.mkdir(parents=True)
    data = cohort / "data.quintuplet"
    data.write_text(
        f"e1\t{EX.A}\tv\t2000-01-01\t2000-01-02\n"
        f"e1\t{EX.B}\tv\t2000-01-03\t2000-01-04\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generator_common, "ROOT_DATA", tmp_path / "data")
    monkeypatch.setattr(qcn_generator2, "results_root", tmp_path / "results")
    entities, properties = qcn_generator2.read_KG("Mini", False)
    network = qcn_generator2.build_entity_network(
        properties=properties,
        default_constraints=ALLEN_RELATIONS,
        top_k_fraction=1.0,
        entities=entities,
    )
    result = qcn_generator2.learner2(
        network, entities, dict.fromkeys(sorted(properties))
    )
    assert result is not None
    qcn_generator2.save_qcn_to_file(result, "Mini")
    path = tmp_path / "results/Mini/conservative/qcn2_Mini_reduced_composition.json"
    assert main([str(path), "--class-iri", str(EX.C), "--data", str(data)]) == 0


def test_inferred_zero_support_is_preserved_and_stage_is_explicit(
    tmp_path: Path,
) -> None:
    path = _write_network(tmp_path / "qcn.json", ["after", "met_by"])
    constraints = load_constraints(path)
    assert constraints[0].relations == frozenset({"after", "met_by"})
    with pytest.raises(ValueError, match="does not contain stage"):
        load_constraints(path, "after_oracle")
    raw = json.loads(path.read_text())["after_propagation"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_constraints(path) == constraints


def test_unknown_and_universal_domains_add_no_temporal_restriction() -> None:
    for relations in (frozenset(), frozenset(ALLEN_RELATIONS)):
        graph = _shapes(relations)
        assert not list(graph.objects(None, SH.sparql))
        assert (TCL.TemporalAssertionShape, RDF.type, SH.NodeShape) in graph


def test_only_equivalent_converse_arcs_are_deduplicated() -> None:
    constraints = [
        TemporalConstraint(str(EX.A), str(EX.B), frozenset({"after"})),
        TemporalConstraint(str(EX.B), str(EX.A), frozenset({"before"})),
    ]
    assert len(list(build_shapes(constraints, str(EX.C)).objects(None, SH.sparql))) == 1
    constraints[1] = TemporalConstraint(str(EX.B), str(EX.A), frozenset({"meets"}))
    assert len(list(build_shapes(constraints, str(EX.C)).objects(None, SH.sparql))) == 2


@pytest.mark.parametrize("relations", [["unknown"], ["after"]])
def test_invalid_networks_are_rejected(tmp_path: Path, relations: list[str]) -> None:
    path = _write_network(tmp_path / "qcn.json", relations)
    if relations == ["after"]:
        data = json.loads(path.read_text())
        data["after_propagation"][f"{EX.A}___{EX.B}"]["relations"]["after"][
            "support"
        ] = [2, 1]
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_constraints(path)


@pytest.mark.parametrize("bad_iri", ["relative", "https://example.org/> ?s ?p ?o . <x"])
def test_class_iri_cannot_inject_sparql(bad_iri: str) -> None:
    with pytest.raises(ValueError):
        build_shapes([], bad_iri)


def test_cli_exports_then_optionally_validates(tmp_path: Path) -> None:
    network = _write_network(tmp_path / "qcn.json", ["after"])
    output = tmp_path / "shapes.ttl"
    arguments = [str(network), "--class-iri", str(EX.C), "--output", str(output)]
    assert main(arguments) == 0
    assert list(Graph().parse(output).objects(None, SH.sparql))
    data = tmp_path / "data.quintuplet"
    data.write_text(
        f"{EX.I}\t{EX.A}\tv\t2000-01-01\t2000-01-02\n{EX.I}\t{EX.B}\tv\t2000-01-03\t2000-01-04\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.ttl"
    assert main([*arguments, "--data", str(data), "--report", str(report)]) == 1
    graph = Graph().parse(report)
    assert (None, SH.conforms, Literal(False)) in graph
    data.write_text(
        f"{EX.I}\t{EX.A}\tv\t2001-01-01\t2001-01-02\n{EX.I}\t{EX.B}\tv\t2000-01-03\t2000-01-04\n",
        encoding="utf-8",
    )
    assert main([*arguments, "--data", str(data)]) == 0


def test_cli_errors_do_not_overwrite_input(tmp_path: Path) -> None:
    network = _write_network(tmp_path / "qcn.json", ["after"])
    original = network.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(network), "--class-iri", str(EX.C), "--output", str(network)])
    assert exc.value.code == 2
    assert network.read_bytes() == original
    assert main([str(network), "--class-iri", "relative"]) == 2


def test_bad_quintuplets_report_the_line_number(tmp_path: Path) -> None:
    path = tmp_path / "data.quintuplet"
    path.write_text("not\ta\tquintuplet\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1:"):
        load_quintuplets(path, str(EX.C))
