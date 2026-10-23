"""Compile TCL JSON networks into SHACL and optionally validate temporal facts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlsplit

import numpy as np
from rdflib import RDF, SH, XSD, BNode, Graph, Literal, Namespace, URIRef
from rdflib.exceptions import ParserError
from rdflib.plugin import PluginException

from .allen_list import ALLEN_CONVERSE, ALLEN_RELATIONS
from .rule_discovery import processing_date_unknown_allowed, to_common_uri
from .time_package import Interval

TCL = Namespace("https://example.org/tcl#")
STAGES = ("initial", "after_oracle", "after_propagation")
ENDPOINT_EXPRESSIONS: Mapping[str, str] = {
    "before": "?aEnd < ?bStart",
    "after": "?bEnd < ?aStart",
    "meets": "?aEnd = ?bStart",
    "met_by": "?bEnd = ?aStart",
    "overlaps": "?aStart < ?bStart && ?bStart < ?aEnd && ?aEnd < ?bEnd",
    "overlapped_by": "?bStart < ?aStart && ?aStart < ?bEnd && ?bEnd < ?aEnd",
    "starts": "?aStart = ?bStart && ?aEnd < ?bEnd",
    "started_by": "?aStart = ?bStart && ?bEnd < ?aEnd",
    "during": "?bStart < ?aStart && ?aEnd < ?bEnd",
    "contains": "?aStart < ?bStart && ?bEnd < ?aEnd",
    "finishes": "?bStart < ?aStart && ?aEnd = ?bEnd",
    "finished_by": "?aStart < ?bStart && ?aEnd = ?bEnd",
    "equals": "?aStart = ?bStart && ?aEnd = ?bEnd",
}


@dataclass(frozen=True)
class TemporalConstraint:
    """An ordered property pair and its allowed endpoint relations."""

    property_a: str
    property_b: str
    relations: frozenset[str]


def _iri(value: str) -> URIRef:
    if any(char.isspace() or ord(char) < 32 or char in '<>"{}|^`\\' for char in value):
        raise ValueError(f"Invalid IRI: {value!r}")
    if not urlsplit(value).scheme:
        raise ValueError(f"Expected an absolute IRI: {value!r}")
    result = URIRef(value)
    result.n3()  # Reject characters that cannot safely occur in SPARQL IRIs.
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def load_constraints(
    path: Path, stage: str = "after_propagation"
) -> list[TemporalConstraint]:
    """Read a saved learner result or a single serialized network.

    Relation membership, including inferred relations with zero observations,
    defines the domain. Scores are not used as validation thresholds.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown network stage: {stage}")
    with path.open(encoding="utf-8") as handle:
        document = _mapping(json.load(handle), "TCL output")
    if any(key in document for key in STAGES):
        if stage not in document:
            raise ValueError(f"TCL output does not contain stage {stage!r}")
        network = _mapping(document[stage], f"Stage {stage!r}")
    else:
        network = document
    domains: dict[tuple[str, str], frozenset[str]] = {}
    for key, value in network.items():
        parts = key.split("___")
        if len(parts) != 2:
            raise ValueError(f"Invalid TCL pair key: {key!r}; expected A___B")
        pair = tuple(str(_iri(to_common_uri(part))) for part in parts)
        domain = _mapping(value, f"Domain {key!r}")
        entries = _mapping(domain.get("relations"), f"Relations of {key!r}")
        unknown = set(entries) - ALLEN_RELATIONS
        if unknown:
            raise ValueError(f"Unknown Allen relations in {key!r}: {sorted(unknown)}")
        for relation, payload in entries.items():
            evidence = _mapping(payload, f"Payload for {relation!r} in {key!r}")
            support = evidence.get("support")
            if (
                not isinstance(support, list)
                or len(support) != 2
                or any(type(n) is not int for n in support)
                or not 0 <= support[0] <= support[1]
            ):
                raise ValueError(f"Invalid support for {relation!r} in {key!r}")
        relations = frozenset(entries)
        normalized_pair = (pair[0], pair[1])
        if normalized_pair in domains and domains[normalized_pair] != relations:
            raise ValueError(f"Conflicting normalized domains for {normalized_pair}")
        domains[normalized_pair] = relations
    return [
        TemporalConstraint(a, b, relations)
        for (a, b), relations in sorted(domains.items())
    ]


def _structural_shape(graph: Graph) -> None:
    shape = TCL.TemporalAssertionShape
    graph.add((shape, RDF.type, SH.NodeShape))
    graph.add((shape, SH.targetClass, TCL.TemporalAssertion))
    for name, path in (
        ("subject", RDF.subject),
        ("predicate", RDF.predicate),
        ("object", RDF.object),
        ("start", TCL.start),
        ("end", TCL.end),
    ):
        field = BNode(f"temporal-{name}")
        graph.add((shape, SH.property, field))
        graph.add((field, SH.path, path))
        graph.add((field, SH.minCount, Literal(1)))
        graph.add((field, SH.maxCount, Literal(1)))
        if name == "subject":
            graph.add((field, SH.nodeKind, SH.BlankNodeOrIRI))
        elif name == "predicate":
            graph.add((field, SH.nodeKind, SH.IRI))
        elif name in {"start", "end"}:
            graph.add((field, SH.datatype, XSD.dateTime))
            graph.add((field, SH.pattern, Literal("(Z|[+]00:00)$")))
            if name == "start":
                graph.add((field, SH.lessThanOrEquals, TCL.end))


def build_shapes(constraints: Iterable[TemporalConstraint], class_iri: str) -> Graph:
    """Build structural and all-pairs temporal shapes for the requested class.

    Empty TCL domains mean unresolved, not forbidden. Full domains add no
    temporal restriction. Equivalent converse arcs are emitted only once.
    """
    target = _iri(class_iri)
    graph = Graph(bind_namespaces="none")
    for prefix, namespace in (("tcl", TCL), ("sh", SH), ("rdf", RDF), ("xsd", XSD)):
        graph.bind(prefix, namespace)
    _structural_shape(graph)
    domains: dict[tuple[str, str], frozenset[str]] = {}
    for constraint in constraints:
        a, b = str(_iri(constraint.property_a)), str(_iri(constraint.property_b))
        if constraint.relations - ALLEN_RELATIONS:
            raise ValueError(f"Unknown Allen relations for {(a, b)}")
        if (a, b) in domains and domains[a, b] != constraint.relations:
            raise ValueError(f"Conflicting domains for {(a, b)}")
        domains[a, b] = constraint.relations
    for (a, b), relations in sorted(domains.items()):
        if not relations or relations == ALLEN_RELATIONS:
            continue
        converse = frozenset(ALLEN_CONVERSE[r] for r in relations)
        if a > b and domains.get((b, a)) == converse:
            continue
        digest = sha256(json.dumps([class_iri, a, b]).encode()).hexdigest()
        shape = URIRef(f"urn:tcl:shape:{digest}")
        query_node = BNode(f"query-{digest}")
        expression = " || ".join(
            f"({ENDPOINT_EXPRESSIONS[r]})" for r in sorted(relations)
        )
        query = f"""PREFIX rdf: <{RDF}>
PREFIX tcl: <{TCL}>
SELECT DISTINCT $this ?value ?other
WHERE {{
  ?value a tcl:TemporalAssertion ;
    rdf:subject $this ; rdf:predicate {_iri(a).n3()} ;
    tcl:start ?aStart ; tcl:end ?aEnd .
  ?other a tcl:TemporalAssertion ;
    rdf:subject $this ; rdf:predicate {_iri(b).n3()} ;
    tcl:start ?bStart ; tcl:end ?bEnd .
  FILTER (!({expression}))
}}"""
        graph.add((shape, RDF.type, SH.NodeShape))
        graph.add((shape, SH.targetClass, target))
        graph.add((shape, SH.sparql, query_node))
        graph.add((query_node, RDF.type, SH.SPARQLConstraint))
        graph.add((query_node, SH.select, Literal(query)))
        graph.add(
            (
                query_node,
                SH.message,
                Literal(f"{a} / {b}: expected {', '.join(sorted(relations))}"),
            )
        )
        for relation in sorted(relations):
            graph.add((shape, TCL.allowedRelation, Literal(relation)))
    return graph


def _endpoint(raw: str) -> np.datetime64 | None:
    parsed = processing_date_unknown_allowed(raw)
    if parsed is not None and np.isnat(parsed):
        raise ValueError(f"Invalid temporal endpoint: {raw!r}")
    return parsed


def _date_literal(value: np.datetime64 | None) -> Literal | None:
    if value is None:
        return None
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(
            f"pySHACL dateTime validation requires years 0001..9999: {value}"
        ) from exc
    return Literal(
        f"{parsed.isoformat()}T00:00:00Z", datatype=XSD.dateTime, normalize=False
    )


def load_quintuplets(path: Path, class_iri: str) -> Graph:
    """Encode a class-specific TCL input split as temporal assertion records.

    Apply the loader's day granularity, missing-bound completion, endpoint
    ordering, predicate normalization, and deduplication of identical facts.
    """
    target = _iri(class_iri)
    graph = Graph()
    graph.bind("tcl", TCL)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    raise ValueError("Expected five tab-separated fields")
                head, prop, value, raw_start, raw_end = fields
                if not head:
                    raise ValueError("Subject must not be empty")
                predicate = _iri(to_common_uri(prop))
                interval = Interval(_endpoint(raw_start), _endpoint(raw_end))
                start, end = _date_literal(interval.start), _date_literal(interval.end)
                if head.startswith("_:"):
                    if not head[2:]:
                        raise ValueError("Blank-node subject must have an identifier")
                    subject = BNode(head[2:])
                elif urlsplit(head).scheme:
                    subject = _iri(head)
                else:
                    subject = URIRef(f"urn:tcl:entity:{quote(head, safe='')}")
                obj = (
                    _iri(value)
                    if value.startswith(("http://", "https://"))
                    else Literal(value)
                )
                identity = json.dumps(
                    [
                        head,
                        str(predicate),
                        value,
                        str(interval.start),
                        str(interval.end),
                    ]
                )
                assertion = URIRef(
                    f"urn:tcl:assertion:{sha256(identity.encode()).hexdigest()}"
                )
                graph.add((subject, RDF.type, target))
                for predicate_, object_ in (
                    (RDF.type, TCL.TemporalAssertion),
                    (RDF.subject, subject),
                    (RDF.predicate, predicate),
                    (RDF.object, obj),
                    (TCL.start, start),
                    (TCL.end, end),
                ):
                    if object_ is not None:
                        graph.add((assertion, predicate_, object_))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return graph


def load_data(path: Path, class_iri: str, data_format: str | None = None) -> Graph:
    """Read quintuplets or RDF already using the temporal assertion vocabulary."""
    if data_format == "quintuplet" or (
        data_format is None and path.suffix == ".quintuplet"
    ):
        return load_quintuplets(path, class_iri)
    graph = Graph().parse(str(path), format=data_format)
    for subject, predicate in list(graph.subject_objects(RDF.predicate)):
        if isinstance(predicate, URIRef):
            graph.remove((subject, RDF.predicate, predicate))
            graph.add((subject, RDF.predicate, _iri(to_common_uri(str(predicate)))))
    return graph


def validate_graph(data: Graph, shapes: Graph) -> tuple[bool, Graph]:
    """Run pySHACL without remote inference or mutation of the input graph."""
    try:
        from pyshacl import validate
    except ImportError as exc:
        raise RuntimeError(
            "Validation requires pySHACL: run 'uv sync --extra shacl'"
        ) from exc
    conforms, report, text = validate(data, shacl_graph=shapes, inference="none")
    if not isinstance(report, Graph):
        raise TypeError(f"SHACL validation failed: {text}")
    return bool(conforms), report


def main(argv: Sequence[str] | None = None) -> int:
    """Export shapes; return 0 for success, 1 for violations, or 2 for errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path, help="TCL JSON output")
    parser.add_argument(
        "--class-iri", required=True, help="Class whose instances share this network"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Shapes Turtle file (default: NETWORK.shacl.ttl)",
    )
    parser.add_argument("--stage", choices=STAGES, default="after_propagation")
    parser.add_argument(
        "--data", type=Path, help="Validate this quintuplet or temporal RDF input"
    )
    parser.add_argument(
        "--data-format",
        help="quintuplet or an RDFLib parser format; otherwise inferred",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Turtle validation report (default: OUTPUT.report.ttl)",
    )
    args = parser.parse_args(argv)
    if args.data is None and (args.report is not None or args.data_format is not None):
        parser.error("--report and --data-format require --data")
    output = args.output or args.network.with_suffix(".shacl.ttl")
    report_path = args.report or output.with_name(f"{output.stem}.report.ttl")
    inputs = {args.network.resolve()}
    if args.data is not None:
        inputs.add(args.data.resolve())
    outputs = [output.resolve()] + (
        [report_path.resolve()] if args.data is not None else []
    )
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        parser.error("Output files must be distinct and must not overwrite input files")
    try:
        shapes = build_shapes(
            load_constraints(args.network, args.stage), args.class_iri
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shapes.serialize(destination=str(output), format="turtle")
        count = len(set(shapes.subjects(SH.targetClass, _iri(args.class_iri))))
        print(f"SHACL shapes: {output} ({count} temporal constraints)")
        if args.data is None:
            return 0
        data = load_data(args.data, args.class_iri, args.data_format)
        conforms, report = validate_graph(data, shapes)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.serialize(destination=str(report_path), format="turtle")
        violations = len(set(report.subjects(RDF.type, SH.ValidationResult)))
        assertions = len(set(data.subjects(RDF.type, TCL.TemporalAssertion)))
        print(f"Conforms: {conforms}; assertions: {assertions}; results: {violations}")
        print(f"Validation report: {report_path}")
        return 0 if conforms else 1
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        SyntaxError,
        ParserError,
        PluginException,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
