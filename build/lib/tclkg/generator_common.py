from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from copy import copy, deepcopy
import os
from pathlib import Path

import numpy as np

from . import time_package as tp
from .allen_list import ALLEN_COMPOSE, ALLEN_CONVERSE, ALLEN_RELATIONS
from .rule_discovery import processing_date_unknown_allowed, to_common_uri

TEMPORAL_GRANULARITY = "D"
TODAY = np.datetime64("2023-12-31", TEMPORAL_GRANULARITY)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "Results"
ITERATIVE_ROOT = PROJECT_ROOT / "Results/Iterative"


def _resolve_data_root() -> Path:
    configured_data_dir = os.environ.get("TCLKG_DATA_DIR")
    if not configured_data_dir:
        return PROJECT_ROOT / "data"

    configured_path = Path(configured_data_dir).expanduser()
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


ROOT_DATA = _resolve_data_root()

SUPPORT_KEY = "support"
STATUS_KEY = "status"
RELATIONS_KEY = "relations"
SupportCounts = tuple[int, int]


def _support_ratio(value: object) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    occurrences, total = value
    if (
        isinstance(occurrences, bool)
        or isinstance(total, bool)
        or not isinstance(occurrences, int)
        or not isinstance(total, int)
        or occurrences < 0
        or total <= 0
        or occurrences > total
    ):
        return None
    return occurrences / total


def _extract_relation_payloads(dom: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Extract per-relation payloads from a structured scored domain."""
    relations_obj = dom.get(RELATIONS_KEY)
    if not isinstance(relations_obj, Mapping):
        raise ValueError(
            "Invalid scored domain format: expected a 'relations' mapping for each constraint domain."
        )
    payloads: dict[str, Mapping[str, object]] = {}
    for relation, payload in relations_obj.items():
        if relation not in ALLEN_RELATIONS:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must map to a payload."
            )
        if SUPPORT_KEY not in payload:
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must contain '{SUPPORT_KEY}'."
            )
        support = payload.get(SUPPORT_KEY)
        if not isinstance(support, (tuple, list)) or _support_ratio(support) is None and support != (0, 0) and support != [0, 0]:
            raise ValueError(
            f"Invalid scored domain format: relation '{relation}' must have an (occurrences, total) support pair."
            )
        payloads[relation] = payload
    return payloads


def _extract_relation_scores(dom: Mapping[str, object]) -> dict[str, float]:
    """Extract numeric Allen-relation supports from structured scored domains."""
    scores: dict[str, float] = {}
    for relation, payload in _extract_relation_payloads(dom).items():
        ratio = _support_ratio(payload.get(SUPPORT_KEY))
        if ratio is not None:
            scores[relation] = ratio
    return scores


def _build_structured_domain(
    relations: Mapping[str, SupportCounts],
    domain_status: str | None = None,
) -> dict[str, object]:
    """Build a scored domain with support stored per relation, plus a domain-level status."""
    return {
        STATUS_KEY: domain_status,
        RELATIONS_KEY: {
            relation: {SUPPORT_KEY: support}
            for relation, support in relations.items()
        }
    }


def read_kg(
    kg_type: str, print_properties_count: bool = True
) -> tuple[dict[str, tp.Entity], set[str]]:
    """Load a KG split into Entity objects and return seen relations/properties."""
    entities: dict[str, tp.Entity] = {}
    properties_seen: set[str] = set()

    print("DÉBUT DU CHARGEMENT DU KNOWLEDGE GRAPH")

    kg_path = ROOT_DATA / kg_type / "data.quintuplet"
    if not kg_path.exists():
        kg_path = ROOT_DATA / kg_type / "train_cst_knowledge.quintuplet"
    with kg_path.open("r", encoding="UTF-8") as f_read:
        for line in f_read:
            head, relation, value, start, end = line.rstrip("\n").split("\t")
            relation = to_common_uri(relation)
            if head not in entities:
                entities[head] = tp.Entity(head, TODAY, TEMPORAL_GRANULARITY)
            entities[head].add_triple(
                tp.Triple(
                    head,
                    relation,
                    value,
                    tp.Interval(
                        processing_date_unknown_allowed(start),
                        processing_date_unknown_allowed(end),
                    ),
                )
            )
            properties_seen.add(relation)

    print("FIN DU CHARGEMENT DU KNOWLEDGE GRAPH")
    print("Nombre d'entités chargées :", len(entities))
    if print_properties_count:
        print("Nombre de propriétés chargées :", len(properties_seen))
    return entities, properties_seen


def BasicRepairInconsistency(qcn: dict, key1: tuple[str, str], key2: tuple[str, str], composition: dict[str, float] | None = None) -> dict:
    """
    En cas d'inconsistance (domaine vide), on applique une réparation minimale en réintroduisant les relations d'Allen.
    Cette fonction est à appeler après un échec de path consistency pour tenter de réparer le réseau de contraintes et poursuivre l'apprentissage.

    :param qcn: QCN modifié après échec de path consistency (contenant un domaine vide)
    :param key1: Clé de la paire (p1, p2) dans le QCN
    :param key2: Clé de la paire inverse (p2, p1) dans le QCN
    :param composition: Domaine composé à réintroduire (si None, utilise ALLEN_RELATIONS complet)
    :return: QCN réparé avec les domaines réintroduits pour les paires inconsistantes
    """
    repaired_qcn = deepcopy(qcn)

    _extract_relation_payloads(qcn[key1])
    _extract_relation_payloads(qcn[key2])


    if composition is not None and composition:


        relation_scores = {r: (0, 0) for r in composition.keys()}
    else:

        relation_scores = {r: (0, 0) for r in ALLEN_RELATIONS}

    what_to_add = _build_structured_domain(relation_scores)
    converse_what_to_add = converse_domain_scores(what_to_add)


    repaired_qcn[key1] = what_to_add
    repaired_qcn[key2] = converse_what_to_add


    return repaired_qcn



def converse_domain_set(dom: set[str]) -> set[str]:
    """Return converse Allen relations for a set-based domain."""
    return {ALLEN_CONVERSE[r] for r in dom}


def converse_domain_scores(dom: Mapping[str, object]) -> dict[str, object]:
    """Return converse Allen relations for a structured scored domain."""
    relations = _extract_relation_payloads(dom)
    converse_relations = {
        ALLEN_CONVERSE[relation]: payload[SUPPORT_KEY]
        for relation, payload in relations.items()
    }
    return _build_structured_domain(converse_relations, dom.get(STATUS_KEY))


def build_complete_network(
    properties: set[str] | list[str],
    default_constraints: set[str] | frozenset[str] = ALLEN_RELATIONS,
) -> dict[tuple[str, str], set[str]]:
    """Build a full pairwise Allen network with set domains."""
    network: dict[tuple[str, str], set[str]] = {}
    for property1 in properties:
        for property2 in properties:
            if property1 != property2:
                network[(property1, property2)] = set(default_constraints)
    return network


def build_scored_network(
    properties: set[str] | list[str],
    default_constraints: set[str] | frozenset[str] = ALLEN_RELATIONS,
) -> dict[tuple[str, str], dict[str, float]]:
    """Build a full pairwise Allen network with scored domains."""
    network: dict[tuple[str, str], dict[str, float]] = {}
    for property1 in properties:
        for property2 in properties:
            if property1 != property2:
                network[(property1, property2)] = {
                    relation: 0.0 for relation in default_constraints
                }
    return network


def build_scored_networkwithThreshold(
    properties: set[str] | list[str],
    default_constraints: set[str] | frozenset[str] = ALLEN_RELATIONS,
    top_k_fraction: float = 0.3,
    entities: dict[str, tp.Entity] | None = None,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build a scored network and optionally filter properties by quadruplet (fact) support.

    If entities is provided, only the top `top_k_fraction` properties (by quadruplet count)
    are kept for QCN construction. Default keeps top 30%.
    """
    if top_k_fraction < 0 or top_k_fraction > 1:
        raise ValueError(f"top_k_fraction must be in [0, 1], got {top_k_fraction}")

    filtered_properties = sorted(properties)
    if entities is not None:

        property_quadruplet_count: dict[str, int] = {p: 0 for p in filtered_properties}

        for entity in entities.values():
            for prop, triples in entity.triples_per_p.items():
                if prop in property_quadruplet_count:
                    property_quadruplet_count[prop] += len(triples)


        sorted_props = sorted(
            property_quadruplet_count.items(),
            key=lambda x: (-x[1], x[0]),
        )
        if top_k_fraction == 0:
            k = 0
        else:
            k = max(1, int(np.ceil(len(filtered_properties) * top_k_fraction)))
        filtered_properties = [p for p, _ in sorted_props[:k]]

    network: dict[tuple[str, str], dict[str, object]] = {}
    for property1 in filtered_properties:
        for property2 in filtered_properties:
            if property1 != property2:
                relation_scores: dict[str, tuple[int, int]] = {
                    relation: (0, 0) for relation in default_constraints
                }
                network[(property1, property2)] = _build_structured_domain(
                    relation_scores,
                )
    return network

def compose_allen_set(r1_domain: set[str], r2_domain: set[str]) -> set[str]:
    """Compose two set-based Allen domains."""
    result: set[str] = set()
    for r1 in r1_domain:
        for r2 in r2_domain:
            result |= ALLEN_COMPOSE[(r1, r2)]
    return result


def compose_allen_scores(
    r1_domain: dict[str, float], r2_domain: dict[str, float]
) -> dict[str, float]:
    """Compose two scored Allen domains, keeping max support per relation.
    Si on a un score < 0, cela signifie que la relation est issue d'une réparation
    et qu'elle n'est pas forcément supportée par les données.
    On choisis alors de ne pas propager le score de ces relations réparées pour
    éviter de biaiser les scores des relations composées.
    """
    result: dict[str, float] = {}
    for r1, score1 in r1_domain.items():
        if r1 not in ALLEN_RELATIONS:
            continue
        if score1 < 0:
            continue
        for r2, score2 in r2_domain.items():
            if r2 not in ALLEN_RELATIONS:
                continue
            if score2 < 0:
                continue
            composed_relations = ALLEN_COMPOSE[(r1, r2)]
            min_score = min(score1, score2)
            for relation in composed_relations:
                if relation in result:
                    result[relation] = max(result[relation], min_score)
                else:
                    result[relation] = min_score
    return result


def path_consistency_classique(qcn: dict) -> dict | None:
    """
    Applique la path consistency sur le qcn
    Retourne le réseau de contraintes modifié si la propagation réussit,
    ou False si un domaine vide est rencontré (échec de la propagation).
    """
    qcn_copy = deepcopy(qcn)
    properties = sorted(
        {p for (x, y) in qcn_copy.keys() for p in (x, y)}
    )

    def intersect_with_composed_domain(
        domain: Mapping[str, object],
        left_domain: Mapping[str, object],
        right_domain: Mapping[str, object],
    ) -> dict[str, SupportCounts]:
        left_relations = set(_extract_relation_scores(left_domain))
        right_relations = set(_extract_relation_scores(right_domain))
        domain_relations = set(_extract_relation_payloads(domain))
        composed_relations = compose_allen_set(left_relations, right_relations)
        return {
            relation: payload[SUPPORT_KEY]
            for relation, payload in _extract_relation_payloads(domain).items()
            if relation in composed_relations
        }

    queue = deque(qcn_copy.keys())

    while queue:
        i, j = queue.popleft()

        for k in properties:
            if k == i or k == j:
                continue
            if (i, k) not in qcn_copy or (k, j) not in qcn_copy:
                print(
                    "Warning: missing arc in qcn_copy during propagation:",
                    (i, k),
                    "or",
                    (k, j),
                    "this should not happen !",
                )
                raise ValueError(
                    f"Missing arc in qcn_copy during propagation: {(i, k)} or {(k, j)}"
                )
            dik = qcn_copy[(i, k)]
            new_dik = intersect_with_composed_domain(
                dik,
                qcn_copy[(i, j)],
                qcn_copy[(j, k)],
            )

            dkj = qcn_copy[(k, j)]
            new_dkj = intersect_with_composed_domain(
                dkj,
                qcn_copy[(k, i)],
                qcn_copy[(i, j)],
            )

            if set(new_dik.keys()) != set(_extract_relation_scores(dik).keys()):
                if not new_dik:
                    print(
                            f"COLLAPSE: domain empty for pair ({i}, {k}) during path consistency ! ORIGIN: PATH CONSISTENCY"
                        )
                    return None
                else:
                    qcn_copy[(i, k)] = _build_structured_domain(new_dik, dik.get(STATUS_KEY))
                    qcn_copy[(k, i)] = converse_domain_scores(
                        qcn_copy[(i, k)]
                    )


                queue.append((i, k))
            if set(new_dkj.keys()) != set(_extract_relation_scores(dkj).keys()):
                if not new_dkj:

                    print(
                            f"COLLAPSE: domain empty for pair ({k}, {j}) during path consistency ! ORIGIN: PATH CONSISTENCY"
                        )
                    return None
                else:
                    qcn_copy[(k, j)] = _build_structured_domain(new_dkj, dkj.get(STATUS_KEY))
                    qcn_copy[(j, k)] = converse_domain_scores(
                        qcn_copy[(k, j)]
                    )


                queue.append((k, j))

    return qcn_copy