import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from collections import deque
from collections.abc import Collection, Mapping
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .allen_list import ALLEN_COMPOSE, ALLEN_CONVERSE, ALLEN_RELATIONS
from .generator_common import (
    RESULTS_ROOT as results_root,
)
from .generator_common import (
    ROOT_DATA,
)
from .generator_common import (
    build_scored_networkwithThreshold as build_entity_network,
)
from .generator_common import (
    read_kg as read_KG,
)
from .oracle_cache import oracle_cache_path as get_oracle_cache_path
from .oracle_cache import read_oracle_cache, write_oracle_cache
from .time_package import Entity

PROPERTY_SUPPORT_THRESHOLD = 1.0  # maybe to set later to 1.0
SUPPORT_KEY = "support"
INFERRED_SCORE_KEY = "inferred_score"
STATUS_KEY = "status"
RELATIONS_KEY = "relations"
STATUS_OBSERVED = "Observed"
# STATUS_REPAIRED = "Repaired"
STATUS_INFERRED = "Inferred"
STATUS_INFERRED_LEGACY_TYPO = "Infered"
SupportCounts = tuple[int, int]


def support_counts(value: object) -> SupportCounts | None:
    """Return a validated (occurrences, total) support pair."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    occurrences, total = value
    if (
        isinstance(occurrences, bool)
        or isinstance(total, bool)
        or not isinstance(occurrences, int)
        or not isinstance(total, int)
        or occurrences < 0
        or total < 0
        or occurrences > total
    ):
        return None
    return occurrences, total


def support_ratio(value: object) -> float | None:
    """Return the support ratio for an observed evidence pair."""
    counts = support_counts(value)
    if counts is None or counts[1] == 0:
        return None
    return counts[0] / counts[1]


def has_positive_support(value: object) -> bool:
    counts = support_counts(value)
    return counts is not None and counts[0] > 0


def has_zero_support(value: object) -> bool:
    counts = support_counts(value)
    return counts is not None and counts[1] > 0 and counts[0] == 0


def has_no_evidence(value: object) -> bool:
    return support_counts(value) == (0, 0)


def normalize_relation_status(status: object) -> str | None:
    """Normalize status labels to the canonical set used by learner2."""
    if not isinstance(status, str):
        return None

    normalized = status.strip().lower()
    if normalized == "observed":
        return STATUS_OBSERVED
    if normalized in {"inferred", "infered"}:
        return STATUS_INFERRED
    # if normalized == "repaired":
    #    return STATUS_REPAIRED
    return status


def domain_status(domain: Mapping[str, object]) -> str | None:
    """Return the domain-level status: Observed if any relation was directly observed."""
    return normalize_relation_status(domain.get(STATUS_KEY))


def relation_entries(domain: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Extract per-relation payloads from a constraint domain."""
    relations_obj = domain.get(RELATIONS_KEY)
    if not isinstance(relations_obj, Mapping):
        raise ValueError(  # noqa: TRY004 -- Preserve the existing validation API.
            "Invalid scored domain format: expected a 'relations' mapping for each constraint domain."
        )
    entries: dict[str, Mapping[str, object]] = {}
    for relation, payload in relations_obj.items():
        if relation not in ALLEN_RELATIONS:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(  # noqa: TRY004 -- Preserve the existing validation API.
                f"Invalid scored domain format: relation '{relation}' must map to a payload."
            )
        if SUPPORT_KEY not in payload:
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must contain '{SUPPORT_KEY}'."
            )
        support = payload.get(SUPPORT_KEY)
        if support_counts(support) is None:
            raise ValueError(
                f"Invalid scored domain format: relation '{relation}' must have an (occurrences, total) support pair."
            )
        entries[relation] = payload
    return entries


def filter_unobserved_relations_after_oracle(qcn: dict) -> dict:
    """Retirer les relations sans occurrence, en conservant les arcs du QCN.

    Un domaine sans observation reste inconnu avec un dictionnaire de relations
    vide ; il peut recevoir des relations lors de la propagation.
    """
    # Do not clone the zero-support payloads that are about to be discarded.
    return {
        pair: {
            **{
                key: copy.deepcopy(value)
                for key, value in domain.items()
                if key != RELATIONS_KEY
            },
            RELATIONS_KEY: {
                relation: copy.deepcopy(payload)
                for relation, payload in relation_entries(domain).items()
                if has_positive_support(payload[SUPPORT_KEY])
            },
        }
        for pair, domain in qcn.items()
    }


def relation_scores(domain: Mapping[str, object]) -> dict[str, float]:
    """Extract numeric supports from a constraint domain."""
    scores: dict[str, float] = {}
    for relation, payload in relation_entries(domain).items():
        ratio = support_ratio(payload.get(SUPPORT_KEY))
        if ratio is not None:
            scores[relation] = ratio
    return scores


def relation_statuses(domain: Mapping[str, object]) -> dict[str, str | None]:
    """Return the domain status for every relation present (relations no longer carry their own status)."""
    status = domain_status(domain)
    return {relation: status for relation in relation_entries(domain)}


def with_relation_data(relations: Mapping[str, SupportCounts]) -> dict[str, object]:
    """Build a domain with support stored per relation, plus a derived domain status."""
    computed_domain_status = (
        STATUS_OBSERVED
        if any(has_positive_support(support) for support in relations.values())
        else None
    )
    return {
        STATUS_KEY: computed_domain_status,
        RELATIONS_KEY: {
            relation: {SUPPORT_KEY: support} for relation, support in relations.items()
        },
    }


def build_output_stem(kg_type: str, reduced_composition: bool = True) -> str:
    """Build the base filename stem for Generator2 outputs."""
    composition_label = (
        "reduced_composition" if reduced_composition else "non_reduced_composition"
    )
    return f"qcn2_{kg_type}_{composition_label}"


def build_oracle_property_index(
    entities: Mapping[str, Entity],
    properties: Collection[str],
) -> dict[str, dict[str, object]]:
    """Index triples by property and entity for repeated oracle queries."""
    property_filter = set(properties)
    index: dict[str, dict[str, object]] = {
        property_: {} for property_ in property_filter
    }
    for entity_name, entity in entities.items():
        for property_, triples in entity.triples_per_p.items():
            if property_ in property_filter:
                triples_tuple = tuple(triples)
                starts = [triple.date.start for triple in triples_tuple]
                ends = [triple.date.end for triple in triples_tuple]
                if any(start is None for start in starts) or any(
                    end is None for end in ends
                ):
                    index[property_][entity_name] = triples_tuple
                else:
                    index[property_][entity_name] = (
                        np.asarray(starts, dtype="datetime64[D]").astype(np.int64),
                        np.asarray(ends, dtype="datetime64[D]").astype(np.int64),
                    )
    return index


def add_vectorized_allen_relation_matches[Count: (int, float)](
    allenrel_matching: dict[str, Count],
    a_starts: np.ndarray,
    a_ends: np.ndarray,
    b_starts: np.ndarray,
    b_ends: np.ndarray,
    max_cells_per_chunk: int = 2_000_000,
) -> int:
    """Increment Allen relation counts for all interval pairs using chunks."""
    total_pairs = 0
    if len(a_starts) == 0 or len(b_starts) == 0:
        return total_pairs

    if (
        len(a_starts) == len(b_starts) == 1
        and a_starts.dtype.kind in "iu"
        and b_starts.dtype.kind in "iu"
    ):
        _add_interval_relation_matches(
            allenrel_matching,
            a_starts[0].item(),
            a_ends[0].item(),
            b_starts[0].item(),
            b_ends[0].item(),
        )
        return 1

    rows_per_chunk = max(1, max_cells_per_chunk // len(b_starts))
    b_starts_row = b_starts[None, :]
    b_ends_row = b_ends[None, :]

    for start in range(0, len(a_starts), rows_per_chunk):
        end = min(start + rows_per_chunk, len(a_starts))
        a_starts_col = a_starts[start:end, None]
        a_ends_col = a_ends[start:end, None]
        total_pairs += (end - start) * len(b_starts)

        allenrel_matching["equals"] += int(
            np.count_nonzero(
                (a_starts_col == b_starts_row) & (a_ends_col == b_ends_row)
            )
        )
        allenrel_matching["before"] += int(np.count_nonzero(a_ends_col < b_starts_row))
        allenrel_matching["after"] += int(np.count_nonzero(b_ends_row < a_starts_col))
        allenrel_matching["meets"] += int(np.count_nonzero(a_ends_col == b_starts_row))
        allenrel_matching["met_by"] += int(np.count_nonzero(b_ends_row == a_starts_col))
        allenrel_matching["overlaps"] += int(
            np.count_nonzero(
                (a_starts_col < b_starts_row)
                & (a_ends_col > b_starts_row)
                & (a_ends_col < b_ends_row)
            )
        )
        allenrel_matching["overlapped_by"] += int(
            np.count_nonzero(
                (b_starts_row < a_starts_col)
                & (b_ends_row > a_starts_col)
                & (b_ends_row < a_ends_col)
            )
        )
        allenrel_matching["during"] += int(
            np.count_nonzero((a_starts_col > b_starts_row) & (a_ends_col < b_ends_row))
        )
        allenrel_matching["contains"] += int(
            np.count_nonzero((b_starts_row > a_starts_col) & (b_ends_row < a_ends_col))
        )
        allenrel_matching["starts"] += int(
            np.count_nonzero((a_starts_col == b_starts_row) & (a_ends_col < b_ends_row))
        )
        allenrel_matching["started_by"] += int(
            np.count_nonzero((b_starts_row == a_starts_col) & (b_ends_row < a_ends_col))
        )
        allenrel_matching["finishes"] += int(
            np.count_nonzero((a_starts_col > b_starts_row) & (a_ends_col == b_ends_row))
        )
        allenrel_matching["finished_by"] += int(
            np.count_nonzero((b_starts_row > a_starts_col) & (b_ends_row == a_ends_col))
        )

    return total_pairs


def add_allen_relation_matches[Count: (int, float)](
    allenrel_matching: dict[str, Count], qp1, qp2
) -> None:
    """Increment all Allen relations matching two triple intervals."""
    interval_a = qp1.date
    interval_b = qp2.date
    _add_interval_relation_matches(
        allenrel_matching,
        interval_a.start,
        interval_a.end,
        interval_b.start,
        interval_b.end,
    )


def _add_interval_relation_matches(
    allenrel_matching, a_start, a_end, b_start, b_end
) -> None:

    if a_start == b_start and a_end == b_end:
        allenrel_matching["equals"] += 1
    if a_end < b_start:
        allenrel_matching["before"] += 1
    if b_end < a_start:
        allenrel_matching["after"] += 1
    if a_end == b_start:
        allenrel_matching["meets"] += 1
    if b_end == a_start:
        allenrel_matching["met_by"] += 1
    if a_start < b_start and a_end > b_start and a_end < b_end:
        allenrel_matching["overlaps"] += 1
    if b_start < a_start and b_end > a_start and b_end < a_end:
        allenrel_matching["overlapped_by"] += 1
    if a_start > b_start and a_end < b_end:
        allenrel_matching["during"] += 1
    if b_start > a_start and b_end < a_end:
        allenrel_matching["contains"] += 1
    if a_start == b_start and a_end < b_end:
        allenrel_matching["starts"] += 1
    if b_start == a_start and b_end < a_end:
        allenrel_matching["started_by"] += 1
    if a_start > b_start and a_end == b_end:
        allenrel_matching["finishes"] += 1
    if b_start > a_start and b_end == a_end:
        allenrel_matching["finished_by"] += 1


def fast_converse_domain(domain: Mapping[str, Any]) -> dict[str, object]:
    """Build a converse domain without re-validating the hot-path payload shape."""
    return {
        STATUS_KEY: domain.get(STATUS_KEY),
        RELATIONS_KEY: {
            ALLEN_CONVERSE[relation]: {
                SUPPORT_KEY: payload.get(SUPPORT_KEY),
                **(
                    {INFERRED_SCORE_KEY: payload[INFERRED_SCORE_KEY]}
                    if INFERRED_SCORE_KEY in payload
                    else {}
                ),
            }
            for relation, payload in domain[RELATIONS_KEY].items()
        },
    }


@lru_cache(maxsize=100_000)
def _compose_allen_scores_cached(
    left_items: tuple[tuple[str, float], ...],
    right_items: tuple[tuple[str, float], ...],
    reduced_composition: bool = True,
) -> tuple[tuple[str, float], ...]:
    result: dict[str, float] = {}
    for r1, score1 in left_items:
        if score1 < 0:
            raise ValueError(f"Invalid score {score1} for relation {r1} in left_items")
            # continue
        for r2, score2 in right_items:
            if score2 < 0:
                raise ValueError(
                    f"Invalid score {score2} for relation {r2} in right_items"
                )
                # continue
            min_score = min(score1, score2)
            composed_relations = ALLEN_COMPOSE[(r1, r2)]
            if reduced_composition and len(composed_relations) == len(ALLEN_RELATIONS):
                continue
            for relation in composed_relations:
                existing_score = result.get(relation)
                if existing_score is None or min_score > existing_score:
                    result[relation] = min_score
    return tuple(sorted(result.items()))


def compose_allen_scores_fast(
    left_domain: dict[str, float],
    right_domain: dict[str, float],
    reduced_composition: bool = True,
) -> dict[str, float]:
    """Compose scored Allen domains with memoization for repeated propagation states."""
    if not left_domain or not right_domain:
        return {}
    left_items = tuple(sorted(left_domain.items()))
    right_items = tuple(sorted(right_domain.items()))
    return dict(
        _compose_allen_scores_cached(left_items, right_items, reduced_composition)
    )


def propagateAndFilterConservative(
    qcn: dict,
    domain_reliability_threshold: int = 0,
    relation_occurrence_threshold: float = 0.0,
    reduced_composition: bool = True,
) -> dict:
    """Compose frozen observed sources, visiting only eligible two-edge paths.

    Inferred targets never become sources in conservative propagation. Replaying
    them in a work queue therefore cannot add evidence. Keep the original source
    and intermediary order so converse updates retain the same semantics.
    """
    result = copy.deepcopy(qcn)
    properties = sorted({p for pair in result for p in pair})
    if len(properties) >= 3:
        for i in properties:
            for k in properties:
                if i != k and (i, k) not in result:
                    raise ValueError(
                        f"Missing arc in qcn_copy during propagation: {(i, k)}"
                    )

    positive = {}
    composable = {}
    observed = {}
    outgoing = {p: set() for p in properties}
    incoming = {p: set() for p in properties}
    for pair, domain in result.items():
        observed[pair] = domain_status(domain) == STATUS_OBSERVED
        scores = {}
        eligible = {}
        for relation, payload in domain[RELATIONS_KEY].items():
            support = support_counts(payload.get(SUPPORT_KEY))
            ratio = support_ratio(support)
            if support is not None and ratio is not None and ratio > 0:
                scores[relation] = ratio
                if (
                    observed[pair]
                    and support[1] >= domain_reliability_threshold
                    and ratio >= relation_occurrence_threshold
                ):
                    eligible[relation] = ratio
        positive[pair] = scores
        composable[pair] = eligible
        if eligible:
            outgoing[pair[0]].add(pair[1])
            incoming[pair[1]].add(pair[0])

    def update(pair, relation, score):
        entries = result[pair][RELATIONS_KEY]
        counts = support_counts(entries.get(relation, {}).get(SUPPORT_KEY)) or (0, 0)
        reverse = (pair[1], pair[0])
        converse = ALLEN_CONVERSE[relation]
        entries[relation] = {SUPPORT_KEY: counts, INFERRED_SCORE_KEY: float(score)}
        result[reverse][RELATIONS_KEY][converse] = {
            SUPPORT_KEY: counts,
            INFERRED_SCORE_KEY: float(score),
        }
        for target in (pair, reverse):
            if domain_status(result[target]) != STATUS_OBSERVED:
                result[target][STATUS_KEY] = STATUS_INFERRED
        positive[pair][relation] = float(score)
        positive[reverse][converse] = float(score)

    for (i, j), left in composable.items():
        if not left:
            continue
        for k in sorted(outgoing[j] | incoming[i]):
            if k == i or k == j:
                continue
            if not observed[(i, k)] and composable[(j, k)]:
                composition = compose_allen_scores_fast(
                    left, composable[(j, k)], reduced_composition
                )
                for relation, score in composition.items():
                    if (
                        relation not in positive[(i, k)]
                        or score > positive[(i, k)][relation]
                    ):
                        update((i, k), relation, score)
            if not observed[(k, j)] and composable[(k, i)]:
                composition = compose_allen_scores_fast(
                    composable[(k, i)], left, reduced_composition
                )
                for relation, score in composition.items():
                    if (
                        relation not in positive[(k, j)]
                        or score > positive[(k, j)][relation]
                    ):
                        update((k, j), relation, score)

    for domain in result.values():
        status = domain_status(domain)
        if status == STATUS_INFERRED:
            domain[RELATIONS_KEY] = {
                relation: payload
                for relation, payload in domain[RELATIONS_KEY].items()
                if INFERRED_SCORE_KEY in payload
            }
        elif status == STATUS_OBSERVED:
            domain[RELATIONS_KEY] = {
                relation: payload
                for relation, payload in domain[RELATIONS_KEY].items()
                if has_positive_support(payload.get(SUPPORT_KEY))
            }
    return result


def propagateAndFilterTwoStage(
    qcn: dict,
    domain_reliability_threshold: int = 0,
    relation_occurrence_threshold: float = 0.0,
    reduced_composition: bool = True,
) -> dict:
    """Run exactly up to two frozen stages and record inference_stage."""
    return propagateAndFilterStrass(
        qcn,
        domain_reliability_threshold,
        relation_occurrence_threshold,
        reduced_composition,
        max_stages=2,
    )


def propagateAndFilterStrass(
    qcn: dict,
    domain_reliability_threshold: int = 0,
    relation_occurrence_threshold: float = 0.0,
    reduced_composition: bool = True,
    max_stages: int | None = None,
) -> dict:
    """Propagate in frozen stages, stopping at max_stages or no new domains.

    Stage 1 uses observed evidence. Each subsequent stage uses observed domains
    and earlier inferred domains, targeting only pairs still None at its start.
    All paths contribute before a new domain is frozen and becomes a source.
    inference_stage records its first stage. None means no stage limit.
    """
    if max_stages is not None and (
        isinstance(max_stages, bool)
        or not isinstance(max_stages, int)
        or max_stages < 1
    ):
        raise ValueError("max_stages must be a positive integer or None")
    qcn_copy = propagateAndFilterConservative(
        qcn,
        domain_reliability_threshold,
        relation_occurrence_threshold,
        reduced_composition,
    )
    for domain in qcn_copy.values():
        if domain_status(domain) == STATUS_INFERRED:
            domain["inference_stage"] = 1
    stage = 2
    while max_stages is None or stage <= max_stages:
        remaining_pairs = {
            pair for pair, domain in qcn_copy.items() if domain_status(domain) is None
        }
        if not remaining_pairs:
            break
        properties = sorted({p for pair in qcn_copy for p in pair})
        sources = {}
        for pair, domain in qcn_copy.items():
            scores = {}
            status = domain_status(domain)
            for relation, payload in domain[RELATIONS_KEY].items():
                if status == STATUS_INFERRED:
                    score = payload.get(INFERRED_SCORE_KEY)
                    if (
                        isinstance(score, (int, float))
                        and not isinstance(score, bool)
                        and 0 < score <= 1
                    ):
                        scores[relation] = float(score)
                elif status == STATUS_OBSERVED:
                    support = support_counts(payload.get(SUPPORT_KEY))
                    ratio = support_ratio(support)
                    if (
                        support is not None
                        and ratio is not None
                        and ratio > 0
                        and support[1] >= domain_reliability_threshold
                        and ratio >= relation_occurrence_threshold
                    ):
                        scores[relation] = ratio
            sources[pair] = scores

        for i, k in sorted(remaining_pairs):
            if (k, i) not in remaining_pairs or i >= k:
                continue
            inferred = {}
            for j in properties:
                if j == i or j == k:
                    continue
                composition = compose_allen_scores_fast(
                    sources[(i, j)],
                    sources[(j, k)],
                    reduced_composition,
                )
                for relation, score in composition.items():
                    inferred[relation] = max(score, inferred.get(relation, score))
            if not inferred:
                continue
            entries = qcn_copy[(i, k)][RELATIONS_KEY]
            # Keep existing inference scores when present, as in the first stage.
            for relation, score in inferred.items():
                previous = entries.get(relation, {})
                old_score = previous.get(INFERRED_SCORE_KEY)
                if old_score is None or score > old_score:
                    entries[relation] = {
                        SUPPORT_KEY: support_counts(previous.get(SUPPORT_KEY))
                        or (0, 0),
                        INFERRED_SCORE_KEY: score,
                    }
                    qcn_copy[(k, i)][RELATIONS_KEY][ALLEN_CONVERSE[relation]] = dict(
                        entries[relation]
                    )
            for pair in ((i, k), (k, i)):
                qcn_copy[pair][STATUS_KEY] = STATUS_INFERRED
                qcn_copy[pair]["inference_stage"] = stage
                qcn_copy[pair][RELATIONS_KEY] = {
                    relation: payload
                    for relation, payload in qcn_copy[pair][RELATIONS_KEY].items()
                    if INFERRED_SCORE_KEY in payload
                }
        if not any(
            domain_status(qcn_copy[pair]) == STATUS_INFERRED for pair in remaining_pairs
        ):
            break
        stage += 1
    return qcn_copy


def propagateAndFilterIterative(
    qcn: dict,
    domain_reliability_threshold: int = 0,
    relation_occurrence_threshold: float = 0.0,
    reduced_composition: bool = True,
) -> dict:
    """Propagate observed and inferred evidence until no domain changes.

    A path is (i, j, k), composing (i, j) with (j, k) for target (i, k).
    None and Inferred targets remain eligible whenever a source changes.
    Updated domains and their converses are queued again, so paths are replayed
    after new relations or improved scores. Only actual changes are queued.
    Scores increase through max/min over the finite set of input scores, which
    ensures termination once all paths have propagated their latest evidence.
    Observed sources use thresholded support ratios; inferred sources use
    positive inferred_score values. Observed targets are never modified.
    """
    result = copy.deepcopy(qcn)
    properties = sorted({p for pair in result for p in pair})
    for i in properties:
        for k in properties:
            if i != k and (i, k) not in result:
                raise ValueError(f"Missing arc in QCN: {(i, k)}")

    sources = {}

    def refresh(pair):
        domain = result[pair]
        status = domain_status(domain)
        scores = {}
        for relation, payload in domain[RELATIONS_KEY].items():
            if status == STATUS_OBSERVED:
                counts = support_counts(payload.get(SUPPORT_KEY))
                score = support_ratio(counts)
                if (
                    counts is not None
                    and score is not None
                    and score > 0
                    and counts[1] >= domain_reliability_threshold
                    and score >= relation_occurrence_threshold
                ):
                    scores[relation] = score
            elif status == STATUS_INFERRED:
                score = payload.get(INFERRED_SCORE_KEY)
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and 0 < score <= 1
                ):
                    scores[relation] = float(score)
        sources[pair] = scores

    for pair in result:
        refresh(pair)
    queue = deque(sorted(pair for pair in result if sources[pair]))
    queued = set(queue)

    def infer(i, j, k):
        pair, reverse = (i, k), (k, i)
        if any(
            domain_status(result[target]) not in (None, STATUS_INFERRED)
            for target in (pair, reverse)
        ):
            return
        composition = compose_allen_scores_fast(
            sources[(i, j)],
            sources[(j, k)],
            reduced_composition,
        )
        if not composition:
            return
        changed = set()
        for relation, score in composition.items():
            for target, target_relation in (
                (pair, relation),
                (reverse, ALLEN_CONVERSE[relation]),
            ):
                entries = result[target][RELATIONS_KEY]
                previous = entries.get(target_relation, {})
                old_score = previous.get(INFERRED_SCORE_KEY)
                if old_score is None or score > old_score:
                    entries[target_relation] = {
                        SUPPORT_KEY: support_counts(previous.get(SUPPORT_KEY))
                        or (0, 0),
                        INFERRED_SCORE_KEY: score,
                    }
                    changed.add(target)
                if domain_status(result[target]) is None:
                    result[target][STATUS_KEY] = STATUS_INFERRED
                    changed.add(target)
        for target in sorted(changed):
            refresh(target)
            if target not in queued:
                queue.append(target)
                queued.add(target)

    while queue:
        i, j = queue.popleft()
        queued.remove((i, j))
        for k in properties:
            if k != i and k != j:
                infer(i, j, k)
                infer(k, i, j)

    for domain in result.values():
        status = domain_status(domain)
        if status == STATUS_INFERRED:
            domain[RELATIONS_KEY] = {
                relation: payload
                for relation, payload in domain[RELATIONS_KEY].items()
                if INFERRED_SCORE_KEY in payload
            }
        elif status == STATUS_OBSERVED:
            domain[RELATIONS_KEY] = {
                relation: payload
                for relation, payload in domain[RELATIONS_KEY].items()
                if has_positive_support(payload.get(SUPPORT_KEY))
            }
    return result


def learner2(
    qcn: dict,
    entities: dict,
    properties: Collection[str],
    domain_reliability_threshold: int = 0,
    relation_Occurence_threshold: float = 0.0,
    reduced_composition: bool = True,
    mode: str = "conservative",
    max_stages: int | None = None,
    prepared_oracle: dict[str, Any] | None = None,
    oracle_cache_path: Path | None = None,
) -> dict | None:
    """
    Learner threshold-free with an oracle phase followed by propagation/filtering.
    Chaque relation du QCN porte un score correspondant à la proportion d'entités
    vérifiant la relation d'Allen pour le couple de propriétés considéré.
    """

    cpu_start = time.process_time()
    oracle_phase_start = cpu_start
    if prepared_oracle is None:
        initial_qcn = copy.deepcopy(qcn)
        kept_properties = set(properties)
        kept_facts_count = 0
        for entity in entities.values():
            for prop, triples in entity.triples_per_p.items():
                if prop in kept_properties:
                    kept_facts_count += len(triples)

        total_oracle_calls = 0
        min_oracle_cpu_time = float("inf")
        max_oracle_cpu_time = float("-inf")
        oracle_sum_cpu_time = 0.0
        oracle_property_index = build_oracle_property_index(entities, properties)
        for total_oracle_calls, (p1, p2) in enumerate(combinations(properties, 2), 1):
            if (p1, p2) not in qcn:
                print(
                    "Warning: missing arc in qcn for pair:",
                    (p1, p2),
                    "this should not happen !",
                )
                raise ValueError(f"Missing arc in qcn for pair: {(p1, p2)}")
            if (p2, p1) not in qcn:
                print(
                    "Warning: missing arc in qcn for pair:",
                    (p2, p1),
                    "this should not happen !",
                )
                raise ValueError(f"Missing arc in qcn for pair: {(p2, p1)}")

            V = (p1, p2)
            oraclestart = time.process_time()

            query_result = Query(V, entities, oracle_property_index)
            oracle_end_time = time.process_time() - oraclestart
            min_oracle_cpu_time = min(min_oracle_cpu_time, oracle_end_time)
            max_oracle_cpu_time = max(max_oracle_cpu_time, oracle_end_time)
            oracle_sum_cpu_time += oracle_end_time

            if query_result is not None:
                qcn[(p1, p2)] = query_result

                qcn[(p2, p1)] = fast_converse_domain(qcn[(p1, p2)])

                if not relation_entries(qcn[(p1, p2)]):
                    print(f"COLLAPSE: apres la requête sur {V} ! ORIGINE: ORACLE")
                    return None

        oracle_phase_cpu_endtime = time.process_time() - oracle_phase_start

        qcn = filter_unobserved_relations_after_oracle(qcn)
        if oracle_cache_path is not None:
            write_oracle_cache(
                oracle_cache_path,
                {
                    "initial": initial_qcn,
                    "after_oracle": qcn,
                    "properties": list(properties),
                    "kept_facts_count": kept_facts_count,
                },
            )
    else:
        initial_qcn = copy.deepcopy(prepared_oracle["initial"])
        qcn = copy.deepcopy(prepared_oracle["after_oracle"])
        kept_properties = set(properties)
        kept_facts_count = prepared_oracle["kept_facts_count"]
        total_oracle_calls = 0
        min_oracle_cpu_time = float("inf")
        max_oracle_cpu_time = float("-inf")
        oracle_sum_cpu_time = 0.0
        oracle_phase_cpu_endtime = time.process_time() - oracle_phase_start

    qcn_after_oracle = qcn

    qcn_stats, after_oracle_report_stats = (
        compute_qcn_relation_stats_with_after_oracle_report(qcn_after_oracle)
    )
    avg_oracle_cpu = (
        oracle_sum_cpu_time / total_oracle_calls if total_oracle_calls > 0 else 0.0
    )
    min_oracle_cpu_time_safe = min_oracle_cpu_time if total_oracle_calls > 0 else 0.0
    max_oracle_cpu_time_safe = max_oracle_cpu_time if total_oracle_calls > 0 else 0.0

    print(f"\n{'─' * 60}")
    print("  STATISTIQUES QCN — après phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Domaines (paires)          : {qcn_stats['num_domains']}")
    print(f"  Relations totales          : {qcn_stats['total_relations']}")
    print(f"  Rel observed & sup > 0     : {qcn_stats['count_observed']}")
    print(
        f"    -> proportion            : {qcn_stats['count_observed']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_observed_sup_pos']:.2f}%)"
    )
    print(f"  Relations sans preuve (0, 0)   : {qcn_stats['count_none_none']}")
    print(
        f"    -> proportion            : {qcn_stats['count_none_none']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_none']:.2f}%)"
    )
    print(f"  Relations testées absentes (0,n): {qcn_stats['count_none_zero']}")
    print(
        f"    -> proportion            : {qcn_stats['count_none_zero']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_zero']:.2f}%)"
    )
    print(f"  Rel inferred               : {qcn_stats['count_inferred']}")
    print(
        f"  % domaines sans relation observée : "
        f"{qcn_stats['count_domains_without_observed']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_without_observed']:.2f}%)"
    )
    print(
        f"  % domaines avec >=1 rel observée & sup>0 : "
        f"{qcn_stats['count_domains_with_observed_sup_pos']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_with_observed_sup_pos']:.2f}%)"
    )
    print(
        f"  Support > 0 / domaine      : moy={qcn_stats['avg_support_per_domain']:.2f}"
        f"  min={qcn_stats['min_support_per_domain']}"
        f"  max={qcn_stats['max_support_per_domain']}"
        f"  σ={qcn_stats['std_support_per_domain']:.2f}"
    )
    print(f"{'─' * 60}")
    print("  TEMPS CPU — phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Propriétés gardées          : {len(kept_properties)}")
    print(f"  Faits gardés                : {kept_facts_count}")
    print(f"  Appels oracle              : {total_oracle_calls}")
    print(f"  Coût total phase oracle    : {oracle_phase_cpu_endtime:.4f} s")
    print(
        f"  Temps oracle / appel — moy : {avg_oracle_cpu:.6f} s"
        f"  min={min_oracle_cpu_time_safe:.6f} s"
        f"  max={max_oracle_cpu_time_safe:.6f} s"
    )
    print(f"{'─' * 60}\n")

    propagate_and_filter_start = time.process_time()
    propagate = {
        "conservative": propagateAndFilterConservative,
        "two_stage": propagateAndFilterTwoStage,
        "iterative": propagateAndFilterIterative,
        "fix_point": propagateAndFilterIterative,
        "strass": propagateAndFilterStrass,
    }[mode]
    qcn = propagate(
        qcn,
        domain_reliability_threshold,
        relation_Occurence_threshold,
        reduced_composition,
        **({"max_stages": max_stages} if mode == "strass" else {}),
    )
    propagate_and_filter_cpu_time = time.process_time() - propagate_and_filter_start

    qcn_after_propagation = qcn
    qcn_after_propagation_stats = compute_qcn_relation_stats(qcn_after_propagation)
    oracle_to_propagation_transition_stats = (
        compute_oracle_to_propagation_transition_stats(
            initial_qcn,
            qcn_after_oracle,
            qcn_after_propagation,
        )
    )
    learner_total_cpu_time = time.process_time() - cpu_start
    cumulative_after_oracle = oracle_phase_cpu_endtime
    cumulative_after_propagation = (
        oracle_phase_cpu_endtime + propagate_and_filter_cpu_time
    )

    print(f"{'─' * 60}")
    print("  TEMPS CPU — learner2")
    print(f"{'─' * 60}")
    print(f"  Oracle                          : {oracle_phase_cpu_endtime:.4f} s")
    print(f"  Propagation + filtre            : {propagate_and_filter_cpu_time:.4f} s")
    print(f"  Total learner2                  : {learner_total_cpu_time:.4f} s")
    print(f"{'─' * 60}\n")

    return {
        "initial": initial_qcn,
        "after_oracle": qcn_after_oracle,
        "after_propagation": qcn_after_propagation,
        "oracle_stats": {
            "total_calls": total_oracle_calls,
            "min_cpu_time": min_oracle_cpu_time_safe,
            "max_cpu_time": max_oracle_cpu_time_safe,
            "avg_cpu_time": avg_oracle_cpu,
            "cost_of_the_oracle_phase": oracle_phase_cpu_endtime,
        },
        "property_filter_stats": {
            "top_k_fraction": PROPERTY_SUPPORT_THRESHOLD,
            "kept_properties_count": len(kept_properties),
            "kept_facts_count": kept_facts_count,
        },
        "qcn_stats": qcn_stats,
        "after_oracle_report_stats": after_oracle_report_stats,
        "after_propagation_qcn_stats": qcn_after_propagation_stats,
        "oracle_to_propagation_transition_stats": oracle_to_propagation_transition_stats,
        "timing": {
            "oracle_phase_cpu_time": oracle_phase_cpu_endtime,
            "propagate_and_filter_cpu_time": propagate_and_filter_cpu_time,
            "learner_total_cpu_time": learner_total_cpu_time,
            "cumulative_after_oracle_cpu_time": cumulative_after_oracle,
            "cumulative_after_propagation_cpu_time": cumulative_after_propagation,
        },
    }


def learnerTuned(
    qcn: dict,
    entities: dict,
    properties: Collection[str],
    domain_reliability_threshold: int = 0,
    relation_Occurence_threshold: float = 0.0,
    tuning_sample_size: int = 10,
    random_seed: int | None = None,
    reduced_composition: bool = True,
    mode: str = "conservative",
    max_stages: int | None = None,
) -> dict | None:
    """
    Evaluate propagation by hiding sampled observed constraints after the oracle.

    A constraint and its converse are hidden together to avoid leaking the
    oracle answer through the reverse arc. ``random_seed`` makes the experiment
    reproducible when desired.
    """

    if tuning_sample_size < 0:
        raise ValueError("tuning_sample_size must be non-negative")

    cpu_start = time.process_time()
    oracle_phase_start = cpu_start
    initial_qcn = copy.deepcopy(qcn)
    kept_properties = set(properties)
    kept_facts_count = 0
    for entity in entities.values():
        for prop, triples in entity.triples_per_p.items():
            if prop in kept_properties:
                kept_facts_count += len(triples)

    total_oracle_calls = 0
    min_oracle_cpu_time = float("inf")
    max_oracle_cpu_time = float("-inf")
    oracle_sum_cpu_time = 0.0
    oracle_property_index = build_oracle_property_index(entities, properties)
    for p1, p2 in combinations(properties, 2):
        if (p1, p2) not in qcn:
            print(
                "Warning: missing arc in qcn for pair:",
                (p1, p2),
                "this should not happen !",
            )
            raise ValueError(f"Missing arc in qcn for pair: {(p1, p2)}")
        if (p2, p1) not in qcn:
            print(
                "Warning: missing arc in qcn for pair:",
                (p2, p1),
                "this should not happen !",
            )
            raise ValueError(f"Missing arc in qcn for pair: {(p2, p1)}")

        V = (p1, p2)
        oraclestart = time.process_time()

        query_result = Query(V, entities, oracle_property_index)
        oracle_end_time = time.process_time() - oraclestart
        total_oracle_calls += 1
        min_oracle_cpu_time = min(min_oracle_cpu_time, oracle_end_time)
        max_oracle_cpu_time = max(max_oracle_cpu_time, oracle_end_time)
        oracle_sum_cpu_time += oracle_end_time

        if query_result is not None:
            qcn[(p1, p2)] = query_result

            qcn[(p2, p1)] = fast_converse_domain(qcn[(p1, p2)])

            if not relation_entries(qcn[(p1, p2)]):
                print(f"COLLAPSE: apres la requête sur {V} ! ORIGINE: ORACLE")
                return None

    oracle_phase_cpu_endtime = time.process_time() - oracle_phase_start

    qcn = filter_unobserved_relations_after_oracle(qcn)
    # Keep the filtered oracle result: the working QCN is masked below.
    qcn_after_oracle = copy.deepcopy(qcn)

    qcn_stats, after_oracle_report_stats = (
        compute_qcn_relation_stats_with_after_oracle_report(qcn_after_oracle)
    )
    avg_oracle_cpu = (
        oracle_sum_cpu_time / total_oracle_calls if total_oracle_calls > 0 else 0.0
    )
    min_oracle_cpu_time_safe = min_oracle_cpu_time if total_oracle_calls > 0 else 0.0
    max_oracle_cpu_time_safe = max_oracle_cpu_time if total_oracle_calls > 0 else 0.0

    print(f"\n{'─' * 60}")
    print("  STATISTIQUES QCN — après phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Domaines (paires)          : {qcn_stats['num_domains']}")
    print(f"  Relations totales          : {qcn_stats['total_relations']}")
    print(f"  Rel observed & sup > 0     : {qcn_stats['count_observed']}")
    print(
        f"    -> proportion            : {qcn_stats['count_observed']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_observed_sup_pos']:.2f}%)"
    )
    print(f"  Relations sans preuve (0, 0)   : {qcn_stats['count_none_none']}")
    print(
        f"    -> proportion            : {qcn_stats['count_none_none']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_none']:.2f}%)"
    )
    print(f"  Relations testées absentes (0,n): {qcn_stats['count_none_zero']}")
    print(
        f"    -> proportion            : {qcn_stats['count_none_zero']} / "
        f"{qcn_stats['total_relations']} "
        f"({qcn_stats['percent_none_zero']:.2f}%)"
    )
    print(f"  Rel inferred               : {qcn_stats['count_inferred']}")
    print(
        f"  % domaines sans relation observée : "
        f"{qcn_stats['count_domains_without_observed']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_without_observed']:.2f}%)"
    )
    print(
        f"  % domaines avec >=1 rel observée & sup>0 : "
        f"{qcn_stats['count_domains_with_observed_sup_pos']} / {qcn_stats['num_domains']} "
        f"({qcn_stats['percent_domains_with_observed_sup_pos']:.2f}%)"
    )
    print(
        f"  Support > 0 / domaine      : moy={qcn_stats['avg_support_per_domain']:.2f}"
        f"  min={qcn_stats['min_support_per_domain']}"
        f"  max={qcn_stats['max_support_per_domain']}"
        f"  σ={qcn_stats['std_support_per_domain']:.2f}"
    )
    print(f"{'─' * 60}")
    print("  TEMPS CPU — phase Oracle")
    print(f"{'─' * 60}")
    print(f"  Propriétés gardées          : {len(kept_properties)}")
    print(f"  Faits gardés                : {kept_facts_count}")
    print(f"  Appels oracle              : {total_oracle_calls}")
    print(f"  Coût total phase oracle    : {oracle_phase_cpu_endtime:.4f} s")
    print(
        f"  Temps oracle / appel — moy : {avg_oracle_cpu:.6f} s"
        f"  min={min_oracle_cpu_time_safe:.6f} s"
        f"  max={max_oracle_cpu_time_safe:.6f} s"
    )
    print(f"{'─' * 60}\n")
    # Work with unordered constraints: both directed arcs represent the same
    # Allen constraint and must be selected/masked as one unit.
    observed_constraints: list[tuple[str, str]] = []
    for p1, p2 in combinations(properties, 2):
        if (
            domain_status(qcn_after_oracle[(p1, p2)]) == STATUS_OBSERVED
            and domain_status(qcn_after_oracle[(p2, p1)]) == STATUS_OBSERVED
        ):
            observed_constraints.append((p1, p2))

    sample_size = tuning_sample_size
    rng = random.Random(random_seed)
    selected_domains = rng.sample(observed_constraints, sample_size)
    selected_domains_before = {
        pair: copy.deepcopy(qcn_after_oracle[pair]) for pair in selected_domains
    }

    for p1, p2 in selected_domains:
        for pair in ((p1, p2), (p2, p1)):
            qcn[pair][STATUS_KEY] = None
            qcn[pair][RELATIONS_KEY] = {
                relation: {SUPPORT_KEY: (0, 0)} for relation in ALLEN_RELATIONS
            }

    qcn_before_propagation = copy.deepcopy(qcn)

    propagate_and_filter_start = time.process_time()
    propagate = {
        "conservative": propagateAndFilterConservative,
        "two_stage": propagateAndFilterTwoStage,
        "iterative": propagateAndFilterIterative,
        "fix_point": propagateAndFilterIterative,
        "strass": propagateAndFilterStrass,
    }[mode]
    qcn = propagate(
        qcn,
        domain_reliability_threshold,
        relation_Occurence_threshold,
        reduced_composition,
        **({"max_stages": max_stages} if mode == "strass" else {}),
    )
    propagate_and_filter_cpu_time = time.process_time() - propagate_and_filter_start

    qcn_after_propagation = qcn

    selected_domains_after = {
        pair: copy.deepcopy(qcn_after_propagation[pair]) for pair in selected_domains
    }
    print(f"\n{'\u2500' * 60}")
    print(f"  LEARNER TUNED — {sample_size} domaines masqués")
    print(f"{'\u2500' * 60}")
    for pair in selected_domains:
        before = {
            relation: score
            for relation, score in relation_scores(
                selected_domains_before[pair]
            ).items()
            if score > 0
        }
        after = {
            relation: score
            for relation, score in relation_scores(selected_domains_after[pair]).items()
            if score > 0
        }
        inferred = {
            relation: payload[INFERRED_SCORE_KEY]
            for relation, payload in relation_entries(
                selected_domains_after[pair]
            ).items()
            if INFERRED_SCORE_KEY in payload
        }
        after_display = inferred or after
        print(f"  {pair[0]} -> {pair[1]}")
        print(f"    avant masquage       : taille={len(before)}, domaine={before}")
        print(
            f"    après propagation   : taille={len(after_display)}, "
            f"domaine={after_display}"
        )
    print(f"{'\u2500' * 60}\n")

    qcn_after_propagation_stats = compute_qcn_relation_stats(qcn_after_propagation)
    oracle_to_propagation_transition_stats = (
        compute_oracle_to_propagation_transition_stats(
            initial_qcn,
            qcn_before_propagation,
            qcn_after_propagation,
        )
    )
    learner_total_cpu_time = time.process_time() - cpu_start
    cumulative_after_oracle = oracle_phase_cpu_endtime
    cumulative_after_propagation = (
        oracle_phase_cpu_endtime + propagate_and_filter_cpu_time
    )

    print(f"{'─' * 60}")
    print("  TEMPS CPU — learnerTuned")
    print(f"{'─' * 60}")
    print(f"  Oracle                          : {oracle_phase_cpu_endtime:.4f} s")
    print(f"  Propagation + filtre            : {propagate_and_filter_cpu_time:.4f} s")
    print(f"  Total learnerTuned              : {learner_total_cpu_time:.4f} s")
    print(f"{'─' * 60}\n")

    return {
        "initial": initial_qcn,
        "after_oracle": qcn_after_oracle,
        "before_propagation": qcn_before_propagation,
        "after_propagation": qcn_after_propagation,
        "tuning_evaluation": {
            "requested_sample_size": tuning_sample_size,
            "sample_size": sample_size,
            "random_seed": random_seed,
            "selected_domains": selected_domains,
            "before": selected_domains_before,
            "after": selected_domains_after,
        },
        "oracle_stats": {
            "total_calls": total_oracle_calls,
            "min_cpu_time": min_oracle_cpu_time_safe,
            "max_cpu_time": max_oracle_cpu_time_safe,
            "avg_cpu_time": avg_oracle_cpu,
            "cost_of_the_oracle_phase": oracle_phase_cpu_endtime,
        },
        "property_filter_stats": {
            "top_k_fraction": PROPERTY_SUPPORT_THRESHOLD,
            "kept_properties_count": len(kept_properties),
            "kept_facts_count": kept_facts_count,
        },
        "qcn_stats": qcn_stats,
        "after_oracle_report_stats": after_oracle_report_stats,
        "after_propagation_qcn_stats": qcn_after_propagation_stats,
        "oracle_to_propagation_transition_stats": oracle_to_propagation_transition_stats,
        "timing": {
            "oracle_phase_cpu_time": oracle_phase_cpu_endtime,
            "propagate_and_filter_cpu_time": propagate_and_filter_cpu_time,
            "learner_total_cpu_time": learner_total_cpu_time,
            "cumulative_after_oracle_cpu_time": cumulative_after_oracle,
            "cumulative_after_propagation_cpu_time": cumulative_after_propagation,
        },
    }


def count_total_relations(qcn):
    """Compte le nombre total de relations d'Allen dans tous les domaines"""
    total = 0
    for domain in qcn.values():
        total += len(relation_entries(domain))
    return total


def compute_qcn_relation_stats_with_after_oracle_report(qcn: dict) -> tuple[dict, dict]:
    """Compute QCN stats and after-oracle report counters in one pass."""
    import math

    total_relations = 0
    count_observed = 0
    count_inferred = 0
    count_none_none = 0
    count_none_zero = 0
    count_domains_without_observed = 0
    count_domains_with_observed_sup_pos = 0
    support_counts_per_domain: list[int] = []
    inferred_scores: list[float] = []

    observed_support_pos_relations = 0
    zero_support_relations = 0
    domains_with_observed_status = 0
    domains_with_none_support = 0

    for domain in qcn.values():
        entries = relation_entries(domain)
        total_relations += len(entries)
        domain_support_count = 0
        domain_is_observed = domain_status(domain) == STATUS_OBSERVED
        report_has_none_support = not entries

        for payload in entries.values():
            support = payload.get(SUPPORT_KEY)
            inferred_score = payload.get(INFERRED_SCORE_KEY)
            if isinstance(inferred_score, (int, float)) and not isinstance(
                inferred_score, bool
            ):
                count_inferred += 1
                inferred_scores.append(float(inferred_score))
            elif domain_is_observed and has_positive_support(support):
                count_observed += 1
            elif has_zero_support(support):
                count_none_zero += 1
            elif has_no_evidence(support):
                count_none_none += 1

            if has_positive_support(support):
                observed_support_pos_relations += 1
                domain_support_count += 1
            elif has_zero_support(support):
                zero_support_relations += 1

            if has_no_evidence(support):
                report_has_none_support = True

        if not domain_is_observed:
            count_domains_without_observed += 1
        else:
            count_domains_with_observed_sup_pos += 1
        if domain_is_observed:
            domains_with_observed_status += 1
        if report_has_none_support:
            domains_with_none_support += 1
        support_counts_per_domain.append(domain_support_count)

    n = len(support_counts_per_domain)
    if n > 0:
        avg_support = sum(support_counts_per_domain) / n
        variance = sum((x - avg_support) ** 2 for x in support_counts_per_domain) / n
        std_support = math.sqrt(variance)
        min_support = min(support_counts_per_domain)
        max_support = max(support_counts_per_domain)
    else:
        avg_support = std_support = min_support = max_support = 0.0

    qcn_stats = {
        "num_domains": n,
        "total_relations": total_relations,
        "count_observed": count_observed,
        "count_inferred": count_inferred,
        "count_none_none": count_none_none,
        "count_none_zero": count_none_zero,
        "percent_observed_sup_pos": (
            (count_observed / total_relations * 100) if total_relations > 0 else 0.0
        ),
        "percent_none_none": (
            (count_none_none / total_relations * 100) if total_relations > 0 else 0.0
        ),
        "percent_none_zero": (
            (count_none_zero / total_relations * 100) if total_relations > 0 else 0.0
        ),
        "count_domains_without_observed": count_domains_without_observed,
        "count_domains_with_observed_sup_pos": count_domains_with_observed_sup_pos,
        "percent_domains_without_observed": (
            (count_domains_without_observed / n * 100) if n > 0 else 0.0
        ),
        "percent_domains_with_observed_sup_pos": (
            (count_domains_with_observed_sup_pos / n * 100) if n > 0 else 0.0
        ),
        "avg_support_per_domain": avg_support,
        "std_support_per_domain": std_support,
        "min_support_per_domain": min_support,
        "max_support_per_domain": max_support,
        "avg_inferred_score": (
            sum(inferred_scores) / len(inferred_scores) if inferred_scores else 0.0
        ),
        "min_inferred_score": min(inferred_scores) if inferred_scores else 0.0,
        "max_inferred_score": max(inferred_scores) if inferred_scores else 0.0,
    }

    after_oracle_report_stats = {
        "total_domains": len(qcn),
        "total_relations": total_relations,
        "observed_support_pos_relations": observed_support_pos_relations,
        "zero_support_relations": zero_support_relations,
        "domains_with_observed_status": domains_with_observed_status,
        "domains_with_none_support": domains_with_none_support,
    }
    return qcn_stats, after_oracle_report_stats


def compute_qcn_relation_stats(qcn: dict) -> dict:
    """
    Calcule des statistiques sur les relations d'un QCN :
    - nombre total de relations
        - nombre de relations par statut métier (Observed, Inferred,
            status=None avec support None/0)
        - pour chaque domaine : nombre de relations avec support > 0
      (min, max, moyenne, écart-type sur l'ensemble des domaines)
    """
    import math

    total_relations = 0
    count_observed = 0
    count_inferred = 0
    count_none_none = 0
    count_none_zero = 0
    count_domains_without_observed = 0
    count_domains_with_observed_sup_pos = 0
    support_counts_per_domain: list[int] = []
    inferred_scores: list[float] = []

    for domain in qcn.values():
        entries = relation_entries(domain)
        total_relations += len(entries)
        domain_support_count = 0
        domain_is_observed = domain_status(domain) == STATUS_OBSERVED
        for payload in entries.values():
            support = payload.get(SUPPORT_KEY)
            inferred_score = payload.get(INFERRED_SCORE_KEY)
            if isinstance(inferred_score, (int, float)) and not isinstance(
                inferred_score, bool
            ):
                count_inferred += 1
                inferred_scores.append(float(inferred_score))
            elif domain_is_observed and has_positive_support(support):
                count_observed += 1
            elif has_zero_support(support):
                count_none_zero += 1
            elif has_no_evidence(support):
                count_none_none += 1

            if has_positive_support(support):
                domain_support_count += 1

        if not domain_is_observed:
            count_domains_without_observed += 1
        else:
            count_domains_with_observed_sup_pos += 1
        support_counts_per_domain.append(domain_support_count)

    n = len(support_counts_per_domain)
    if n > 0:
        avg_support = sum(support_counts_per_domain) / n
        variance = sum((x - avg_support) ** 2 for x in support_counts_per_domain) / n
        std_support = math.sqrt(variance)
        min_support = min(support_counts_per_domain)
        max_support = max(support_counts_per_domain)
    else:
        avg_support = std_support = min_support = max_support = 0.0

    percent_observed_sup_pos = (
        (count_observed / total_relations * 100) if total_relations > 0 else 0.0
    )
    percent_none_none = (
        (count_none_none / total_relations * 100) if total_relations > 0 else 0.0
    )
    percent_none_zero = (
        (count_none_zero / total_relations * 100) if total_relations > 0 else 0.0
    )
    percent_domains_without_observed = (
        (count_domains_without_observed / n * 100) if n > 0 else 0.0
    )
    percent_domains_with_observed_sup_pos = (
        (count_domains_with_observed_sup_pos / n * 100) if n > 0 else 0.0
    )

    return {
        "num_domains": n,
        "total_relations": total_relations,
        "count_observed": count_observed,
        "count_inferred": count_inferred,
        "count_none_none": count_none_none,
        "count_none_zero": count_none_zero,
        "percent_observed_sup_pos": percent_observed_sup_pos,
        "percent_none_none": percent_none_none,
        "percent_none_zero": percent_none_zero,
        "count_domains_without_observed": count_domains_without_observed,
        "count_domains_with_observed_sup_pos": count_domains_with_observed_sup_pos,
        "percent_domains_without_observed": percent_domains_without_observed,
        "percent_domains_with_observed_sup_pos": percent_domains_with_observed_sup_pos,
        "avg_support_per_domain": avg_support,
        "std_support_per_domain": std_support,
        "min_support_per_domain": min_support,
        "max_support_per_domain": max_support,
        "avg_inferred_score": (
            sum(inferred_scores) / len(inferred_scores) if inferred_scores else 0.0
        ),
        "min_inferred_score": min(inferred_scores) if inferred_scores else 0.0,
        "max_inferred_score": max(inferred_scores) if inferred_scores else 0.0,
    }


def compute_after_oracle_report_stats(qcn: dict) -> dict:
    """Compute after-oracle counters used in the text report."""
    total_domains = len(qcn)
    total_relations = 0
    observed_support_pos_relations = 0
    zero_support_relations = 0
    domains_with_observed_status = 0
    domains_with_none_support = 0

    for domain in qcn.values():
        entries = relation_entries(domain)
        total_relations += len(entries)

        domain_is_observed = domain_status(domain) == STATUS_OBSERVED
        has_none_support = not entries

        for payload in entries.values():
            support = payload.get(SUPPORT_KEY)

            if has_positive_support(support):
                observed_support_pos_relations += 1
            elif has_zero_support(support):
                zero_support_relations += 1

            if has_no_evidence(support):
                has_none_support = True

        if domain_is_observed:
            domains_with_observed_status += 1
        if has_none_support:
            domains_with_none_support += 1

    return {
        "total_domains": total_domains,
        "total_relations": total_relations,
        "observed_support_pos_relations": observed_support_pos_relations,
        "zero_support_relations": zero_support_relations,
        "domains_with_observed_status": domains_with_observed_status,
        "domains_with_none_support": domains_with_none_support,
    }


def compute_oracle_to_propagation_transition_stats(
    initial_qcn: dict,
    after_oracle_qcn: dict,
    after_propagation_qcn: dict,
) -> dict:
    """Compute transition counters between oracle and propagation stages."""
    none_after_oracle_total = 0
    none_to_inferred_count = 0
    observed_after_oracle_total = 0
    domains_without_positive_support_after_oracle = 0
    domains_without_positive_support_to_positive_after_propagation = 0

    for pair, oracle_domain in after_oracle_qcn.items():
        oracle_entries = relation_entries(oracle_domain)
        propagation_domain = after_propagation_qcn.get(pair, {})
        propagation_entries = relation_entries(propagation_domain)

        oracle_has_positive_support = domain_status(oracle_domain) == STATUS_OBSERVED
        propagation_has_positive_support = (
            domain_status(propagation_domain) == STATUS_OBSERVED
        )
        propagation_has_inference = any(
            INFERRED_SCORE_KEY in payload for payload in propagation_entries.values()
        )

        for relation, oracle_payload in oracle_entries.items():
            oracle_support = oracle_payload.get(SUPPORT_KEY)
            oracle_status = (
                STATUS_OBSERVED if has_positive_support(oracle_support) else None
            )
            propagation_payload = propagation_entries.get(relation)
            if propagation_payload is None:
                propagation_status = None
            elif INFERRED_SCORE_KEY in propagation_payload:
                propagation_status = STATUS_INFERRED
            elif propagation_has_positive_support and has_positive_support(
                propagation_payload.get(SUPPORT_KEY)
            ):
                propagation_status = STATUS_OBSERVED
            else:
                propagation_status = None

            if oracle_status is None:
                none_after_oracle_total += 1
                if propagation_status == STATUS_INFERRED:
                    none_to_inferred_count += 1

            if oracle_status == STATUS_OBSERVED:
                observed_after_oracle_total += 1

        if not oracle_has_positive_support:
            domains_without_positive_support_after_oracle += 1
            if propagation_has_inference:
                domains_without_positive_support_to_positive_after_propagation += 1

    return {
        "none_after_oracle_total": none_after_oracle_total,
        "none_to_inferred_count": none_to_inferred_count,
        "observed_after_oracle_total": observed_after_oracle_total,
        "initial_total_relations": count_total_relations(initial_qcn),
        "after_propagation_total_relations": count_total_relations(
            after_propagation_qcn
        ),
        "domains_without_positive_support_after_oracle": domains_without_positive_support_after_oracle,
        "domains_without_positive_support_to_positive_after_propagation": domains_without_positive_support_to_positive_after_propagation,
    }


def processReductionRates(initialQCN, QCNAfterOracle, QCNAfterPC):
    """
    Calcule le nombre cumulé de relations d'Allen dans trois QCN pour analyser les taux de réduction.

    :param initialQCN: QCN initial (domaines complets)
    :param QCNAfterOracle: QCN après interrogation de l'oracle
    :param QCNAfterPC: QCN après path consistency
    :return: dict contenant les statistiques de réduction
    """

    initial_count = count_total_relations(initialQCN)
    after_oracle_count = count_total_relations(QCNAfterOracle)
    after_pc_count = count_total_relations(QCNAfterPC)

    oracle_reduction = (
        ((initial_count - after_oracle_count) / initial_count * 100)
        if initial_count > 0
        else 0
    )
    pc_reduction = (
        ((after_oracle_count - after_pc_count) / after_oracle_count * 100)
        if after_oracle_count > 0
        else 0
    )
    total_reduction = (
        ((initial_count - after_pc_count) / initial_count * 100)
        if initial_count > 0
        else 0
    )

    reductionRates = {
        "initial_total_relations": initial_count,
        "after_oracle_total_relations": after_oracle_count,
        "after_pc_total_relations": after_pc_count,
        "oracle_reduction_percent": oracle_reduction,
        "pc_reduction_percent": pc_reduction,
        "total_reduction_percent": total_reduction,
        "num_constraints": len(initialQCN),
    }

    return reductionRates


def save_qcn_to_file(
    learning_result: dict,
    kg_type: str,
    reduced_composition: bool = True,
    mode: str = "conservative",
):
    """
    Sauvegarde les résultats de l'apprentissage dans 2 fichiers :
    1. Un fichier JSON avec les QCNs (initial, après oracle, après propagation)
    2. Un fichier texte avec les statistiques de réduction

    :param learning_result: dict contenant 'initial', 'after_oracle', 'after_propagation', 'stats'
    :param kg_type: type du knowledge graph
    """

    if mode not in {"conservative", "two_stage", "iterative", "fix_point", "strass"}:
        raise ValueError(f"Unknown propagation mode: {mode}")
    results_dir = results_root / kg_type / mode
    os.makedirs(results_dir, exist_ok=True)

    output_stem = build_output_stem(kg_type, reduced_composition)
    qcn_filename = f"{output_stem}.json"
    stats_filename = f"{output_stem}_stats.txt"
    qcn_path = results_dir / qcn_filename
    stats_path = results_dir / stats_filename

    def convert_qcn_for_json(qcn):
        """Convertit le QCN en format JSON-compatible"""
        return {f"{k[0]}___{k[1]}": v for k, v in qcn.items()}

    qcn_data = {
        "initial": convert_qcn_for_json(learning_result["initial"]),
        "after_oracle": convert_qcn_for_json(learning_result["after_oracle"]),
    }
    if "after_propagation" in learning_result:
        qcn_data["after_propagation"] = convert_qcn_for_json(
            learning_result["after_propagation"]
        )

    with qcn_path.open("w", encoding="UTF-8") as f:
        # Encode one domain at a time with the C JSON encoder. This bounds the
        # temporary string size and avoids millions of tiny pretty-print writes.
        f.write("{")
        for section_index, (section, domains) in enumerate(qcn_data.items()):
            if section_index:
                f.write(",")
            f.write(json.dumps(section) + ":{")
            for pair_index, (pair, domain) in enumerate(domains.items()):
                if pair_index:
                    f.write(",")
                f.write(json.dumps(pair, ensure_ascii=False) + ":")
                f.write(json.dumps(domain, separators=(",", ":"), ensure_ascii=False))
            f.write("}")
        f.write("}")

    print(f"✅ QCNs saved to: {qcn_path}")

    stats = learning_result.get("stats")
    oracle_stats = learning_result.get("oracle_stats", {})
    property_filter_stats = learning_result.get("property_filter_stats", {})
    qcn_stats = learning_result.get("qcn_stats", {})
    after_oracle_report_stats = learning_result.get("after_oracle_report_stats", {})
    after_propagation_qcn_stats = learning_result.get("after_propagation_qcn_stats", {})
    oracle_to_propagation_transition_stats = learning_result.get(
        "oracle_to_propagation_transition_stats", {}
    )
    timing = learning_result.get("timing", {})

    def pct(part: int, total: int) -> float:
        return (part / total * 100.0) if total > 0 else 0.0

    with stats_path.open("w", encoding="UTF-8") as f:
        f.write(f"# Reduction statistics - Knowledge Graph: {kg_type}\n")
        f.write("# Mode: Learner2 (threshold-free)\n")
        f.write("#" + "=" * 70 + "\n\n")

        if stats is not None:
            f.write("## TOTAL NUMBER OF ALLEN RELATIONS\n")
            f.write(
                f"Initial (before learning)        : {stats['initial_total_relations']:>8}\n"
            )
            f.write(
                f"After Oracle                     : {stats['after_oracle_total_relations']:>8}\n"
            )
            f.write(
                f"After Path Consistency           : {stats['after_pc_total_relations']:>8}\n"
            )
            f.write("\n")

            f.write("## REDUCTION RATES\n")
            f.write(
                f"Reduction by Oracle              : {stats['oracle_reduction_percent']:>7.2f}%\n"
            )
            f.write(
                f"Reduction by Path Consistency    : {stats['pc_reduction_percent']:>7.2f}%\n"
            )
            f.write(
                f"Total Reduction                  : {stats['total_reduction_percent']:>7.2f}%\n"
            )
            f.write("\n")

            f.write("## ADDITIONAL INFORMATION\n")
            f.write(
                f"Number of constraints (pairs)    : {stats['num_constraints']:>8}\n"
            )

            avg_initial = (
                stats["initial_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            avg_oracle = (
                stats["after_oracle_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            avg_pc = (
                stats["after_pc_total_relations"] / stats["num_constraints"]
                if stats["num_constraints"] > 0
                else 0
            )
            f.write("Average relations per constraint\n")
            f.write(f"  - Initial                      : {avg_initial:>7.2f}\n")
            f.write(f"  - After Oracle                 : {avg_oracle:>7.2f}\n")
            f.write(f"  - After PC                     : {avg_pc:>7.2f}\n")
            if "cpu_time_seconds" in stats:
                f.write("\n")
                f.write("## PERFORMANCE\n")
                f.write(
                    f"CPU time (learner2)              : {stats['cpu_time_seconds']:>8.4f} s\n"
                )
        else:
            f.write("## AVAILABLE STATISTICS (ORACLE PHASE)\n")
            if qcn_stats:
                f.write(
                    f"Domains (pairs)                 : {qcn_stats.get('num_domains', 0):>8}\n"
                )
                f.write(
                    f"Total relations                 : {qcn_stats.get('total_relations', 0):>8}\n"
                )
                f.write(
                    "Observed rel. & support > 0    : "
                    f"{qcn_stats.get('count_observed', 0):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{qcn_stats.get('count_observed', 0)} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_observed_sup_pos', 0):.2f}%)\n"
                )
                f.write(
                    "Relations sans preuve (0, 0)  : "
                    f"{qcn_stats.get('count_none_none', 0):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{qcn_stats.get('count_none_none', 0)} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_none_none', 0):.2f}%)\n"
                )
                f.write(
                    "Relations testées absentes (0,n): "
                    f"{qcn_stats.get('count_none_zero', 0):>8}\n"
                )
                f.write(
                    "  -> proportion                : "
                    f"{qcn_stats.get('count_none_zero', 0)} / "
                    f"{qcn_stats.get('total_relations', 0)} "
                    f"({qcn_stats.get('percent_none_zero', 0):.2f}%)\n"
                )
                f.write(
                    "Inferred relations             : "
                    f"{qcn_stats.get('count_inferred', 0):>8}\n"
                )
                f.write(
                    "% domains without observed relation: "
                    f"{qcn_stats.get('count_domains_without_observed', 0)} / "
                    f"{qcn_stats.get('num_domains', 0)} "
                    f"({qcn_stats.get('percent_domains_without_observed', 0):.2f}%)\n"
                )
                f.write(
                    "% domains with >=1 observed rel. & sup>0: "
                    f"{qcn_stats.get('count_domains_with_observed_sup_pos', 0)} / "
                    f"{qcn_stats.get('num_domains', 0)} "
                    f"({qcn_stats.get('percent_domains_with_observed_sup_pos', 0):.2f}%)\n"
                )
                f.write("Support>0 per domain (avg/min/max/std) : ")
                f.write(
                    f"{qcn_stats.get('avg_support_per_domain', 0):.2f}/"
                    f"{qcn_stats.get('min_support_per_domain', 0)}/"
                    f"{qcn_stats.get('max_support_per_domain', 0)}/"
                    f"{qcn_stats.get('std_support_per_domain', 0):.2f}\n"
                )
            if oracle_stats:
                f.write(
                    f"Oracle calls                    : {oracle_stats.get('total_calls', 0):>8}\n"
                )
                f.write(
                    f"Total oracle phase cost (s)     : {oracle_stats.get('cost_of_the_oracle_phase', 0):>8.4f}\n"
                )
                f.write(
                    f"Oracle time min/max/avg (s)     : {oracle_stats.get('min_cpu_time', 0):.6f} / "
                    f"{oracle_stats.get('max_cpu_time', 0):.6f} / "
                    f"{oracle_stats.get('avg_cpu_time', 0):.6f}\n"
                )

        if oracle_stats:
            f.write("\n## TEMPS CPU - PHASE ORACLE\n")
            f.write(
                f"Appels oracle              : {int(oracle_stats.get('total_calls', 0))}\n"
            )
            f.write(
                f"Coût total phase oracle    : {float(oracle_stats.get('cost_of_the_oracle_phase', 0.0)):.4f} s\n"
            )
            f.write(
                "Temps oracle / appel - moy : "
                f"{float(oracle_stats.get('avg_cpu_time', 0.0)):.6f} s  "
                f"min={float(oracle_stats.get('min_cpu_time', 0.0)):.6f} s  "
                f"max={float(oracle_stats.get('max_cpu_time', 0.0)):.6f} s\n"
            )

        if property_filter_stats:
            f.write("\n## FILTRAGE PROPRIETES (TOP-K)\n")
            f.write(
                "Fraction top-k (proprietes)   : "
                f"{float(property_filter_stats.get('top_k_fraction', 0.0)):.2f}\n"
            )
            f.write(
                f"Proprietes gardees            : {int(property_filter_stats.get('kept_properties_count', 0))}\n"
            )
            f.write(
                f"Faits gardes                  : {int(property_filter_stats.get('kept_facts_count', 0))}\n"
            )

        if after_oracle_report_stats:
            oracle_total_rel = int(after_oracle_report_stats.get("total_relations", 0))
            oracle_total_dom = int(after_oracle_report_stats.get("total_domains", 0))
            observed_support_pos = int(
                after_oracle_report_stats.get("observed_support_pos_relations", 0)
            )
            zero_support = int(
                after_oracle_report_stats.get("zero_support_relations", 0)
            )
            dom_with_observed = int(
                after_oracle_report_stats.get("domains_with_observed_status", 0)
            )
            dom_with_none = int(
                after_oracle_report_stats.get("domains_with_none_support", 0)
            )

            f.write("\n## AFTER ORACLE\n")
            f.write(
                "Relations support > 0 (Observed) : "
                f"{observed_support_pos} / {oracle_total_rel} "
                f"({pct(observed_support_pos, oracle_total_rel):.2f}%)\n"
            )
            f.write(
                "Relations support == 0           : "
                f"{zero_support} / {oracle_total_rel} "
                f"({pct(zero_support, oracle_total_rel):.2f}%)\n"
            )
            f.write(
                "Domains with >=1 Observed status : "
                f"{dom_with_observed} / {oracle_total_dom} "
                f"({pct(dom_with_observed, oracle_total_dom):.2f}%)\n"
            )
            f.write(
                "Domains with None support        : "
                f"{dom_with_none} / {oracle_total_dom} "
                f"({pct(dom_with_none, oracle_total_dom):.2f}%)\n"
            )

        if after_propagation_qcn_stats:
            prop_total_rel = int(after_propagation_qcn_stats.get("total_relations", 0))
            prop_observed = int(after_propagation_qcn_stats.get("count_observed", 0))
            prop_inferred = int(after_propagation_qcn_stats.get("count_inferred", 0))
            inferred_score_avg = float(
                after_propagation_qcn_stats.get("avg_inferred_score", 0.0)
            )
            inferred_score_min = float(
                after_propagation_qcn_stats.get("min_inferred_score", 0.0)
            )
            inferred_score_max = float(
                after_propagation_qcn_stats.get("max_inferred_score", 0.0)
            )

            f.write("\n## AFTER PROPAGATION\n")
            f.write(
                "Relations status Observed        : "
                f"{prop_observed} / {prop_total_rel} "
                f"({pct(prop_observed, prop_total_rel):.2f}%)\n"
            )
            f.write(
                "Relations status Inferred        : "
                f"{prop_inferred} / {prop_total_rel} "
                f"({pct(prop_inferred, prop_total_rel):.2f}%)\n"
            )
            if prop_inferred:
                f.write(
                    "Scores inferred min/avg/max    : "
                    f"{inferred_score_min:.6g} / {inferred_score_avg:.6g} / "
                    f"{inferred_score_max:.6g}\n"
                )

        if oracle_to_propagation_transition_stats:
            none_after_oracle_total = int(
                oracle_to_propagation_transition_stats.get("none_after_oracle_total", 0)
            )
            none_to_inferred_count = int(
                oracle_to_propagation_transition_stats.get("none_to_inferred_count", 0)
            )
            _observed_after_oracle_total = int(
                oracle_to_propagation_transition_stats.get(
                    "observed_after_oracle_total", 0
                )
            )
            initial_total_relations = int(
                oracle_to_propagation_transition_stats.get("initial_total_relations", 0)
            )
            after_propagation_total_relations = int(
                oracle_to_propagation_transition_stats.get(
                    "after_propagation_total_relations", 0
                )
            )
            domains_without_positive_support_after_oracle = int(
                oracle_to_propagation_transition_stats.get(
                    "domains_without_positive_support_after_oracle", 0
                )
            )
            domains_without_positive_support_to_positive_after_propagation = int(
                oracle_to_propagation_transition_stats.get(
                    "domains_without_positive_support_to_positive_after_propagation",
                    0,
                )
            )

            f.write("\n## TRANSITIONS ORACLE -> PROPAGATION\n")
            f.write(
                "Relations None -> Inferred       : "
                f"{none_to_inferred_count} / {none_after_oracle_total} "
                f"({pct(none_to_inferred_count, none_after_oracle_total):.2f}%)\n"
            )
            f.write(f"Total relations in initial QCN  : {initial_total_relations}\n")
            f.write(
                "Total relations after propagation: "
                f"{after_propagation_total_relations}\n"
            )
            f.write(
                "Domains sans observation -> inférés: "
                f"{domains_without_positive_support_to_positive_after_propagation} / "
                f"{domains_without_positive_support_after_oracle} "
                f"({pct(domains_without_positive_support_to_positive_after_propagation, domains_without_positive_support_after_oracle):.2f}%)\n"
            )

        if timing:
            f.write("\n## CPU TIME PER STAGE\n")
            f.write("Initial stage (reference)       :   0.0000 s\n")
            f.write(
                f"Stage after_oracle (s)          : {timing.get('oracle_phase_cpu_time', 0):>8.4f}\n"
            )
            f.write(
                f"Stage after_propagation (s)     : {timing.get('propagate_and_filter_cpu_time', 0):>8.4f}\n"
            )
            f.write("\n## CUMULATIVE CPU TIME PER STAGE\n")
            f.write("Cumulative after initial (s)    :   0.0000 s\n")
            f.write(
                f"Cumulative after after_oracle(s): {timing.get('cumulative_after_oracle_cpu_time', 0):>8.4f}\n"
            )
            f.write(
                f"Cumulative after after_propagation(s): {timing.get('cumulative_after_propagation_cpu_time', 0):>8.4f}\n"
            )
            f.write("\n## GLOBAL CPU TIME\n")
            f.write(
                f"Total learner2 (s)              : {timing.get('learner_total_cpu_time', 0):>8.4f}\n"
            )

    print(f"✅ Statistics saved to: {stats_path}")


def learner_with_timeout(
    entity_network,
    entities,
    properties,
    kgName,
    timeout=600,
    domain_reliability_threshold=0,
    relation_Occurence_threshold=0,
    reduced_composition: bool = True,
    mode: str = "conservative",
    max_stages: int | None = None,
):
    """
    Lance l'apprentissage avec timeout et sauvegarde les résultats.

    :param entity_network: réseau de contraintes initial (QCN)
    :param entities: dictionnaire des entités
    :param properties: ensemble/liste des propriétés
    :param kgName: type du knowledge graph
    :param timeout: timeout en secondes
    :param relation_Occurence_threshold: seuil d'occurrence d'une relation pour décider si elle peut participer à l'inférence
    :param domain_reliability_threshold: seuil de support pour décider si un domaine peut participer à l'inférence
    """
    with mp.Pool(processes=1) as pool:
        result = pool.apply_async(
            learner2,
            (
                entity_network,
                entities,
                properties,
                domain_reliability_threshold,
                relation_Occurence_threshold,
                reduced_composition,
                mode,
                max_stages,
            ),
        )
        try:
            learning_result = result.get(timeout=timeout)
            if learning_result is not None:
                save_qcn_to_file(learning_result, kgName, reduced_composition, mode)
            return learning_result
        except mp.TimeoutError:
            print(f"⏱️ Timeout : calcul interrompu après {timeout / 60} minutes")
            pool.terminate()
            pool.join()
            print("⚠️ Apprentissage interrompu - aucun résultat sauvegardé")
            return None


def Query(Vocabulary, entities, entity_property_index=None):
    """
    Vérifié
    Docstring for Query

    :param Vocabulary: Description
    :param entities: Description
    :param threshold: Description
    :return: Description
    :rtype: bool
    """

    p1, p2 = Vocabulary

    allenrelMatching: dict[str, int] = {}

    for r in ALLEN_RELATIONS:
        allenrelMatching[r] = 0
    nbr_total = 0

    if entity_property_index is None:
        p1_entities = {
            head: tuple(entity.triples_per_p[p1])
            for head, entity in entities.items()
            if p1 in entity.triples_per_p
        }
        p2_entities = {
            head: tuple(entity.triples_per_p[p2])
            for head, entity in entities.items()
            if p2 in entity.triples_per_p
        }
    else:
        p1_entities = entity_property_index.get(p1, {})
        p2_entities = entity_property_index.get(p2, {})

    for head in p1_entities.keys() & p2_entities.keys():
        p1_data = p1_entities[head]
        p2_data = p2_entities[head]
        if (
            isinstance(p1_data, tuple)
            and isinstance(p2_data, tuple)
            and len(p1_data) == 2
            and len(p2_data) == 2
            and isinstance(p1_data[0], np.ndarray)
            and isinstance(p2_data[0], np.ndarray)
        ):
            nbr_total += add_vectorized_allen_relation_matches(
                allenrelMatching,
                p1_data[0],
                p1_data[1],
                p2_data[0],
                p2_data[1],
            )
        else:
            quadruplet_p1 = p1_data
            quadruplet_p2 = p2_data
            for qp1 in quadruplet_p1:
                for qp2 in quadruplet_p2:
                    nbr_total += 1
                    add_allen_relation_matches(allenrelMatching, qp1, qp2)

    if nbr_total == 0:
        return with_relation_data({relation: (0, 0) for relation in ALLEN_RELATIONS})
    else:
        relation_counts = {
            relation: (occurrences, nbr_total)
            for relation, occurrences in allenrelMatching.items()
        }
        return with_relation_data(relation_counts)


def analyze_qcn_scores(
    kg_type: str,
    reduced_composition: bool = True,
    mode: str = "conservative",
    learning_result: dict | None = None,
) -> dict | None:
    """
    Analyse un fichier JSON QCN pour compter les pourcentages de relations par catégorie de score.

    :param kg_type: type du knowledge graph (Q6256, Q215380, Q82955)
    :return: dict contenant les statistiques par QCN (after_oracle, after_propagation)
    """
    if learning_result is not None:
        qcn_data = learning_result
    else:
        qcn_path = (
            results_root
            / kg_type
            / mode
            / f"{build_output_stem(kg_type, reduced_composition)}.json"
        )

        if not qcn_path.exists():
            print(f"Fichier non trouvé: {qcn_path}")
            return None

        with qcn_path.open("r", encoding="UTF-8") as f:
            qcn_data = json.load(f)

    results = {}

    for qcn_name in ["after_oracle", "after_propagation"]:
        if qcn_name not in qcn_data:
            continue

        qcn = qcn_data[qcn_name]

        total_relations = 0
        count_repaired = 0
        count_no_entities = 0
        count_observed = 0
        count_zero = 0

        for domain in qcn.values():
            for score in relation_scores(domain).values():
                total_relations += 1

                if score == -1.0:
                    count_repaired += 1
                elif score == -2.0:
                    count_no_entities += 1
                elif score > 0:
                    count_observed += 1
                elif score == 0:
                    count_zero += 1

        if total_relations > 0:
            results[qcn_name] = {
                "total_relations": total_relations,
                "repaired_count": count_repaired,
                "repaired_percent": (count_repaired / total_relations) * 100,
                "no_entities_count": count_no_entities,
                "no_entities_percent": (count_no_entities / total_relations) * 100,
                "observed_count": count_observed,
                "observed_percent": (count_observed / total_relations) * 100,
                "zero_count": count_zero,
                "zero_percent": (count_zero / total_relations) * 100,
            }

    return results


def print_score_analysis(
    kg_type: str,
    reduced_composition: bool = True,
    mode: str = "conservative",
    learning_result: dict | None = None,
) -> None:
    """
    Affiche l'analyse des scores pour un KG donné.

    :param kg_type: type du knowledge graph
    """
    results = analyze_qcn_scores(kg_type, reduced_composition, mode, learning_result)

    if not results:
        return

    print(f"\n{'=' * 70}")
    print(f"ANALYSE DES SCORES - Knowledge Graph: {kg_type}")
    print(f"{'=' * 70}\n")

    for qcn_name, stats in results.items():
        print(f"📊 {qcn_name.upper().replace('_', ' ')}")
        print(f"   Total de relations: {stats['total_relations']}")
        print(
            f"   Relations observées (score > 0):  {stats['observed_count']:>6} ({stats['observed_percent']:>6.2f}%)"
        )
        print(
            f"   Relations réparées (score = -1): {stats['repaired_count']:>6} ({stats['repaired_percent']:>6.2f}%)"
        )
        print(
            f"   Aucune entité (score = -2):      {stats['no_entities_count']:>6} ({stats['no_entities_percent']:>6.2f}%)"
        )
        if stats["zero_count"] > 0:
            print(
                f"   Score zéro (score = 0):           {stats['zero_count']:>6} ({stats['zero_percent']:>6.2f}%)"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QCN Generator 2 - Apprentissage de contraintes temporelles (threshold-free)"
    )
    parser.add_argument("kg", type=str, help="Knowledge Graph (Q6256, Q215380, Q82955)")
    parser.add_argument(
        "occurences",
        type=float,
        help=(
            "Seuil de support pour permettre à une relation de participer "
            "à l'inférence (ex. 0.1 pour une proportion de 10 pour cent)"
        ),
    )
    parser.add_argument(
        "nbr_total_min",
        type=int,
        help="Nombre minimum de fois qu'une paire de prpopriétés doit apparaître pour être considérée (ex: 10)",
    )
    parser.add_argument(
        "timeout", type=int, help="Timeout en secondes (ex: 600 pour 10 min)"
    )

    parser.add_argument(
        "--reduced-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignorer les compositions produisant les 13 relations d’Allen.",
    )
    parser.add_argument(
        "--propagation-mode",
        choices=("conservative", "two_stage", "iterative", "fix_point", "strass"),
        default="conservative",
        help="Propagation conservative, two_stage ou iterative (alias : fix_point).",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=None,
        help="Nombre maximal de stages Strass (défaut : jusqu’à stabilisation).",
    )
    parser.add_argument(
        "--oracle-cache-dir",
        type=Path,
        default=None,
        help="Cache privé de préparation de l’oracle pour une campagne.",
    )
    parser.add_argument(
        "--_direct-learning", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.max_stages is not None and args.max_stages < 1:
        parser.error("--max-stages doit être positif")
    kg_name = args.kg
    occurences_threshold = args.occurences
    nbr_total_min = args.nbr_total_min
    timeout_seconds = args.timeout

    if not args._direct_learning:
        # Let one child own data loading, learning and saving; no large
        # Entity/QCN payload is transferred just to enforce a timeout.
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tclkg.qcn_generator2",
                    *sys.argv[1:],
                    "--_direct-learning",
                ],
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"Timeout après {timeout_seconds}s", flush=True)
            raise SystemExit(124) from None
        if completed.returncode:
            raise SystemExit(completed.returncode)
        return

    print(
        "-----------------------QCNGenerator2.py (threshold-free)--------------------------"
    )
    print(f"Knowledge Graph: {kg_name}")
    print(f"Property support threshold: {occurences_threshold:.2%}")
    print(f"Minimum total occurrences: {nbr_total_min}")
    print(f"Reduced composition: {args.reduced_composition}")
    print(f"Propagation mode: {args.propagation_mode}")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds / 60:.1f} min)")
    print(f"{'=' * 70}")

    cache_path = (
        get_oracle_cache_path(
            kg_name, ROOT_DATA, args.oracle_cache_dir, PROPERTY_SUPPORT_THRESHOLD
        )
        if args.oracle_cache_dir is not None
        else None
    )
    prepared = read_oracle_cache(cache_path) if cache_path is not None else None
    if prepared is None:
        entities, initial_properties = read_KG(kg_name)
        entity_network = build_entity_network(
            properties=initial_properties,
            default_constraints=ALLEN_RELATIONS,
            top_k_fraction=PROPERTY_SUPPORT_THRESHOLD,
            entities=entities,
        )
        properties = sorted({p for pair in entity_network for p in pair})
        kept_properties = set(properties)
        kept_facts_count = sum(
            len(triples)
            for entity in entities.values()
            for prop, triples in entity.triples_per_p.items()
            if prop in kept_properties
        )
        print(
            f"Top {PROPERTY_SUPPORT_THRESHOLD:.0%} properties by quadruplet count: "
            f"{len(initial_properties)} -> {len(properties)} properties kept"
        )
    else:
        entities = {}
        entity_network = {}
        properties = prepared["properties"]
        kept_facts_count = prepared["kept_facts_count"]
        print(f"Oracle cache hit: {cache_path}; {len(properties)} properties kept")

    print(
        f"Facts kept after top-{PROPERTY_SUPPORT_THRESHOLD:.0%} filter: {kept_facts_count}"
    )

    print("\n🚀 Début de l'apprentissage des contraintes temporelles...")
    learned_entity_network = learner2(
        entity_network,
        entities,
        properties,
        domain_reliability_threshold=nbr_total_min,
        relation_Occurence_threshold=occurences_threshold,
        reduced_composition=args.reduced_composition,
        mode=args.propagation_mode,
        max_stages=args.max_stages,
        prepared_oracle=prepared,
        oracle_cache_path=cache_path,
    )

    if learned_entity_network is not None:
        save_qcn_to_file(
            learned_entity_network,
            kg_name,
            args.reduced_composition,
            args.propagation_mode,
        )
        print(f"✅ Apprentissage terminé avec succès pour {kg_name}")

        print_score_analysis(
            kg_name,
            args.reduced_composition,
            args.propagation_mode,
            learned_entity_network,
        )
    else:
        print(f"⚠️ Apprentissage interrompu par timeout pour {kg_name}")

    print(f"{'=' * 70}")
    print("🏁 EXPÉRIENCE TERMINÉE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
