import argparse
import copy
import csv
import re
import statistics
import json
import math
from collections.abc import Mapping
from collections import deque
import os
from itertools import combinations
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import allen_relations as AllenR
from .allen_list import ALLEN_RELATIONS
from .generator_common import (
    RESULTS_ROOT as results_root,
    build_complete_network as build_entity_network,
    compose_allen_set as composition_Allen,
    converse_domain_set as converse_domain,
    converse_domain_scores,
    path_consistency_classique,
    read_kg as read_KG,
)


'''
This module is for running some axperiments
'''



def read_qcn_from_json(jsonfile: str) -> dict[tuple[str, str], dict[str, float]]:
    '''
    Read a QCN from a JSON file and return it as a dictionary.
    The JSON file should have the following structure:
    {
        "after_oracle": {
            "p1___p2": {"before": 1.0, "meets": 0.9},
            "p2___p1": {"after": 1.0, "met_by": 0.9},
            ...
        }
    }
    '''
    json_path = Path(jsonfile)
    if not json_path.is_absolute():
        json_path = results_root / json_path

    if not json_path.exists():
        raise FileNotFoundError(f"QCN JSON file not found: {json_path}")

    with json_path.open("r", encoding="UTF-8") as handle:
        qcn_data = json.load(handle)

    after_oracle = qcn_data.get("after_oracle")
    if after_oracle is None:
        raise ValueError(f"Missing 'after_oracle' section in {json_path}")

    qcn: dict[tuple[str, str], dict[str, float]] = {}
    for pair_key, raw_domain in after_oracle.items():
        if "___" not in pair_key:
            raise ValueError(f"Invalid QCN pair key: {pair_key}")
        property1, property2 = pair_key.split("___", 1)

        if isinstance(raw_domain, list):
            domain = {relation: 0.0 for relation in raw_domain}
        elif isinstance(raw_domain, dict) and isinstance(
            raw_domain.get("relations"), Mapping
        ):
            domain = {}
            for relation, payload in raw_domain["relations"].items():
                if not isinstance(payload, Mapping):
                    raise TypeError(
                        f"Invalid relation payload for pair {pair_key}: {relation}"
                    )
                support = payload.get("support")
                if (
                    not isinstance(support, (tuple, list))
                    or len(support) != 2
                    or not all(isinstance(value, int) for value in support)
                    or support[1] <= 0
                ):
                    continue
                domain[relation] = support[0] / support[1]
        elif isinstance(raw_domain, dict):
            domain = {
                relation: float(score)
                for relation, score in raw_domain.items()
            }
        else:
            raise TypeError(
                f"Unsupported domain type for pair {pair_key}: {type(raw_domain).__name__}"
            )

        qcn[(property1, property2)] = domain

    return qcn


def domainswith1relations(jsonfile: str) -> Path:
    '''
    Create a text file listing each pair whose domain contains relations with score 1.0,
    alongside the number of such relations.
    '''
    qcn = read_qcn_from_json(jsonfile)

    json_path = Path(jsonfile)
    report_name = f"{json_path.stem}_domains_with_score_1.txt"
    report_path = results_root / report_name

    lines: list[str] = []
    total_pairs = 0
    for (property1, property2), domain in qcn.items():
        nb_relations_score_1 = sum(1 for score in domain.values() if score == 1.0)
        if nb_relations_score_1 > 0:
            total_pairs += 1
            lines.append(
                f"{property1}___{property2}\t{nb_relations_score_1}"
            )

    header = [
        "pair\tnumber_of_relations_with_score_1.0",
        f"total_pairs\t{total_pairs}",
        "",
    ]
    report_path.write_text("\n".join(header + lines) + "\n", encoding="UTF-8")
    return report_path


def _prepare_qcn_for_threshold(
    qcn: dict[tuple[str, str], dict[str, float]], threshold: float
) -> dict[tuple[str, str], dict[str, float]]:
    '''
    Ensure each domain has at least one relation with score >= threshold,
    otherwise reset it to all Allen relations with score 0.0.
    '''
    for _, domain in qcn.items():
        if not any(score >= threshold for score in domain.values()):
            for relation in ALLEN_RELATIONS:
                domain[relation] = 0.0
    return qcn


def _format_domain(domain: dict[str, float]) -> str:
    if not domain:
        return "{}"
    ordered = sorted(domain.items(), key=lambda item: item[0])
    return "{" + ", ".join(f"{rel}:{score:.6g}" for rel, score in ordered) + "}"


def incoherence_report(
    jsonfile: str,
    threshold: float = 1.0,
    max_triplets: int = 300,
    trace_tail: int = 30,
) -> Path:
    '''
    Generate a report of contradictory triplets and the propagation path that
    leads to the first detected collapse during path consistency.
    '''
    initial_qcn = read_qcn_from_json(jsonfile)
    qcn = _prepare_qcn_for_threshold(copy.deepcopy(initial_qcn), threshold)
    properties = sorted({p for (x, y) in qcn.keys() for p in (x, y)})

    contradictory_triplets: list[tuple[str, str, str, set[str], set[str], set[str]]] = []
    for i in properties:
        for j in properties:
            if j == i:
                continue
            for k in properties:
                if k == i or k == j:
                    continue
                if (i, j) not in qcn or (j, k) not in qcn or (i, k) not in qcn:
                    continue
                domain_ij = set(qcn[(i, j)].keys())
                domain_jk = set(qcn[(j, k)].keys())
                domain_ik = set(qcn[(i, k)].keys())
                composed = composition_Allen(domain_ij, domain_jk)
                if not (domain_ik & composed):
                    contradictory_triplets.append((i, j, k, domain_ij, domain_jk, domain_ik))
                    if len(contradictory_triplets) >= max_triplets:
                        break
            if len(contradictory_triplets) >= max_triplets:
                break
        if len(contradictory_triplets) >= max_triplets:
            break

    qcn_pc = copy.deepcopy(qcn)
    queue = deque(qcn_pc.keys())
    trace_events: list[str] = []
    collapse_info: dict[str, Any] | None = None
    step = 0

    while queue and collapse_info is None:
        i, j = queue.popleft()
        for k in properties:
            if k == i or k == j:
                continue
            if (i, k) not in qcn_pc or (k, j) not in qcn_pc:
                raise ValueError(
                    f"Missing arc in qcn_copy during propagation: {(i, k)} or {(k, j)}"
                )

            step += 1

            dik = qcn_pc[(i, k)]
            composed_ij_jk = composition_Allen(set(qcn_pc[(i, j)]), set(qcn_pc[(j, k)]))
            new_dik = {relation: 0.0 for relation in dik if relation in composed_ij_jk}

            if new_dik != dik:
                if not new_dik:
                    collapse_info = {
                        "collapsed_pair": (i, k),
                        "path": (i, j, k),
                        "left_pair": (i, j),
                        "right_pair": (j, k),
                        "target_before": dict(dik),
                        "composed": sorted(composed_ij_jk),
                        "step": step,
                    }
                    break
                qcn_pc[(i, k)] = new_dik
                qcn_pc[(k, i)] = converse_domain_scores(new_dik)
                queue.append((i, k))
                trace_events.append(
                    f"step {step}: reduce ({i}, {k}) via ({i}, {j}) o ({j}, {k}) | "
                    f"before={sorted(dik.keys())} after={sorted(new_dik.keys())}"
                )

            dkj = qcn_pc[(k, j)]
            composed_ki_ij = composition_Allen(set(qcn_pc[(k, i)]), set(qcn_pc[(i, j)]))
            new_dkj = {relation: 0.0 for relation in dkj if relation in composed_ki_ij}

            if new_dkj != dkj:
                if not new_dkj:
                    collapse_info = {
                        "collapsed_pair": (k, j),
                        "path": (k, i, j),
                        "left_pair": (k, i),
                        "right_pair": (i, j),
                        "target_before": dict(dkj),
                        "composed": sorted(composed_ki_ij),
                        "step": step,
                    }
                    break
                qcn_pc[(k, j)] = new_dkj
                qcn_pc[(j, k)] = converse_domain_scores(new_dkj)
                queue.append((k, j))
                trace_events.append(
                    f"step {step}: reduce ({k}, {j}) via ({k}, {i}) o ({i}, {j}) | "
                    f"before={sorted(dkj.keys())} after={sorted(new_dkj.keys())}"
                )

    json_path = Path(jsonfile)
    report_path = results_root / f"{json_path.stem}_incoherence_report.txt"

    lines: list[str] = [
        f"source_json\t{jsonfile}",
        f"threshold\t{threshold}",
        f"contradictory_triplets_found\t{len(contradictory_triplets)}",
        f"triplets_truncated\t{len(contradictory_triplets) >= max_triplets}",
        "",
        "[A] Contradictory triplets (initial domains after threshold preparation)",
    ]

    if contradictory_triplets:
        for idx, (i, j, k, _, _, _) in enumerate(contradictory_triplets, 1):
            lines.append(f"{idx}. path: {i} -> {j} -> {k}")
            lines.append(f"   domain({i}, {j}) initial: {_format_domain(initial_qcn[(i, j)])}")
            lines.append(f"   domain({j}, {k}) initial: {_format_domain(initial_qcn[(j, k)])}")
            lines.append(f"   domain({i}, {k}) initial: {_format_domain(initial_qcn[(i, k)])}")
            lines.append("")
    else:
        lines.append("No contradictory triplet found at initialization.")
        lines.append("")

    lines.append("[B] Path consistency collapse path")
    if collapse_info is None:
        lines.append("No collapse detected during path consistency propagation.")
    else:
        p1, p2, p3 = collapse_info["path"]
        c1, c2 = collapse_info["collapsed_pair"]
        l1, l2 = collapse_info["left_pair"]
        r1, r2 = collapse_info["right_pair"]
        lines.append(f"collapse_step: {collapse_info['step']}")
        lines.append(f"collapse_pair: ({c1}, {c2})")
        lines.append(f"incoherence_path: {p1} -> {p2} -> {p3}")
        lines.append(f"left_arc: ({l1}, {l2})")
        lines.append(f"right_arc: ({r1}, {r2})")
        lines.append(
            "composed_relations(left_arc o right_arc): "
            + str(collapse_info["composed"])
        )
        lines.append(
            "target_domain_before_collapse: "
            + _format_domain(collapse_info["target_before"])
        )
        lines.append("")
        lines.append("Initial domains of collapse triplet:")
        lines.append(f"domain({l1}, {l2}) initial: {_format_domain(initial_qcn[(l1, l2)])}")
        lines.append(f"domain({r1}, {r2}) initial: {_format_domain(initial_qcn[(r1, r2)])}")
        lines.append(f"domain({c1}, {c2}) initial: {_format_domain(initial_qcn[(c1, c2)])}")
        lines.append("")
        lines.append(f"Recent propagation steps before collapse (last {trace_tail}):")
        if trace_events:
            lines.extend(trace_events[-trace_tail:])
        else:
            lines.append("No prior reduction before collapse.")

    report_path.write_text("\n".join(lines) + "\n", encoding="UTF-8")
    return report_path


def property_implied_quadruplets(kg_name: str) -> None:
    '''
    Plot a histogram: for each property, the number of quadruplets (triples) it is involved in.
    kg_name is the KG identifier (e.g. "Q6256") as used by read_KG.
    Properties are numbered 1..n (sorted by count desc). The plot is saved locally.
    '''
    entities, _ = read_KG(kg_name, False)

    histogram: dict[str, int] = {}
    for entity in entities.values():
        for prop, triples in entity.triples_per_p.items():
            histogram[prop] = histogram.get(prop, 0) + len(triples)

    sorted_items = sorted(histogram.items(), key=lambda x: -x[1])
    indices = list(range(1, len(sorted_items) + 1))
    counts = [item[1] for item in sorted_items]
    labels = [f"{i}. {item[0].split('/')[-1]}" for i, item in enumerate(sorted_items, 1)]

    fig, ax = plt.subplots(figsize=(max(10, len(indices) * 0.5), 6))
    ax.bar(indices, counts)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Property (ranked by quadruplet count)")
    ax.set_ylabel("Number of quadruplets (log scale)")
    ax.set_title(f"Quadruplets per property — {kg_name}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = Path(__file__).resolve().parents[2] / "Results" / f"quadruplets_histogram_{kg_name}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def _extract_supports_from_domain(raw_domain: Any) -> list[float]:
    """Extract non-None numeric supports in [0, 1] from one domain representation."""
    supports: list[float] = []

    if isinstance(raw_domain, Mapping):
        structured_relations = raw_domain.get("relations")
        if isinstance(structured_relations, Mapping):
            for payload in structured_relations.values():
                if isinstance(payload, Mapping):
                    support = payload.get("support")
                    if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                        supports.append(float(support))
            return supports


        for value in raw_domain.values():
            if value is not None and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                supports.append(float(value))
            elif isinstance(value, Mapping):
                support = value.get("support")
                if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                    supports.append(float(support))

    return supports


def _build_histogram_counts(values: list[float], bin_width: float) -> tuple[list[float], list[int]]:
    """Build histogram counts on [0, 1] with fixed-width bins."""
    nbins = int(round(1.0 / bin_width))
    edges = [round(i * bin_width, 10) for i in range(nbins + 1)]
    counts = [0 for _ in range(nbins)]

    for value in values:
        if value < 0.0 or value > 1.0:
            continue
        index = min(int(value / bin_width), nbins - 1)
        counts[index] += 1

    return edges, counts


def _extract_inferred_only_supports(
    oracle_domain: Any, propagation_domain: Any
) -> list[float]:
    """
    Extract supports of relations that exist only in propagation_domain
    (inferred only - not present in oracle_domain).
    """
    supports: list[float] = []
    if not isinstance(oracle_domain, Mapping) or not isinstance(propagation_domain, Mapping):
        return supports


    oracle_relations = set()
    propagation_relations = set()


    oracle_structured = oracle_domain.get("relations")
    if isinstance(oracle_structured, Mapping):
        oracle_relations.update(oracle_structured.keys())
    else:

        oracle_relations.update(k for k in oracle_domain.keys() if k != "relations")


    propagation_structured = propagation_domain.get("relations")
    if isinstance(propagation_structured, Mapping):
        for relation, payload in propagation_structured.items():
            if relation not in oracle_relations and isinstance(payload, Mapping):
                support = payload.get("support")
                if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                    supports.append(float(support))
    else:

        for relation, value in propagation_domain.items():
            if relation == "relations":
                continue
            if relation not in oracle_relations:
                if value is not None and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    supports.append(float(value))
                elif isinstance(value, Mapping):
                    support = value.get("support")
                    if support is not None and isinstance(support, (int, float)) and 0.0 <= float(support) <= 1.0:
                        supports.append(float(support))

    return supports


def _domain_has_positive_support(raw_domain: Any) -> bool:
    """Return True if a domain has at least one non-None support strictly > 0."""
    if not isinstance(raw_domain, Mapping):
        return False

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                continue
            support = payload.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                return True
        return False


    for value in raw_domain.values():
        if value is not None and isinstance(value, (int, float)) and float(value) > 0.0:
            return True
        if isinstance(value, Mapping):
            support = value.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                return True
    return False


def _count_positive_relations_in_domain(raw_domain: Any) -> int:
    """Count relations with non-None support strictly > 0 in one domain."""
    if not isinstance(raw_domain, Mapping):
        return 0

    count = 0
    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                continue
            support = payload.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                count += 1
        return count


    for value in raw_domain.values():
        if value is not None and isinstance(value, (int, float)) and float(value) > 0.0:
            count += 1
        elif isinstance(value, Mapping):
            support = value.get("support")
            if support is not None and isinstance(support, (int, float)) and float(support) > 0.0:
                count += 1
    return count


def _extract_statuses_from_domain(raw_domain: Any) -> list[str]:
    """Extract relation statuses from one domain (structured and legacy-safe)."""
    statuses: list[str] = []
    if not isinstance(raw_domain, Mapping):
        return statuses

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        for payload in structured_relations.values():
            if not isinstance(payload, Mapping):
                statuses.append("other")
                continue
            status = payload.get("status")
            if status is None:
                statuses.append("none")
            elif isinstance(status, str):
                normalized = status.strip().lower()
                if normalized == "observed":
                    statuses.append("observed")
                elif normalized in {"inferred", "infered"}:
                    statuses.append("inferred")
                elif normalized == "repaired":
                    statuses.append("repaired")
                else:
                    statuses.append("other")
            else:
                statuses.append("other")
        return statuses


    if isinstance(raw_domain, Mapping):
        for _ in raw_domain.values():
            statuses.append("none")
    return statuses


def _normalize_transition_status(status: Any) -> str | None:
    """Normalize raw status labels to the four publication heatmap categories."""
    if status is None:
        return "none"
    if not isinstance(status, str):
        return None

    normalized = status.strip().lower()
    if normalized == "observed":
        return "observed"
    if normalized in {"inferred", "infered"}:
        return "inferred"
    if normalized == "none":
        return "none"
    return None


def _extract_relation_status_map(raw_domain: Any) -> dict[str, str | None]:
    """Extract a relation -> normalized status mapping from one domain."""
    if not isinstance(raw_domain, Mapping):
        return {}

    structured_relations = raw_domain.get("relations")
    if isinstance(structured_relations, Mapping):
        status_map: dict[str, str | None] = {}
        for relation, payload in structured_relations.items():
            if isinstance(payload, Mapping):
                status_map[str(relation)] = _normalize_transition_status(payload.get("status"))
            else:
                status_map[str(relation)] = None
        return status_map

    return {
        str(relation): _normalize_transition_status(None)
        for relation in raw_domain.keys()
    }


def count_domains_without_positive_to_positive(
    after_oracle_qcn: Mapping[str, Any],
    after_propagation_qcn: Mapping[str, Any],
) -> dict[str, int | float]:
    """
    Count domains that have no positive support in after_oracle and gain
    at least one positive support in after_propagation.

    The two QCN mappings are expected to use the same pair keys
    (e.g. "p1___p2"), but missing keys in after_propagation are tolerated.
    """
    domains_without_positive_after_oracle = 0
    domains_became_positive_after_propagation = 0

    for pair_key, oracle_domain in after_oracle_qcn.items():
        oracle_has_positive = _domain_has_positive_support(oracle_domain)
        propagation_domain = after_propagation_qcn.get(pair_key, {})
        propagation_has_positive = _domain_has_positive_support(propagation_domain)

        if not oracle_has_positive:
            domains_without_positive_after_oracle += 1
            if propagation_has_positive:
                domains_became_positive_after_propagation += 1

    ratio = (
        domains_became_positive_after_propagation
        / domains_without_positive_after_oracle
        * 100.0
        if domains_without_positive_after_oracle > 0
        else 0.0
    )

    return {
        "domains_without_positive_after_oracle": domains_without_positive_after_oracle,
        "domains_became_positive_after_propagation": domains_became_positive_after_propagation,
        "percentage_became_positive": ratio,
    }


def compute_iswc_after_propagation_positive_relation_stats() -> dict[str, dict[str, float]]:
    """
    Compute min/max/average number of positive relations per domain in
    after_propagation, for each TKG under Results/XXXX-1.
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    counts_per_kg: dict[str, list[int]] = {}
    files_seen_per_kg: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation")
        if not isinstance(after_propagation, Mapping):
            continue

        counts_per_kg.setdefault(kg_name, [])
        files_seen_per_kg[kg_name] = files_seen_per_kg.get(kg_name, 0) + 1

        for domain in after_propagation.values():
            counts_per_kg[kg_name].append(_count_positive_relations_in_domain(domain))

    if not counts_per_kg:
        raise ValueError("No valid after_propagation sections found in XXXX-1 JSON files")

    summary: dict[str, dict[str, float]] = {}
    for kg_name in sorted(counts_per_kg):
        domain_counts = counts_per_kg[kg_name]
        n = len(domain_counts)
        avg = (sum(domain_counts) / n) if n > 0 else 0.0

        summary[kg_name] = {
            "domains_count": float(n),
            "positive_relations_per_domain_min": float(min(domain_counts)) if n > 0 else 0.0,
            "positive_relations_per_domain_max": float(max(domain_counts)) if n > 0 else 0.0,
            "positive_relations_per_domain_avg": avg,
            "json_files_seen": float(files_seen_per_kg.get(kg_name, 0)),
        }

    for kg_name in sorted(summary):
        stats = summary[kg_name]
        print(
            f"{kg_name}: min={int(stats['positive_relations_per_domain_min'])} "
            f"max={int(stats['positive_relations_per_domain_max'])} "
            f"moy={stats['positive_relations_per_domain_avg']:.4f} "
            f"(domains={int(stats['domains_count'])})"
        )

    return summary


def plot_iswc_domain_size_subplots() -> dict[str, dict[str, Any]]:
    """
    Generate, for each TKG under Results/XXXX-1, a 2-subplot figure with:
    1) Distribution of domain sizes for domains that contain at least one
       observed relation in after_propagation.
     2) Distribution of inferred-only domain sizes in after_propagation,
         restricted to domains with no observed information.

    A domain size is the number of relations in the domain.
    Inferred-only size counts relations with status "inferred" in
    after_propagation that are absent (or status none) in after_oracle.
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    observed_sizes_per_kg: dict[str, list[int]] = {}
    inferred_only_sizes_per_kg: dict[str, list[int]] = {}
    files_seen_per_kg: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            continue

        observed_sizes_per_kg.setdefault(kg_name, [])
        inferred_only_sizes_per_kg.setdefault(kg_name, [])
        files_seen_per_kg[kg_name] = files_seen_per_kg.get(kg_name, 0) + 1

        for pair_key, propagation_domain in after_propagation.items():
            if not isinstance(propagation_domain, Mapping):
                continue

            propagation_relations = propagation_domain.get("relations")
            if not isinstance(propagation_relations, Mapping):
                continue

            oracle_domain = after_oracle.get(pair_key, {})
            oracle_relations = (
                oracle_domain.get("relations", {})
                if isinstance(oracle_domain, Mapping)
                else {}
            )
            if not isinstance(oracle_relations, Mapping):
                oracle_relations = {}

            domain_size = len(propagation_relations)
            has_observed = False
            inferred_only_size = 0

            for relation, prop_payload in propagation_relations.items():
                if not isinstance(prop_payload, Mapping):
                    continue

                prop_status = _normalize_transition_status(prop_payload.get("status"))
                if prop_status == "observed":
                    has_observed = True

                oracle_payload = oracle_relations.get(relation)
                oracle_status = (
                    _normalize_transition_status(oracle_payload.get("status"))
                    if isinstance(oracle_payload, Mapping)
                    else None
                )
                if prop_status == "inferred" and (
                    (not isinstance(oracle_payload, Mapping))
                    or oracle_status in {None, "none"}
                ):
                    inferred_only_size += 1

            if has_observed:
                observed_sizes_per_kg[kg_name].append(domain_size)
            else:
                inferred_only_sizes_per_kg[kg_name].append(inferred_only_size)

    if not observed_sizes_per_kg and not inferred_only_sizes_per_kg:
        raise ValueError("No valid after_oracle/after_propagation data found in XXXX-1 JSON files")

    summary: dict[str, dict[str, Any]] = {}

    all_kgs = sorted(set(observed_sizes_per_kg.keys()) | set(inferred_only_sizes_per_kg.keys()))
    for kg_name in all_kgs:
        observed_sizes = observed_sizes_per_kg.get(kg_name, [])
        inferred_only_sizes = inferred_only_sizes_per_kg.get(kg_name, [])

        max_size = 0
        if observed_sizes:
            max_size = max(max_size, max(observed_sizes))
        if inferred_only_sizes:
            max_size = max(max_size, max(inferred_only_sizes))
        x_values = list(range(max_size + 1))

        observed_hist = [0] * (max_size + 1)
        inferred_only_hist = [0] * (max_size + 1)

        for size in observed_sizes:
            if 0 <= size <= max_size:
                observed_hist[size] += 1
        for size in inferred_only_sizes:
            if 0 <= size <= max_size:
                inferred_only_hist[size] += 1

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5.5))

        ax_left.bar(x_values, observed_hist, color="#2E86AB", width=0.85)
        ax_left.set_title(f"{kg_name} - Observed domains")
        ax_left.set_xlabel("Domain size (#relations)")
        ax_left.set_ylabel("Number of domains")
        ax_left.set_yscale("log")
        ax_left.set_xticks(x_values)
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)

        ax_right.bar(x_values, inferred_only_hist, color="#2CA02C", width=0.85)
        ax_right.set_title(f"{kg_name} - Inferred-only domains")
        ax_right.set_xlabel("Inferred-only size (#relations)")
        ax_right.set_ylabel("Number of domains")
        ax_right.set_yscale("log")
        ax_right.set_xticks(x_values)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)

        plt.suptitle(f"{kg_name} - Domain size distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()

        output_path = iswc_root / kg_name / f"domain_size_subplots_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"XXXX-1 domain-size subplots saved to {output_path}")

        summary[kg_name] = {
            "observed_domain_sizes_count": len(observed_sizes),
            "inferred_only_domain_sizes_count": len(inferred_only_sizes),
            "observed_min": min(observed_sizes) if observed_sizes else 0,
            "observed_max": max(observed_sizes) if observed_sizes else 0,
            "observed_avg": (sum(observed_sizes) / len(observed_sizes)) if observed_sizes else 0.0,
            "inferred_only_min": min(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_max": max(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_avg": (sum(inferred_only_sizes) / len(inferred_only_sizes)) if inferred_only_sizes else 0.0,
            "json_files_seen": files_seen_per_kg.get(kg_name, 0),
        }

    return summary


def plot_iswc_file_domain_size_histograms() -> dict[str, dict[str, Any]]:
    """
    For each JSON file under Results/XXXX-1, build one histogram of
    after_propagation domain sizes with bins 0..13.

    Bars represent all domains. A green hatched overlay marks domains that are
    inferred-only (no observed information and at least one inferred relation).

    Output file is saved next to each JSON input as:
    <json_stem>_domain_size_histogram.png
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    label_by_kg = {
        "Q6256": "Country",
        "Q215380": "Musical group",
        "Q82955": "Politician",
    }

    summary: dict[str, dict[str, Any]] = {}
    x_values = list(range(14))

    for json_path in json_files:
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = payload.get("after_propagation")
        if not isinstance(after_propagation, Mapping):
            continue

        total_hist = [0] * 14
        inferred_only_hist = [0] * 14

        for domain in after_propagation.values():
            if not isinstance(domain, Mapping):
                continue
            relations = domain.get("relations")
            if not isinstance(relations, Mapping):
                continue

            domain_size = len(relations)
            size_bin = domain_size if 0 <= domain_size <= 13 else 13
            total_hist[size_bin] += 1

            has_observed = False
            has_inferred = False
            for payload_rel in relations.values():
                if not isinstance(payload_rel, Mapping):
                    continue
                status = _normalize_transition_status(payload_rel.get("status"))
                if status == "observed":
                    has_observed = True
                if status == "inferred":
                    has_inferred = True

            if (not has_observed) and has_inferred:
                inferred_only_hist[size_bin] += 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        ax.bar(x_values, total_hist, width=0.85, color="#2E86AB", label="all domains")
        ax.bar(
            x_values,
            inferred_only_hist,
            width=0.85,
            facecolor="none",
            edgecolor="green",
            hatch="\\\\\\\\",
            linewidth=1.0,
            label="inferred-only domains",
        )

        rel_path = json_path.relative_to(iswc_root)
        kg_id = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem
        kg_label = label_by_kg.get(kg_id, kg_id)
        ax.set_title(f"{kg_label} ({kg_id}) - after_propagation domain-size distribution")
        ax.set_xlabel("Domain size (#relations)")
        ax.set_ylabel("Number of domains")
        ax.set_yscale("log")
        ax.set_xticks(x_values)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        output_path = json_path.with_name(f"{json_path.stem}_domain_size_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"XXXX-1 file histogram saved to {output_path}")

        summary[str(rel_path)] = {
            "total_domains": sum(total_hist),
            "inferred_only_domains": sum(inferred_only_hist),
            "total_hist": total_hist,
            "inferred_only_hist": inferred_only_hist,
            "output_file": str(output_path.relative_to(results_root.parent if results_root.parent.exists() else output_path.parent)),
        }

    return summary


def plot_iswc_status_transitions_stacked() -> dict[str, dict[str, dict[str, float]]]:
    """
    Plot stacked bar charts of status proportions for after_oracle and
    after_propagation, for each dataset under Results/XXXX-1.

    Returns per-dataset status proportions for both stages.
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    stages = ("after_oracle", "after_propagation")
    status_order = [
        "none",
        "observed",
        "inferred",
        "repaired",
        "other",
    ]
    status_labels = {
        "none": "none",
        "observed": "observed",
        "inferred": "inferred",
        "repaired": "repaired",
        "other": "other",
    }
    status_colors = {
        "none": "#9AA5B1",
        "observed": "#2E86AB",
        "inferred": "#F18F01",
        "repaired": "#C73E1D",
        "other": "#6C5B7B",
    }

    counts_per_dataset: dict[str, dict[str, dict[str, int]]] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        dataset = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        counts_per_dataset.setdefault(
            dataset,
            {
                stage: {status: 0 for status in status_order}
                for stage in stages
            },
        )

        for stage in stages:
            stage_domains = payload.get(stage)
            if not isinstance(stage_domains, Mapping):
                continue
            for domain in stage_domains.values():
                for status in _extract_statuses_from_domain(domain):
                    if status not in counts_per_dataset[dataset][stage]:
                        counts_per_dataset[dataset][stage]["other"] += 1
                    else:
                        counts_per_dataset[dataset][stage][status] += 1

    if not counts_per_dataset:
        raise ValueError("No valid stage data found in XXXX-1 JSON files")

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for dataset in sorted(counts_per_dataset.keys()):
        dataset_counts = counts_per_dataset[dataset]
        stage_props: dict[str, dict[str, float]] = {}

        for stage in stages:
            total = sum(dataset_counts[stage].values())
            if total == 0:
                stage_props[stage] = {status: 0.0 for status in status_order}
            else:
                stage_props[stage] = {
                    status: (dataset_counts[stage][status] / total) * 100.0
                    for status in status_order
                }

        summary[dataset] = stage_props


        fig, ax = plt.subplots(figsize=(9, 5.5))
        x_labels = ["after_oracle", "after_propagation"]
        x_pos = [0, 1]
        bottoms = [0.0, 0.0]

        for status in status_order:
            values = [
                stage_props["after_oracle"][status],
                stage_props["after_propagation"][status],
            ]
            ax.bar(
                x_pos,
                values,
                bottom=bottoms,
                color=status_colors[status],
                label=status_labels[status],
                width=0.6,
            )
            bottoms = [bottoms[i] + values[i] for i in range(2)]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("Proportion of relations (%)")
        ax.set_xlabel("Stage")
        ax.set_title(f"{dataset} - Status transitions (stacked)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        plt.tight_layout()
        output_path = iswc_root / dataset / f"status_transitions_stacked_{dataset}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Status transition plot saved to {output_path}")

    return summary


def plot_iswc_status_transition_heatmaps() -> dict[str, dict[str, Any]]:
    """
    Generate publication-ready heatmaps of relation-level status transitions
    from after_oracle to after_propagation for each XXXX-1 dataset.

    Rows correspond to the status before propagation and columns to the
    status after propagation. Cell values are proportions of relations (%).
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    dataset_display_names = {
        "Q6256": "Country",
        "Q215380": "Musical Group",
        "Q82955": "Politician",
    }
    status_order = ["none", "observed", "inferred"]
    status_labels = ["none", "observed", "inferred"]
    status_to_index = {status: index for index, status in enumerate(status_order)}

    transition_counts_per_dataset: dict[str, list[list[int]]] = {}
    matched_relations_per_dataset: dict[str, int] = {}
    skipped_relations_per_dataset: dict[str, int] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        dataset = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            continue

        if dataset not in transition_counts_per_dataset:
            transition_counts_per_dataset[dataset] = [
                [0 for _ in status_order] for _ in status_order
            ]
            matched_relations_per_dataset[dataset] = 0
            skipped_relations_per_dataset[dataset] = 0

        pair_keys = set(after_oracle.keys()) | set(after_propagation.keys())
        for pair_key in pair_keys:
            before_statuses = _extract_relation_status_map(after_oracle.get(pair_key, {}))
            after_statuses = _extract_relation_status_map(after_propagation.get(pair_key, {}))

            relation_names = set(before_statuses.keys()) | set(after_statuses.keys())
            for relation_name in relation_names:
                before_status = before_statuses.get(relation_name, "none")
                after_status = after_statuses.get(relation_name, "none")

                if before_status not in status_to_index or after_status not in status_to_index:
                    skipped_relations_per_dataset[dataset] += 1
                    continue

                transition_counts_per_dataset[dataset][status_to_index[before_status]][status_to_index[after_status]] += 1
                matched_relations_per_dataset[dataset] += 1

    if not transition_counts_per_dataset:
        raise ValueError("No valid after_oracle/after_propagation sections found in XXXX-1 JSON files")

    summary: dict[str, dict[str, Any]] = {}
    rc_params = {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    preferred_order = ["Q6256", "Q215380", "Q82955"]
    ordered_datasets = [
        dataset for dataset in preferred_order if dataset in transition_counts_per_dataset
    ]
    ordered_datasets.extend(
        sorted(
            dataset
            for dataset in transition_counts_per_dataset.keys()
            if dataset not in ordered_datasets
        )
    )

    for dataset in ordered_datasets:
        counts = transition_counts_per_dataset[dataset]
        matched_relations = matched_relations_per_dataset[dataset]
        skipped_relations = skipped_relations_per_dataset[dataset]
        if matched_relations == 0:
            percentages = [[0.0 for _ in status_order] for _ in status_order]
        else:
            percentages = [
                [cell_count * 100.0 / matched_relations for cell_count in row]
                for row in counts
            ]

        summary[dataset] = {
            "percentages": percentages,
            "counts": counts,
            "matched_relations": matched_relations,
            "skipped_relations": skipped_relations,
        }

    global_vmax = max(
        max(max(row) for row in summary[dataset]["percentages"])
        for dataset in ordered_datasets
    )
    if global_vmax <= 0.0:
        global_vmax = 1.0

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1,
            len(ordered_datasets),
            figsize=(4.2 * len(ordered_datasets), 4),
            constrained_layout=True,
            sharey=True,
        )
        if len(ordered_datasets) == 1:
            axes = [axes]

        image = None
        for index, dataset in enumerate(ordered_datasets):
            ax = axes[index]
            display_name = dataset_display_names.get(dataset, dataset)
            percentages = summary[dataset]["percentages"]

            image = ax.imshow(percentages, cmap="Blues", vmin=0.0, vmax=global_vmax)
            ax.set_xticks(range(len(status_labels)))
            ax.set_xticklabels(status_labels, rotation=0, ha="center")
            ax.set_title(display_name, pad=8)

            ax.set_yticks(range(len(status_labels)))
            if index == 0:
                ax.set_yticklabels(status_labels)
                ax.set_ylabel("Before propagation")
                ax.tick_params(axis="y", left=True, labelleft=True)
            else:

                ax.tick_params(axis="y", left=False, labelleft=False)

            local_max = max(max(row) for row in percentages) if percentages else 0.0
            threshold = local_max * 0.5
            for row_index, row in enumerate(percentages):
                for col_index, value in enumerate(row):
                    text_color = "white" if value > threshold and value > 0 else "black"
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color=text_color,
                    )

        fig.supxlabel("After propagation")

        if image is not None:
            colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, shrink=0.92)
            colorbar.set_label("Relations (%)", fontsize=10)
            colorbar.ax.tick_params(labelsize=9)

        combined_output_path = iswc_root / "status_transition_heatmaps_iswc.png"
        plt.savefig(combined_output_path, dpi=300)
        plt.close(fig)

    for dataset in ordered_datasets:
        summary[dataset]["output_path"] = str(combined_output_path)
        print(
            f"Status transition subplot included for {dataset} "
            f"(matched={summary[dataset]['matched_relations']}, "
            f"skipped={summary[dataset]['skipped_relations']})"
        )
    print(f"Combined status transition heatmaps saved to {combined_output_path}")

    return summary


def plot_iswc_after_oracle_positive_domain_proportions() -> dict[str, dict[str, float]]:
    """
    Read JSON files in Results/XXXX-1 and compute, for each KG, the proportion
    (0..100) of after_oracle domains having at least one support > 0.

    A common plot for all KGs is saved in Results/XXXX-1.
    """
    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    per_kg_counts: dict[str, dict[str, int]] = {}

    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle")
        if not isinstance(after_oracle, Mapping):
            continue

        if kg_name not in per_kg_counts:
            per_kg_counts[kg_name] = {
                "domains_total": 0,
                "domains_with_positive": 0,
                "json_files_seen": 0,
            }

        per_kg_counts[kg_name]["json_files_seen"] += 1

        for raw_domain in after_oracle.values():
            per_kg_counts[kg_name]["domains_total"] += 1
            if _domain_has_positive_support(raw_domain):
                per_kg_counts[kg_name]["domains_with_positive"] += 1

    if not per_kg_counts:
        raise ValueError("No valid after_oracle sections found in XXXX-1 JSON files")

    kg_names = sorted(per_kg_counts.keys())
    percentages: list[float] = []
    summary: dict[str, dict[str, float]] = {}

    for kg_name in kg_names:
        total = per_kg_counts[kg_name]["domains_total"]
        positive = per_kg_counts[kg_name]["domains_with_positive"]
        pct = (positive / total * 100.0) if total > 0 else 0.0
        percentages.append(pct)
        summary[kg_name] = {
            "domains_total": float(total),
            "domains_with_positive": float(positive),
            "percentage_domains_with_positive": pct,
            "json_files_seen": float(per_kg_counts[kg_name]["json_files_seen"]),
        }

    colors = ["#2E86AB", "#F18F01", "#5A9367", "#C73E1D", "#6C5B7B"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        kg_names,
        percentages,
        color=[colors[i % len(colors)] for i in range(len(kg_names))],
    )
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Proportion of domains with >=1 support > 0 (%)")
    ax.set_xlabel("TKG")
    ax.set_title("After Oracle - Positive-support domain proportion")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = iswc_root / "after_oracle_positive_domain_proportions.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Common histogram saved to {output_path}")
    for kg_name in kg_names:
        pct = summary[kg_name]["percentage_domains_with_positive"]
        positive = int(summary[kg_name]["domains_with_positive"])
        total = int(summary[kg_name]["domains_total"])
        print(f"{kg_name}: {positive}/{total} domains ({pct:.2f}%)")

    return summary


def plot_iswc_support_distributions(bin_width: float = 0.10) -> dict[str, dict[str, Any]]:
    """
    Read all JSON files under Results/XXXX-1 and compute support distributions for
    QCN sections after_oracle and after_propagation, with fixed bins from 0 to 1.

    One histogram figure is generated per KG under Results/XXXX-1/<KG>/.
    Returns raw counts per KG and per section.
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    iswc_root = results_root / "XXXX-1"
    if not iswc_root.exists():
        raise FileNotFoundError(f"Missing XXXX-1 results directory: {iswc_root}")

    json_files = sorted(iswc_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under: {iswc_root}")

    label_by_kg = {
        "Q6256": "Country",
        "Q215380": "Musical group",
        "Q82955": "Politician",
    }

    supports_per_kg: dict[str, dict[str, list[float]]] = {}
    for json_path in json_files:
        rel_path = json_path.relative_to(iswc_root)
        kg_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else json_path.stem

        if kg_name not in supports_per_kg:
            supports_per_kg[kg_name] = {
                "after_oracle": [],
                "after_propagation": [],
                "inferred_only": [],
            }

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)


        for section_name in ("after_oracle", "after_propagation"):
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for raw_domain in section.values():
                supports_per_kg[kg_name][section_name].extend(
                    _extract_supports_from_domain(raw_domain)
                )


        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if isinstance(after_oracle, Mapping) and isinstance(after_propagation, Mapping):
            for pair_key, propagation_domain in after_propagation.items():
                oracle_domain = after_oracle.get(pair_key, {})
                inferred_only_supports = _extract_inferred_only_supports(oracle_domain, propagation_domain)
                supports_per_kg[kg_name]["inferred_only"].extend(inferred_only_supports)

    summary: dict[str, dict[str, Any]] = {}
    for kg_name, section_values in sorted(supports_per_kg.items()):
        oracle_values = section_values["after_oracle"]
        propagation_values = section_values["after_propagation"]
        inferred_only_values = section_values["inferred_only"]

        oracle_zero_count = sum(1 for value in oracle_values if value == 0.0)
        propagation_zero_count = sum(1 for value in propagation_values if value == 0.0)

        edges, oracle_counts = _build_histogram_counts(oracle_values, bin_width)
        _, propagation_counts = _build_histogram_counts(propagation_values, bin_width)
        _, inferred_only_counts = _build_histogram_counts(inferred_only_values, bin_width)

        x_positions = edges[:-1]
        series_width = bin_width * 0.42

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        oracle_x_positions = [x + (bin_width * 0.04) for x in x_positions]
        propagation_x_positions = [x + (bin_width * 0.54) for x in x_positions]
        ax.bar(
            oracle_x_positions,
            oracle_counts,
            width=series_width,
            align="edge",
            color="#2E86AB",
            label=f"after oracle",
        )
        ax.bar(
            propagation_x_positions,
            propagation_counts,
            width=series_width,
            align="edge",
            color="#F18F01",
            label=f"after Propagate&Filter",
        )


        if oracle_counts and oracle_zero_count > 0:
            ax.bar(
                [oracle_x_positions[0]],
                [oracle_zero_count],
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
            )
        if propagation_counts and propagation_zero_count > 0:
            ax.bar(
                [propagation_x_positions[0]],
                [propagation_zero_count],
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
            )


        if inferred_only_counts:
            ax.bar(
                propagation_x_positions,
                inferred_only_counts,
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="green",
                hatch="\\\\\\\\",
                linewidth=1.0,
            )

        kg_label = label_by_kg.get(kg_name, kg_name)
        ax.set_title(f"{kg_label} ({kg_name}) - Confidence score distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence score bin")
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label=" proportion of 0 confidence in [0, 0.1]",
            )
        )
        labels.append("proportion of 0 confidence in [0, 0.1]")
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="green",
                hatch="\\\\\\\\",
                label=" inferred-only relations (not in oracle)",
            )
        )
        labels.append("inferred-only relations (not in oracle)")
        ax.legend(handles, labels)

        tick_positions = [round(i * bin_width, 10) for i in range(int(round(1.0 / bin_width)) + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xlim(0.0, 1.0)

        plt.tight_layout()
        output_path = iswc_root / kg_name / f"support_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Support histograms saved to {output_path}")

        summary[kg_name] = {
            "bins": edges,
            "after_oracle_counts": oracle_counts,
            "after_propagation_counts": propagation_counts,
            "inferred_only_counts": inferred_only_counts,
            "after_oracle_total": len(oracle_values),
            "after_propagation_total": len(propagation_values),
            "inferred_only_total": len(inferred_only_values),
            "after_oracle_zero_count": oracle_zero_count,
            "after_propagation_zero_count": propagation_zero_count,
            "json_files_seen": len(
                [
                    path
                    for path in json_files
                    if (path.relative_to(iswc_root).parts[0] if len(path.relative_to(iswc_root).parts) >= 2 else path.stem) == kg_name
                ]
            ),
        }

    return summary


def plot_qcn3_support_distributions(
    kg_names: list[str] | None = None,
    bin_width: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """
    Read Results/<KG>/qcn3_<KG>.json for each KG and compute support distributions
    for after_oracle and after_propagation.
    One histogram figure (2 axes side by side) is saved to
    Results/<KG>/qcn3_support_histograms_<KG>.png.

    :param kg_names: list of KG names (e.g. ["Q6256", "Q215380"]). If None, auto-discover.
    :param bin_width: histogram bin width on [0, 1].
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    if kg_names is None:
        kg_names = [
            d.name
            for d in results_root.iterdir()
            if d.is_dir() and (d / f"qcn3_{d.name}.json").exists()
        ]
    if not kg_names:
        raise FileNotFoundError(f"No qcn3_<KG>.json files found under {results_root}")

    summary: dict[str, dict[str, Any]] = {}
    for kg_name in sorted(kg_names):
        json_path = results_root / kg_name / f"qcn3_{kg_name}.json"
        if not json_path.exists():
            print(f"⚠️ Fichier introuvable : {json_path}")
            continue

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        oracle_values: list[float] = []
        propagation_values: list[float] = []
        inferred_only_values: list[float] = []

        for section_name, values_list in (("after_oracle", oracle_values), ("after_propagation", propagation_values)):
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for raw_domain in section.values():
                values_list.extend(_extract_supports_from_domain(raw_domain))

        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if isinstance(after_oracle, Mapping) and isinstance(after_propagation, Mapping):
            for pair_key, propagation_domain in after_propagation.items():
                oracle_domain = after_oracle.get(pair_key, {})
                inferred_only_values.extend(
                    _extract_inferred_only_supports(oracle_domain, propagation_domain)
                )

        oracle_zero_count = sum(1 for v in oracle_values if v == 0.0)
        propagation_zero_count = sum(1 for v in propagation_values if v == 0.0)
        inferred_zero_count = sum(1 for v in inferred_only_values if v == 0.0)

        edges, oracle_counts = _build_histogram_counts(oracle_values, bin_width)
        _, propagation_counts = _build_histogram_counts(propagation_values, bin_width)
        _, inferred_only_counts = _build_histogram_counts(inferred_only_values, bin_width)

        x_positions = edges[:-1]
        series_width = bin_width * 0.42
        tick_positions = [round(i * bin_width, 10) for i in range(int(round(1.0 / bin_width)) + 1)]

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        oracle_x = [x + bin_width * 0.04 for x in x_positions]
        propagation_x = [x + bin_width * 0.54 for x in x_positions]

        ax.bar(oracle_x, oracle_counts, width=series_width, align="edge", color="#2E86AB", label="after oracle")
        ax.bar(propagation_x, propagation_counts, width=series_width, align="edge", color="#F18F01", label="after Propagate&Filter")
        if oracle_counts and oracle_zero_count > 0:
            ax.bar([oracle_x[0]], [oracle_zero_count], width=series_width, align="edge",
                   facecolor="none", edgecolor="black", hatch="///", linewidth=0.0)
        if propagation_counts and propagation_zero_count > 0:
            ax.bar([propagation_x[0]], [propagation_zero_count], width=series_width, align="edge",
                   facecolor="none", edgecolor="black", hatch="///", linewidth=0.0)


        if inferred_only_counts:
            ax.bar(
                propagation_x,
                inferred_only_counts,
                width=series_width,
                align="edge",
                facecolor="none",
                edgecolor="green",
                hatch="\\\\\\\\",
                linewidth=1.0,
            )

        ax.set_title(f"{kg_name} (qcn3) - Confidence score distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Confidence score bin")
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="///",
                label="proportion of 0 confidence in [0, 0.1]",
            )
        )
        labels.append("proportion of 0 confidence in [0, 0.1]")
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="green",
                hatch="\\\\\\\\",
                label="inferred-only relations (not in oracle)",
            )
        )
        labels.append("inferred-only relations (not in oracle)")
        ax.legend(handles, labels)

        ax.set_xticks(tick_positions)
        ax.set_xlim(0.0, 1.0)

        plt.tight_layout()
        output_path = results_root / kg_name / f"qcn3_support_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Support histograms saved to {output_path}")

        summary[kg_name] = {
            "bins": edges,
            "after_oracle_counts": oracle_counts,
            "after_propagation_counts": propagation_counts,
            "inferred_only_counts": inferred_only_counts,
            "after_oracle_total": len(oracle_values),
            "after_propagation_total": len(propagation_values),
            "inferred_only_total": len(inferred_only_values),
            "after_oracle_zero_count": oracle_zero_count,
            "after_propagation_zero_count": propagation_zero_count,
        }

    return summary


def plot_qcn3_domain_size_histograms(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Plot domain-size histograms for qcn3 results.

        Domain size is the number of relations in a pair-domain:
        - observed size: count of relations with status in
            {Observed, Inferred}; only domains with size > 0
            are kept in the observed histogram
    - inferred-only size: count of relations with status Inferred in after_propagation
      that are absent (or status None) in after_oracle.

    Saves one figure per KG to:
    Results/<KG>/qcn3_domain_size_histograms_<KG>.png
    """
    observed_statuses = {"Observed", "Inferred"}

    if kg_names is None:
        kg_names = [
            d.name
            for d in results_root.iterdir()
            if d.is_dir() and (d / f"qcn3_{d.name}.json").exists()
        ]
    if not kg_names:
        raise FileNotFoundError(f"No qcn3_<KG>.json files found under {results_root}")

    summary: dict[str, dict[str, Any]] = {}

    for kg_name in sorted(kg_names):
        json_path = results_root / kg_name / f"qcn3_{kg_name}.json"
        if not json_path.exists():
            print(f"Warning: missing file {json_path}")
            continue

        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_oracle = payload.get("after_oracle", {})
        after_propagation = payload.get("after_propagation", {})
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            print(f"Warning: invalid qcn3 payload structure in {json_path}")
            continue

        observed_sizes: list[int] = []
        inferred_only_sizes: list[int] = []
        observed_zero_excluded = 0

        pair_keys = set(after_oracle.keys()) | set(after_propagation.keys())
        for pair_key in pair_keys:
            oracle_domain = after_oracle.get(pair_key, {})
            propagation_domain = after_propagation.get(pair_key, {})

            if not isinstance(oracle_domain, Mapping) or not isinstance(propagation_domain, Mapping):
                continue

            oracle_relations = oracle_domain.get("relations", {})
            propagation_relations = propagation_domain.get("relations", {})
            if not isinstance(oracle_relations, Mapping) or not isinstance(propagation_relations, Mapping):
                continue

            observed_count = 0
            inferred_only_count = 0

            for relation, p_payload in propagation_relations.items():
                if not isinstance(p_payload, Mapping):
                    continue

                p_status = p_payload.get("status")
                if p_status in observed_statuses:
                    observed_count += 1

                o_payload = oracle_relations.get(relation)
                o_status = o_payload.get("status") if isinstance(o_payload, Mapping) else None
                if p_status == "Inferred" and (o_payload is None or o_status is None):
                    inferred_only_count += 1

            if observed_count > 0:
                observed_sizes.append(observed_count)
            else:
                observed_zero_excluded += 1
            inferred_only_sizes.append(inferred_only_count)

        max_size = 0
        if observed_sizes:
            max_size = max(max_size, max(observed_sizes))
        if inferred_only_sizes:
            max_size = max(max_size, max(inferred_only_sizes))
        x_sizes = list(range(max_size + 1))

        observed_hist = [0] * (max_size + 1)
        inferred_only_hist = [0] * (max_size + 1)
        for size in observed_sizes:
            observed_hist[size] += 1
        for size in inferred_only_sizes:
            inferred_only_hist[size] += 1

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5.5))

        ax_left.bar(x_sizes, observed_hist, color="#2E86AB", width=0.85)
        ax_left.set_title(f"{kg_name} - Domain size (Observed/Inferred, non-zero)")
        ax_left.set_xlabel("Domain size (#relations)")
        ax_left.set_ylabel("Number of domains")
        ax_left.set_xticks(x_sizes)
        ax_left.grid(axis="y", linestyle="--", alpha=0.5)

        ax_right.bar(x_sizes, inferred_only_hist, color="#2CA02C", width=0.85)
        ax_right.set_title(f"{kg_name} - Domain size (Inferred-only)")
        ax_right.set_xlabel("Domain size (#relations)")
        ax_right.set_ylabel("Number of domains")
        ax_right.set_xticks(x_sizes)
        ax_right.grid(axis="y", linestyle="--", alpha=0.5)

        plt.suptitle(f"{kg_name} (qcn3) - Domain size distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()

        output_path = results_root / kg_name / f"qcn3_domain_size_histograms_{kg_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Domain-size histograms saved to {output_path}")

        summary[kg_name] = {
            "sizes": x_sizes,
            "observed_hist": observed_hist,
            "inferred_only_hist": inferred_only_hist,
            "observed_count": len(observed_sizes),
            "observed_zero_excluded": observed_zero_excluded,
            "inferred_only_count": len(inferred_only_sizes),
            "observed_min": min(observed_sizes) if observed_sizes else 0,
            "observed_max": max(observed_sizes) if observed_sizes else 0,
            "observed_avg": (sum(observed_sizes) / len(observed_sizes)) if observed_sizes else 0.0,
            "inferred_only_min": min(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_max": max(inferred_only_sizes) if inferred_only_sizes else 0,
            "inferred_only_avg": (sum(inferred_only_sizes) / len(inferred_only_sizes)) if inferred_only_sizes else 0.0,
        }

    return summary


def plot_qcn3_after_propagation_domain_size_histogram(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Plot one histogram per threshold configuration for after_propagation
    domain sizes stored under Results/New.

    Domain size = total number of relations present in each domain
    of the after_propagation section.

    Saves one figure next to each QCN input as:
    Results/New/<KG>/<configuration>/qcn2_<KG>_domain_size_histogram.png
    """
    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")

    summary: dict[str, dict[str, Any]] = {}

    for kg_name in sorted(kg_names):
        kg_directory = new_results_root / kg_name
        if not kg_directory.is_dir():
            print(f"Warning: missing directory {kg_directory}")
            continue

        json_paths = sorted(kg_directory.glob(f"occ_*/qcn2_{kg_name}.json"))
        if not json_paths:
            print(f"Warning: no qcn2 result found under {kg_directory}")
            continue

        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation", {})
            if not isinstance(after_propagation, Mapping):
                print(f"Warning: invalid after_propagation structure in {json_path}")
                continue

            domain_sizes: list[int] = []
            inferred_domain_sizes: list[int] = []
            for domain in after_propagation.values():
                if not isinstance(domain, Mapping):
                    continue
                relations = domain.get("relations", {})
                if isinstance(relations, Mapping):
                    domain_size = len(relations)
                    domain_sizes.append(domain_size)
                    if domain.get("status") == "Inferred":
                        inferred_domain_sizes.append(domain_size)

            if not domain_sizes:
                print(f"Warning: no valid domains found in {json_path}")
                continue

            min_size = min(domain_sizes)
            max_size = max(domain_sizes)
            avg_size = sum(domain_sizes) / len(domain_sizes)

            x_sizes = list(range(min_size, max_size + 1))
            counts = [0] * len(x_sizes)
            inferred_counts = [0] * len(x_sizes)
            for size in domain_sizes:
                counts[size - min_size] += 1
            for size in inferred_domain_sizes:
                inferred_counts[size - min_size] += 1

            configuration_name = json_path.parent.name
            fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
            ax.bar(x_sizes, counts, color="#1F77B4", width=0.85, label="All domains")
            ax.bar(
                x_sizes,
                inferred_counts,
                width=0.85,
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.8,
                label="Inferred domains",
            )
            ax.set_title(
                f"{kg_name} - After propagation domain sizes\n{configuration_name}"
            )
            ax.set_xlabel("Domain size (#relations)")
            ax.set_ylabel("Number of domains")
            ax.set_yscale("log")
            ax.set_xticks(x_sizes)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.legend()

            subtitle = f"min={min_size}, max={max_size}, avg={avg_size:.2f}, n={len(domain_sizes)}"
            ax.text(0.01, 0.98, subtitle, transform=ax.transAxes, va="top", ha="left")

            plt.tight_layout()
            output_path = json_path.with_name(f"{json_path.stem}_domain_size_histogram.png")
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"After-propagation domain-size histogram saved to {output_path}")

            summary[f"{kg_name}/{configuration_name}"] = {
                "sizes": x_sizes,
                "counts": counts,
                "inferred_counts": inferred_counts,
                "num_domains": len(domain_sizes),
                "num_inferred_domains": len(inferred_domain_sizes),
                "min_size": min_size,
                "max_size": max_size,
                "avg_size": avg_size,
                "output_path": str(output_path),
            }

    return summary


def after_propagation_domain_sizes_statistics(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize QCN2 files recursively under Results and write a CSV there.

    Size statistics and informative percentages concern inferred domains only.
    A nonempty inferred domain is informative if it has fewer than 13 relations,
    or 13 relations with different inferred scores. Directed arcs are counted
    separately, as stored in the QCN. Empty populations have undefined statistics
    (None, written as empty CSV cells). Legacy filenames are also supported.
    """
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    summary: dict[str, dict[str, Any]] = {}
    requested = set(kg_names) if kg_names is not None else None
    for json_path in sorted(results_root.rglob("qcn2_*.json")):
        match = re.fullmatch(
            r"qcn2_(Q[0-9]+)(?:_(non_reduced_composition|reduced_composition))?",
            json_path.stem,
        )
        if match is None:
            continue
        kg_name, version = match.groups()
        if requested is not None and kg_name not in requested:
            continue
        if version is None:
            version = next(
                (part for part in json_path.relative_to(results_root).parts[:-1]
                 if part in {"reduced_composition", "non_reduced_composition"}),
                "unknown",
            )
        configuration = json_path.parent.name
        thresholds = re.fullmatch(
            r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
            configuration,
        )
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            print(f"Warning: invalid QCN payload in {json_path}")
            continue
        oracle = payload.get("after_oracle")
        propagated = payload.get("after_propagation")
        if not isinstance(oracle, Mapping) or not isinstance(propagated, Mapping):
            print(f"Warning: missing or invalid QCN stages in {json_path}")
            continue

        eligible = {
            pair for pair, domain in oracle.items()
            if isinstance(domain, Mapping) and domain.get("status") is None
        }
        sizes: list[int] = []
        informative = 0
        inferred_from_none = 0
        for pair, domain in propagated.items():
            if not isinstance(domain, Mapping) or domain.get("status") not in {"Inferred", "Infered"}:
                continue
            relations = domain.get("relations")
            if not isinstance(relations, Mapping):
                continue
            size = len(relations)
            sizes.append(size)  
            inferred_from_none += pair in eligible
            if 0 < size < len(ALLEN_RELATIONS):
                informative += 1
            elif size == len(ALLEN_RELATIONS):
                scores = [
                    relation.get("inferred_score")
                    for relation in relations.values()
                    if isinstance(relation, Mapping)
                ]
                if (len(scores) == size
                    and all(isinstance(score, (int, float)) and not isinstance(score, bool)
                            for score in scores)
                    and len(set(scores)) > 1):
                    informative += 1

        relative_path = str(json_path.relative_to(results_root))
        summary[relative_path] = {
            "kg": kg_name,
            #"configuration": configuration,
            "version": version,
            "occurence_thresh": float(thresholds[1].replace("p", ".")) if thresholds else None,
            "support_thresh": int(thresholds[2]) if thresholds else None,
            "n_domains (possible inferences)": len(eligible),
            "n_inferred_domains": len(sizes),
            "%inferred from none": f"{(inferred_from_none/ len(eligible))*100:.2f}",
            "%domaines_informatifs": f"{(informative / len(sizes))*100:.2f}" if sizes else None,
            "min_domain_sizes": min(sizes) if sizes else None,
            "avg_domain_sizes": f"{statistics.mean(sizes):.2f}" if sizes else None,
            "median_domain_sizes": f"{statistics.median(sizes):.2f}" if sizes else None,
            "max_domain_sizes": f"{max(sizes):.2f}" if sizes else None,
        }

    if not summary:
        print(f"Warning: no matching complete QCN2 results under {results_root}")
        return summary
    output_path = results_root / "after_propagation_domain_sizes_statistics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", *next(iter(summary.values()))])
        writer.writeheader()
        for file, row in summary.items():
            writer.writerow({"file": file, **row})
    print(f"Statistics for {len(summary)} files written to {output_path}")
    return summary


def after_propagation_domain_inference_scores_statistics(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Print inference-score statistics and save them as CSV for informative inferred domains.

    Compute min/max over relation scores per input file. The two averages
    give each informative domain equal weight: mean of maxima and mean of means.
    Informative domains have 1–12 relations, or 13 with distinct scores.
    Domains with missing, nonnumeric or nonfinite scores are skipped and counted.
    Directed arcs are counted separately. Empty populations yield None (empty CSV cells).
    """
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    summary: dict[str, dict[str, Any]] = {}
    requested = set(kg_names) if kg_names is not None else None
    for json_path in sorted(results_root.rglob("qcn2_*.json")):
        match = re.fullmatch(
            r"qcn2_(Q[0-9]+)(?:_(non_reduced_composition|reduced_composition))?",
            json_path.stem,
        )
        if match is None:
            continue
        kg_name, version = match.groups()
        if requested is not None and kg_name not in requested:
            continue
        if version is None:
            version = next(
                (part for part in json_path.relative_to(results_root).parts[:-1]
                 if part in {"reduced_composition", "non_reduced_composition"}),
                "unknown",
            )
        configuration = json_path.parent.name
        thresholds = re.fullmatch(
            r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
            configuration,
        )
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            print(f"Warning: invalid QCN payload in {json_path}")
            continue
        propagated = payload.get("after_propagation")
        if not isinstance(propagated, Mapping):
            print(f"Warning: missing or invalid after_propagation in {json_path}")
            continue

        scores: list[float] = []
        domain_maxima: list[float] = []
        domain_means: list[float] = []
        informative_domains = 0
        invalid_domains = 0
        for domain in propagated.values():
            if not isinstance(domain, Mapping) or domain.get("status") not in {"Inferred", "Infered"}:
                continue
            relations = domain.get("relations")
            if not isinstance(relations, Mapping):
               
                continue
            if not relations:
                continue
            domain_scores = [
                relation.get("inferred_score") if isinstance(relation, Mapping) else None
                for relation in relations.values()
            ]
            if not all(
                isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score)
                for score in domain_scores
            ):
                continue
            if (len(relations) < len(ALLEN_RELATIONS)
                or (len(relations) == len(ALLEN_RELATIONS) and len(set(domain_scores)) > 1)):
                informative_domains += 1
                scores.extend(domain_scores)
                domain_maxima.append(max(domain_scores))
                domain_means.append(statistics.mean(domain_scores))

        relative_path = str(json_path.relative_to(results_root))
        summary[relative_path] = {
            "kg": kg_name,
            "version": version,
            "occurence_thresh": float(thresholds[1].replace("p", ".")) if thresholds else None,
            "support_thresh": int(thresholds[2]) if thresholds else None,
            "min_inference_score": f"{min(scores):.2f}" if scores else None,
            "avg_best_candidate_score": f"{statistics.mean(domain_maxima):.2f}" if domain_maxima else None,
            "avg_domain_mean_score": f"{statistics.mean(domain_means):.2f}" if domain_means else None,
            "max_inference_score": f"{max(scores):.2f}" if scores else None,
        }

    output_path = results_root / "after_propagation_domain_inference_scores_statistics.csv"
    lines = [
        "Inference scores — informative inferred domains only",
        "Min/max over relation scores; averages of domain maxima and domain means. Directed arcs counted separately.",
        "N/A: no scores. Domains with invalid scores are excluded.",
        "",
    ]
    for file, row in summary.items():
        lines.append(f"File: {file}")
        for key, value in row.items():
            formatted = "N/A" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"  {key}: {formatted}")
        lines.append("")
    if not summary:
        lines.append("No matching QCN2 results with after_propagation found.")
    report = "\n".join(lines) + "\n"
    fieldnames = [
        "file", "kg", "version", "occurence_thresh",
        "support_thresh",
        "min_inference_score", "avg_best_candidate_score", "avg_domain_mean_score", "max_inference_score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file, row in summary.items():
            writer.writerow({"file": file, **row})
    print(report, end="")
    print(f"Statistics for {len(summary)} files written to {output_path}")
    return summary


def plot_qcn3_after_propagation_inferred_domain_size_histogram(
    kg_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Plot one domain-size histogram per file for inferred domains only."""
    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [
            directory.name
            for directory in new_results_root.iterdir()
            if directory.is_dir()
        ]
    if not kg_names:
        raise FileNotFoundError(
            f"No knowledge-graph directories found under {new_results_root}"
        )

    summary: dict[str, dict[str, Any]] = {}

    for kg_name in sorted(kg_names):
        kg_directory = new_results_root / kg_name
        if not kg_directory.is_dir():
            print(f"Warning: missing directory {kg_directory}")
            continue

        json_paths = sorted(kg_directory.glob(f"occ_*/qcn2_{kg_name}.json"))
        if not json_paths:
            print(f"Warning: no qcn2 result found under {kg_directory}")
            continue

        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation", {})
            if not isinstance(after_propagation, Mapping):
                print(f"Warning: invalid after_propagation structure in {json_path}")
                continue

            domain_sizes: list[int] = []
            for domain in after_propagation.values():
                if not isinstance(domain, Mapping):
                    continue
                if domain.get("status") != "Inferred":
                    continue
                relations = domain.get("relations")
                if isinstance(relations, Mapping):
                    domain_sizes.append(len(relations))

            if not domain_sizes:
                print(f"Warning: no inferred domains found in {json_path}")
                continue

            min_size = min(domain_sizes)
            max_size = max(domain_sizes)
            avg_size = sum(domain_sizes) / len(domain_sizes)
            x_sizes = list(range(min_size, max_size + 1))
            counts = [0] * len(x_sizes)
            for size in domain_sizes:
                counts[size - min_size] += 1

            configuration_name = json_path.parent.name
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.bar(x_sizes, counts, color="#F18F01", width=0.85)
            ax.set_title(
                f"{kg_name} - Inferred domain sizes after propagation\n"
                f"{configuration_name}"
            )
            ax.set_xlabel("Domain size (#relations)")
            ax.set_ylabel("Number of inferred domains")
            ax.set_yscale("log")
            ax.set_xticks(x_sizes)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.text(
                0.01,
                0.98,
                f"min={min_size}, max={max_size}, avg={avg_size:.2f}, "
                f"n={len(domain_sizes)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
            )

            plt.tight_layout()
            output_path = json_path.with_name(
                f"{json_path.stem}_inferred_domain_size_histogram.png"
            )
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Inferred domain-size histogram saved to {output_path}")

            summary[f"{kg_name}/{configuration_name}"] = {
                "sizes": x_sizes,
                "counts": counts,
                "num_domains": len(domain_sizes),
                "min_size": min_size,
                "max_size": max_size,
                "avg_size": avg_size,
                "output_path": str(output_path),
            }

    return summary


def plot_iswc_after_propagation_inferred_support_histogram(
    kg_names: list[str] | None = None,
    bin_width: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """
    Build one inferred-score histogram per threshold configuration in
    Results/New. Inferred relations store their score in ``inferred_score``.

    Each figure is saved next to its QCN input.
    """
    if bin_width <= 0 or bin_width > 1:
        raise ValueError(f"bin_width must be in (0, 1], got {bin_width}")

    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")

    summary: dict[str, dict[str, Any]] = {}
    for kg_name in sorted(kg_names):
        json_paths = sorted(
            (new_results_root / kg_name).glob(f"occ_*/qcn2_{kg_name}.json")
        )
        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation")
            if not isinstance(after_propagation, Mapping):
                continue

            inferred_scores: list[float] = []
            for domain in after_propagation.values():
                if not isinstance(domain, Mapping):
                    continue
                relations = domain.get("relations")
                if not isinstance(relations, Mapping):
                    continue
                for relation_payload in relations.values():
                    if not isinstance(relation_payload, Mapping):
                        continue
                    score = relation_payload.get("inferred_score")
                    if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
                        inferred_scores.append(float(score))

            if not inferred_scores:
                print(f"Warning: no inferred scores found in {json_path}")
                continue

            edges, counts = _build_histogram_counts(inferred_scores, bin_width)
            x_positions = edges[:-1]
            configuration_name = json_path.parent.name
            fig, ax = plt.subplots(figsize=(11, 5.5))
            ax.bar(
                x_positions,
                counts,
                width=bin_width * 0.9,
                align="edge",
                color="#F18F01",
                edgecolor="black",
                linewidth=0.4,
            )
            ax.set_xticks(edges)
            ax.set_xlim(0.0, 1.0)
            ax.set_yscale("log")
            ax.set_xlabel("Inferred score bin")
            ax.set_ylabel("Count of inferred relations")
            ax.set_title(
                f"{kg_name} - Inferred score distribution\n{configuration_name}"
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            output_path = json_path.with_name(f"{json_path.stem}_inferred_score_histogram.png")
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Inferred-score histogram saved to {output_path}")

            summary[f"{kg_name}/{configuration_name}"] = {
                "bins": edges,
                "counts": counts,
                "total_inferred_relations": len(inferred_scores),
                "output_path": str(output_path),
            }

    return summary


def checkConsistency(jsonfile: str, threshold: float = 1.0) -> bool:
    '''
    Check the consistency of qcn loaded from a json file
    '''
    def to_structured_qcn(
        legacy_qcn: dict[tuple[str, str], dict[str, float]]
    ) -> dict[tuple[str, str], dict[str, object]]:
        structured_qcn: dict[tuple[str, str], dict[str, object]] = {}
        for pair, domain in legacy_qcn.items():
            structured_qcn[pair] = {
                "relations": {
                    relation: {"support": (1, 1), "status": None}
                    for relation, score in domain.items()
                }
            }
        return structured_qcn

    qcn = read_qcn_from_json(jsonfile)



    qcn = _prepare_qcn_for_threshold(qcn, threshold)

    structured_qcn = to_structured_qcn(qcn)

    return path_consistency_classique(structured_qcn) is not None


def Number_of_relevant_observed_domains(jsonfile: str | None = None) -> dict[str, int]:
    """Print and return relevant observed-domain counts per JSON under Results.

    Read thresholds from each file's parent configuration folder. Count an
    Observed domain in after_oracle once if at least one relation meets both
    thresholds (>=). Keys are file paths relative to Results. Skip files without
    a configuration folder or an after_oracle mapping. Pass jsonfile to count
    only that file; files outside Results use an absolute path as their key.
    """
    from .qcn_generator2 import support_counts

    if jsonfile is None and not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    counts_by_file = {}
    json_paths = (
        [Path(jsonfile).resolve()]
        if jsonfile is not None
        else sorted(path for path in results_root.rglob("*.json")
                    if "MRR" not in path.relative_to(results_root).parts)
    )
    for json_path in json_paths:
        thresholds = re.fullmatch(
            r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
            json_path.parent.name,
        )
        if thresholds is None:
            continue
        occurrence_threshold = float(thresholds[1].replace("p", "."))
        total_threshold = int(thresholds[2])
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            continue
        after_oracle = payload.get("after_oracle")
        if not isinstance(after_oracle, Mapping):
            continue

        count = 0
        for domain in after_oracle.values():
            if not isinstance(domain, Mapping) or domain.get("status") != "Observed":
                continue
            relations = domain.get("relations")
            if not isinstance(relations, Mapping):
                continue
            for relation in relations.values():
                if not isinstance(relation, Mapping):
                    continue
                support = support_counts(relation.get("support"))
                if (
                    support is not None
                    and support[1] > 0
                    and support[1] >= total_threshold
                    and support[0] / support[1] >= occurrence_threshold
                ):
                    count += 1
                    break

        try:
            relative_path = str(json_path.resolve().relative_to(results_root.resolve()))
        except ValueError:
            relative_path = str(json_path.resolve())
        counts_by_file[relative_path] = count
        #print(f"{relative_path}: {count}")

    return counts_by_file




def _ablation_relation_counts(
    observed_domains, inferred_domains, occurrence, total, *, missing_records=None,
    include_unreached=False,
):
    """Count qualifying directed-pair/Allen-relation targets and missing scores.

    Count missing relations only in domains with at least one valid prediction.
    Both counts exclude domains without a strictly positive finite prediction.
    Set include_unreached=True only to count the full observed population.
    Only strictly positive finite numeric scores count as recovered relations,
    at any rank. Zero and negative scores count as missing.
    Apply the same positive-evidence thresholds as the weighted MRR.
    If supplied, append missing targets and their original support counts to
    missing_records: support = (relation occurrences, property-pair occurrences).
    """
    from .qcn_generator2 import support_counts

    observed = missing = 0
    for pair, domain in observed_domains.items():
        predictions = inferred_domains.get(pair, {}).get("relations", {})
        reached = any(
            isinstance(payload.get("inferred_score"), (int, float))
            and not isinstance(payload["inferred_score"], bool)
            and math.isfinite(payload["inferred_score"])
            and payload["inferred_score"] > 0
            for payload in predictions.values()
        )
        if not reached and not include_unreached:
            continue
        for relation, evidence in domain["relations"].items():
            counts = support_counts(evidence.get("support"))
            if (counts is None or counts[1] <= 0 or counts[1] < total
                    or counts[0] <= 0 or counts[0] / counts[1] < occurrence):
                continue
            observed += 1
            if not reached:
                continue
            score = predictions.get(relation, {}).get("inferred_score")
            if not (isinstance(score, (int, float))
                    and not isinstance(score, bool) and math.isfinite(score)
                    and score > 0):
                missing += 1
                if missing_records is not None:
                    missing_records.append({
                        "property_pair": pair,
                        "relation": relation,
                        "property_pair_occurrence": counts[1],
                        "relation_occurrence": counts[0],
                        "occurrence_ratio": counts[0] / counts[1],
                    })
    return observed, missing


def ablation_study_of_inference(
    jsonfile: str, n: int, *, seed: int | None = None,
    return_details: bool = False,
) -> float | dict[str, Any] | None:
    '''
    Return the mean weighted reciprocal rank over ten ablation rounds.

    Keep domains from after_oracle with an evidence total at least ``total``
    and at least one occurrence ratio at least ``occurence``.
    Read both thresholds from the parent folder named
    ``occ_<ratio>_min_<total>_timeout_<seconds>`` (decimal separator: p or .).
    Sample n domains without replacement, preserve their original values,
    mask their evidence, and propagate again in each round. Evaluate all
    selected domains against their original observations, then average the
    scores over predicted domains pooled across rounds. Each domain is sampled
    in at most one round, requiring
    at least 10 * n eligible domains. Use seed to reproduce the samples.
    Skip QCNs with fewer than 100 eligible domains and return None.
    Otherwise, return zero when n is zero.
    With return_details, include eligible observed domains/relations and
    cumulative masked/unreached domains and missing relations over ten rounds.
    Directed pairs are counted separately, following the sampling protocol.
    Archive complete masked and inferred QCNs for every round, ground truth,
    parameters and summary under Results/MRR/<dataset>/<configuration>/.
    A new invocation replaces the previous files for the same configuration.
    '''
    from .qcn_generator2 import (
        propagateAndFilterConservative, propagateAndFilterTwoStage,
        propagateAndFilterIterative, support_counts,
    )

    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")

    configuration = Path(jsonfile).resolve().parent.name
    thresholds = re.fullmatch(
        r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
        configuration,
    )
    if thresholds is None:
        raise ValueError(f"Invalid threshold configuration folder: {configuration}")
    occurence = float(thresholds[1].replace("p", "."))
    total = int(thresholds[2])

    # load json file
    with open(jsonfile, "r", encoding="UTF-8") as handle:
        payload = json.load(handle)

    after_oracle = payload.get("after_oracle")
    if not isinstance(after_oracle, Mapping):
        raise ValueError(f"Missing or invalid 'after_oracle' section in {jsonfile}")

    # get a pool of observed domains observed more than total times
    observed_domains = {}
    for pair, domain in after_oracle.items():
        if not isinstance(domain, Mapping) or domain.get("status") != "Observed":
            continue
        relations = domain.get("relations")
        if not isinstance(relations, Mapping):
            continue
        counts = [
            counts
            for relation in relations.values()
            if isinstance(relation, Mapping)
            and (counts := support_counts(relation.get("support"))) is not None
        ]
        if any(count[1] >= total for count in counts):
            observed_domains[pair] = (domain, counts)

    # Keep domains with at least one occurrence ratio greater or equal than the threshold.
    retained_domains = {
        pair: domain
        for pair, (domain, counts) in observed_domains.items()
        if any(
            count[1] > 0 and count[1] >= total and count[0] / count[1] >= occurence
            for count in counts
        )
    }
    valid_domain_count = len(retained_domains)
    observed_relation_count, _ = _ablation_relation_counts(
        retained_domains, {}, occurence, total, include_unreached=True,
    )
    if valid_domain_count < 100:
        print(
            f"Skipping ablation for {jsonfile}: {valid_domain_count} valid observed "
            "domains (at least 100 required)"
        )
        return None

    num_rounds = 10
    if num_rounds * n > len(retained_domains):
        raise ValueError(
            f"Cannot select {n} distinct domains per round for {num_rounds} rounds: "
            f"need {num_rounds * n}, but only {len(retained_domains)} are eligible"
        )
    source = Path(jsonfile).resolve()
    dataset_match = re.search(r"qcn2_(Q[0-9]+)(?:_|$)", source.stem)
    dataset = dataset_match[1] if dataset_match else source.parent.parent.name
    archive = results_root / "MRR" / dataset / configuration
    archive.mkdir(parents=True, exist_ok=True)
    # Remove only generated files so an interrupted or empty rerun cannot
    # leave stale rounds or a previous successful summary in this directory.
    for previous in archive.iterdir():
        if previous.is_file() and (
            previous.name in {"metadata.json", "summary.json", "results.csv", "missing_relations.json",
                              "masked_rounds.json", "inferred_rounds.json"}
            or re.fullmatch(r"round_[0-9]+_(masked|inferred)\.json", previous.name)
        ):
            previous.unlink()
    metadata = {
        "source_json": str(source), "dataset": dataset,
        "configuration": configuration, "seed": seed,
        "n_per_round": n, "num_rounds": num_rounds if n else 0,
        "occurrence_threshold": occurence, "total_threshold": total,
        "counting_unit": "directed pair / Allen relation",
    }

    def save_json(name, data):
        (archive / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    missing_records = []
    masked_rounds = []
    inferred_rounds = []
    save_json("masked_rounds.json", masked_rounds)
    save_json("inferred_rounds.json", inferred_rounds)

    def finish(details):
        save_json("missing_relations.json", {
            **metadata,
            "n_masked_relations": details["n_masked_relations"],
            "n_missing_relations": details["n_missing_relations"],
            "missing_relations": missing_records,
        })
        save_json("summary.json", {**metadata, **details})
        with (archive / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(details))
            writer.writeheader()
            writer.writerow(details)
        return details if return_details else details["mrr"]

    save_json("metadata.json", metadata)
    if n == 0:
        return finish({
            "mrr": 0.0, "n_observed": valid_domain_count,
            "n_observed_relations": observed_relation_count,
            "n_masked": 0, "n_unreached": 0,
            "n_masked_relations": 0, "n_missing_relations": 0,
        })

    rng = random.Random(seed)
    available_pairs = list(retained_domains)
    all_inferred_domains = {}
    all_selected_domains = {}
    for epoch in range(num_rounds):
        # Selectionner uniformement sans remise n domains et les enregistrer quelque part
        selected_pairs = rng.sample(available_pairs, n)
        selected_pair_set = set(selected_pairs)
        available_pairs = [
            pair for pair in available_pairs if pair not in selected_pair_set
        ]
        selected_domains = {
            pair: copy.deepcopy(retained_domains[pair]) for pair in selected_pairs
        }

        # Mettre les domaines selectionnés à un statu None dans after_oracle et toutes les valeurs de support a zero
        masked_after_oracle = copy.deepcopy(after_oracle)
        for pair in selected_pairs:
            domain = masked_after_oracle[pair]
            domain["status"] = None
            for relation in domain["relations"].values():
                relation["support"] = [0, 0]

        # Appeler Propagate and filter sur le after_oracle modifié
        qcn = {}
        for pair, domain in masked_after_oracle.items():
            if "___" not in pair:
                raise ValueError(f"Invalid QCN pair key: {pair}")
            qcn[tuple(pair.split("___", 1))] = domain
        reduced_composition = (
            "reduced_composition" in Path(jsonfile).resolve().parts
            or Path(jsonfile).stem.endswith("_reduced_composition")
            and not Path(jsonfile).stem.endswith("_non_reduced_composition")
        )
        path_parts = Path(jsonfile).resolve().parts
        if "Iterative" in path_parts or "fix_point" in path_parts:
            propagate = propagateAndFilterIterative
        elif "two_stage" in path_parts:
            propagate = propagateAndFilterTwoStage
        else:
            propagate = propagateAndFilterConservative
        metadata.update({"propagation": propagate.__name__,
                         "reduced_composition": reduced_composition})
        save_json("metadata.json", metadata)
        masked_rounds.append(copy.deepcopy({
            "round": epoch + 1, "selected_pairs": selected_pairs,
            "ground_truth": selected_domains, "masked_qcn": masked_after_oracle,
        }))
        save_json("masked_rounds.json", masked_rounds)
        propagated_qcn = propagate(
            qcn,
            domain_reliability_threshold=total,
            relation_occurrence_threshold=occurence,
            reduced_composition=reduced_composition,
        )
        inferred_rounds.append(copy.deepcopy({
            "round": epoch + 1, "selected_pairs": selected_pairs,
            "inferred_qcn": {"___".join(pair): domain
                             for pair, domain in propagated_qcn.items()},
        }))
        save_json("inferred_rounds.json", inferred_rounds)
        # Compare every selected domain after propagation with its oracle values.
        inferred_domains = { # recuperer les domaines masqués apres propagation
            pair: propagated_qcn.get(tuple(pair.split("___", 1)), {})
            for pair in selected_pairs
        }
        round_missing = []
        _ablation_relation_counts(
            selected_domains, inferred_domains, occurence, total,
            missing_records=round_missing,
        )
        missing_records.extend({"round": epoch + 1, **record} for record in round_missing)
        all_inferred_domains.update(inferred_domains)
        all_selected_domains.update(selected_domains)

    score = weighted_mean_reciprocal_rank_evaluation(
        all_inferred_domains, all_selected_domains, occurence, total,
    )
    unreached = sum(
        not any(
            isinstance(evidence.get("inferred_score"), (int, float))
            and not isinstance(evidence["inferred_score"], bool)
            and math.isfinite(evidence["inferred_score"])
            for evidence in domain.get("relations", {}).values()
        )
        for domain in all_inferred_domains.values()
    )
    masked_relations, missing_relations = _ablation_relation_counts(
        all_selected_domains, all_inferred_domains, occurence, total,
    )
    return finish({
        "mrr": score, "n_observed": valid_domain_count,
        "n_observed_relations": observed_relation_count,
        "n_masked": len(all_selected_domains), "n_unreached": unreached,
        "n_masked_relations": masked_relations,
        "n_missing_relations": missing_relations,
    })


def ablation_study_of_inference_all_domains(
    jsonfile: str, *, seed: int | None = None,
    return_details: bool = False,
) -> float | dict[str, Any] | None:
    """Evaluate all valid observed domains in ten disjoint ablation folds.

    Keep the original ablation eligibility, masking and weighted MRR rules.
    Group eligible directed pairs with their converse, shuffle the groups and
    split them into ten balanced folds. Each epoch masks approximately 10%
    of the groups, including both directions. Every eligible directed pair
    is evaluated once. Remainders go to the first folds. For fewer than ten
    groups, some folds are empty. No minimum of 100 is imposed.
    Return None for no eligible domains. Seed controls reproducibility.
    Archive results separately under Results/MRR_all_domains/<dataset>/
    <composition>/<configuration>/; repeated calls replace this run's files.
    """
    from .qcn_generator2 import (
        propagateAndFilterConservative, propagateAndFilterTwoStage,
        propagateAndFilterIterative, support_counts,
    )

    configuration = Path(jsonfile).resolve().parent.name
    thresholds = re.fullmatch(
        r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
        configuration,
    )
    if thresholds is None:
        raise ValueError(f"Invalid threshold configuration folder: {configuration}")
    occurence = float(thresholds[1].replace("p", "."))
    total = int(thresholds[2])

    # load json file
    with open(jsonfile, "r", encoding="UTF-8") as handle:
        payload = json.load(handle)

    after_oracle = payload.get("after_oracle")
    if not isinstance(after_oracle, Mapping):
        raise ValueError(f"Missing or invalid 'after_oracle' section in {jsonfile}")

    # get a pool of observed domains observed more than total times
    observed_domains = {}
    for pair, domain in after_oracle.items():
        if not isinstance(domain, Mapping) or domain.get("status") != "Observed":
            continue
        relations = domain.get("relations")
        if not isinstance(relations, Mapping):
            continue
        counts = [
            counts
            for relation in relations.values()
            if isinstance(relation, Mapping)
            and (counts := support_counts(relation.get("support"))) is not None
        ]
        if any(count[1] >= total for count in counts):
            observed_domains[pair] = (domain, counts)

    # Keep domains with at least one occurrence ratio greater or equal than the threshold.
    retained_domains = {
        pair: domain
        for pair, (domain, counts) in observed_domains.items()
        if any(
            count[1] > 0 and count[1] >= total and count[0] / count[1] >= occurence
            for count in counts
        )
    }
    valid_domain_count = len(retained_domains)
    observed_relation_count, _ = _ablation_relation_counts(
        retained_domains, {}, occurence, total, include_unreached=True,
    )
    if valid_domain_count == 0:
        print(f"Skipping ablation for {jsonfile}: no valid observed domains")
        return None

    num_rounds = 10
    rng = random.Random(seed)
    pair_groups = {}
    for pair in retained_domains:
        if "___" not in pair:
            raise ValueError(f"Invalid QCN pair key: {pair}")
        group = tuple(sorted(pair.split("___", 1)))
        pair_groups.setdefault(group, []).append(pair)
    shuffled_groups = list(pair_groups.values())
    rng.shuffle(shuffled_groups)
    base_size, remainder = divmod(len(shuffled_groups), num_rounds)
    group_sizes = [base_size + (epoch < remainder) for epoch in range(num_rounds)]
    folds = []
    offset = 0
    for size in group_sizes:
        folds.append([pair for group in shuffled_groups[offset:offset + size] for pair in group])
        offset += size
    fold_sizes = [len(fold) for fold in folds]
    source = Path(jsonfile).resolve()
    dataset_match = re.search(r"qcn2_(Q[0-9]+)(?:_|$)", source.stem)
    dataset = dataset_match[1] if dataset_match else source.parent.parent.name
    archive = results_root / "MRR_all_domains" / dataset / source.parent.parent.name / configuration
    archive.mkdir(parents=True, exist_ok=True)
    # Remove only generated files so an interrupted or empty rerun cannot
    # leave stale rounds or a previous successful summary in this directory.
    for previous in archive.iterdir():
        if previous.is_file() and (
            previous.name in {"metadata.json", "summary.json", "results.csv", "missing_relations.json",
                              "masked_rounds.json", "inferred_rounds.json"}
            or re.fullmatch(r"round_[0-9]+_(masked|inferred)\.json", previous.name)
        ):
            previous.unlink()
    metadata = {
        "source_json": str(source), "dataset": dataset,
        "configuration": configuration, "seed": seed,
        "n_per_round": fold_sizes, "num_rounds": num_rounds,
        "mask_fraction": 0.1, "sampling": "ten disjoint folds of converse pairs",
        "converse_groups_per_round": group_sizes,
        "occurrence_threshold": occurence, "total_threshold": total,
        "counting_unit": "directed pair / Allen relation",
    }

    def save_json(name, data):
        (archive / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    missing_records = []
    masked_rounds = []
    inferred_rounds = []
    save_json("masked_rounds.json", masked_rounds)
    save_json("inferred_rounds.json", inferred_rounds)

    def finish(details):
        save_json("missing_relations.json", {
            **metadata,
            "n_masked_relations": details["n_masked_relations"],
            "n_missing_relations": details["n_missing_relations"],
            "missing_relations": missing_records,
        })
        save_json("summary.json", {**metadata, **details})
        with (archive / "results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(details))
            writer.writeheader()
            writer.writerow(details)
        return details if return_details else details["mrr"]

    save_json("metadata.json", metadata)
    all_inferred_domains = {}
    all_selected_domains = {}
    for epoch, selected_pairs in enumerate(folds):
        selected_domains = {
            pair: copy.deepcopy(retained_domains[pair]) for pair in selected_pairs
        }

        # Mettre les domaines selectionnés à un statu None dans after_oracle et toutes les valeurs de support a zero
        masked_after_oracle = copy.deepcopy(after_oracle)
        masked_pairs = set(selected_pairs)
        for pair in selected_pairs:
            p1, p2 = pair.split("___", 1)
            converse = f"{p2}___{p1}"
            if converse in masked_after_oracle:
                masked_pairs.add(converse)
        for pair in sorted(masked_pairs):
            domain = masked_after_oracle[pair]
            domain["status"] = None
            for relation in domain["relations"].values():
                relation["support"] = [0, 0]
                relation.pop("inferred_score", None)

        # Appeler Propagate and filter sur le after_oracle modifié
        qcn = {}
        for pair, domain in masked_after_oracle.items():
            if "___" not in pair:
                raise ValueError(f"Invalid QCN pair key: {pair}")
            qcn[tuple(pair.split("___", 1))] = domain
        reduced_composition = (
            "reduced_composition" in Path(jsonfile).resolve().parts
            or Path(jsonfile).stem.endswith("_reduced_composition")
            and not Path(jsonfile).stem.endswith("_non_reduced_composition")
        )
        path_parts = Path(jsonfile).resolve().parts
        if "Iterative" in path_parts or "fix_point" in path_parts:
            propagate = propagateAndFilterIterative
        elif "two_stage" in path_parts:
            propagate = propagateAndFilterTwoStage
        else:
            propagate = propagateAndFilterConservative
        metadata.update({"propagation": propagate.__name__,
                         "reduced_composition": reduced_composition})
        save_json("metadata.json", metadata)
        masked_rounds.append(copy.deepcopy({
            "round": epoch + 1, "selected_pairs": selected_pairs,
            "masked_pairs": sorted(masked_pairs),
            "ground_truth": selected_domains, "masked_qcn": masked_after_oracle,
        }))
        save_json("masked_rounds.json", masked_rounds)
        propagated_qcn = propagate(
            qcn,
            domain_reliability_threshold=total,
            relation_occurrence_threshold=occurence,
            reduced_composition=reduced_composition,
        ) if selected_pairs else qcn
        inferred_rounds.append(copy.deepcopy({
            "round": epoch + 1, "selected_pairs": selected_pairs,
            "inferred_qcn": {"___".join(pair): domain
                             for pair, domain in propagated_qcn.items()},
        }))
        save_json("inferred_rounds.json", inferred_rounds)
        # Compare every selected domain after propagation with its oracle values.
        inferred_domains = { # recuperer les domaines masqués apres propagation
            pair: propagated_qcn.get(tuple(pair.split("___", 1)), {})
            for pair in selected_pairs
        }
        round_missing = []
        _ablation_relation_counts(
            selected_domains, inferred_domains, occurence, total,
            missing_records=round_missing,
        )
        missing_records.extend({"round": epoch + 1, **record} for record in round_missing)
        all_inferred_domains.update(inferred_domains)
        all_selected_domains.update(selected_domains)

    score = weighted_mean_reciprocal_rank_evaluation(
        all_inferred_domains, all_selected_domains, occurence, total,
    )
    unreached = sum(
        not any(
            isinstance(evidence.get("inferred_score"), (int, float))
            and not isinstance(evidence["inferred_score"], bool)
            and math.isfinite(evidence["inferred_score"])
            for evidence in domain.get("relations", {}).values()
        )
        for domain in all_inferred_domains.values()
    )
    masked_relations, missing_relations = _ablation_relation_counts(
        all_selected_domains, all_inferred_domains, occurence, total,
    )
    return finish({
        "mrr": score, "n_observed": valid_domain_count,
        "n_observed_relations": observed_relation_count,
        "n_masked": len(all_selected_domains), "n_unreached": unreached,
        "n_masked_relations": masked_relations,
        "n_missing_relations": missing_relations,
    })


def benchmark_propagation_cpu(conservative_dir=None):
    """Retourner le temps CPU moyen (secondes) de cinq propagations par JSON.

    Charger uniquement after_oracle dans conservative. Reprendre les seuils
    du dossier occ_*_min_*_timeout_* et la composition indiquée par le chemin.
    Exclure lecture, préparation des copies et affichage du chronométrage.
    Aucun fichier ni résultat d'inférence n'est enregistré.
    """
    import time
    from .qcn_generator2 import propagateAndFilterConservative

    directory = Path(conservative_dir) if conservative_dir is not None else results_root / "conservative"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing conservative directory: {directory}")
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files in {directory}")
    mean_times = {}
    for path in paths:
        print(f"Exploration du fichier : {path}", flush=True)
        with path.open(encoding="utf-8") as handle:
            content = json.load(handle)
        if not isinstance(content, Mapping) or "after_oracle" not in content:
            print(f"Fichier ignoré (sans after_oracle) : {path}", flush=True)
            continue
        after_oracle = content["after_oracle"]
        if not isinstance(after_oracle, Mapping):
            raise ValueError(f"Invalid 'after_oracle' section in {path}")
        thresholds = next((
            match for part in reversed(path.parent.parts)
            if (match := re.fullmatch(
                r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)", part
            )) is not None
        ), None)
        if thresholds is None:
            raise ValueError(f"Missing threshold configuration folder for {path}")
        occurrence = float(thresholds[1].replace("p", "."))
        total = int(thresholds[2])
        reduced_composition = (
            "reduced_composition" in path.parts
            or path.stem.endswith("_reduced_composition")
            and not path.stem.endswith("_non_reduced_composition")
        )
        qcn = {}
        for pair, domain in after_oracle.items():
            if not isinstance(pair, str) or pair.count("___") != 1:
                raise ValueError(f"Invalid QCN pair key in {path}: {pair}")
            qcn[tuple(pair.split("___", 1))] = domain
        cpu_times = []
        for run in range(1, 6):
            run_qcn = copy.deepcopy(qcn)
            started = time.process_time()
            propagated = propagateAndFilterConservative(
                run_qcn,
                domain_reliability_threshold=total,
                relation_occurrence_threshold=occurrence,
                reduced_composition=reduced_composition,
            )
            elapsed = time.process_time() - started
            del propagated
            cpu_times.append(elapsed)
            print(f"{path} | propagation {run}/5 : {elapsed:.6f} s CPU", flush=True)
        mean_times[str(path)] = statistics.mean(cpu_times)
        print(f"Fin du fichier {path} : moyenne CPU = {mean_times[str(path)]:.6f} s", flush=True)
    return mean_times


def report_zero_inferred_scores(conservative_dir=None):
    """Signaler les scores inférés nuls dans les domaines de statut Inferred.

    Parcourir récursivement les JSON et leurs sections QCN (initial,
    after_oracle, after_propagation). Retourner les anomalies sans modifier
    les fichiers. Les booléens et les valeurs manquantes ne sont pas des zéros.
    """
    from .qcn_generator2 import normalize_relation_status, STATUS_INFERRED

    directory = Path(conservative_dir) if conservative_dir is not None else results_root / "conservative"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing conservative directory: {directory}")
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files in {directory}")
    findings = []
    for path in paths:
        print(f"Exploration du fichier : {path}", flush=True)
        findings_before = len(findings)
        with path.open(encoding="utf-8") as handle:
            content = json.load(handle)
        if not isinstance(content, Mapping):
            content = {}
        for section, domains in content.items():
            if not isinstance(domains, Mapping):
                continue
            for pair, domain in domains.items():
                if not isinstance(domain, Mapping):
                    continue
                if normalize_relation_status(domain.get("status")) != STATUS_INFERRED:
                    continue
                relations = domain.get("relations", {})
                if not isinstance(relations, Mapping):
                    continue
                zeros = []
                for relation, evidence in relations.items():
                    if not isinstance(evidence, Mapping):
                        continue
                    score = evidence.get("inferred_score")
                    if isinstance(score, (int, float)) and not isinstance(score, bool) and score == 0:
                        zeros.append(relation)
                if zeros:
                    finding = {"file": str(path), "section": section,
                               "domain": pair, "relations": zeros}
                    findings.append(finding)
                    print(f"{path} | {section} | {pair}: inferred_score=0 pour {', '.join(zeros)}", flush=True)
        file_findings = len(findings) - findings_before
        if file_findings:
            print(f"Fin du fichier {path} : {file_findings} domaines/sections avec un score inféré nul.", flush=True)
        else:
            print(f"Fin du fichier {path} : aucun score inféré nul.", flush=True)
    print(f"{len(paths)} JSON examinés ; {len(findings)} domaines/sections avec un score inféré nul.")
    return findings


def recompute_reached_relations_percentages(archive_dir, output_csv=None):
    """Recompter les relations depuis les archives, sans propagation ni ablation.

    Le pourcentage porte uniquement sur les domaines ayant au moins un score
    fini strictement positif. Les archives et les MRR restent inchangés.
    """
    archive_dir = Path(archive_dir)
    summaries = sorted(archive_dir.rglob("summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No archived summaries in {archive_dir}")
    rows = []
    for summary_path in summaries:
        folder = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        masked = json.loads((folder / "masked_rounds.json").read_text(encoding="utf-8"))
        inferred = json.loads((folder / "inferred_rounds.json").read_text(encoding="utf-8"))
        predictions_by_round = {item["round"]: item for item in inferred}
        if len(predictions_by_round) != len(inferred) or {item["round"] for item in masked} != set(predictions_by_round):
            raise ValueError(f"Mismatched archived rounds in {folder}")
        targets = missing = 0
        for item in masked:
            prediction = predictions_by_round[item["round"]]["inferred_qcn"]
            ground_truth = item["ground_truth"]
            count, absent = _ablation_relation_counts(
                ground_truth, {pair: prediction.get(pair, {}) for pair in ground_truth},
                summary["occurrence_threshold"], summary["total_threshold"],
            )
            targets += count
            missing += absent
        percentage = 100 * (targets - missing) / targets if targets else None
        row = {"file": str(folder.relative_to(archive_dir)),
               "relations_in_reached_domains": targets,
               "recovered_relations": targets - missing,
               "missing_relations": missing,
               "reached_relations_percent": percentage}
        rows.append(row)
        formatted = f"{percentage:.2f}%" if percentage is not None else "--"
        print(f"{row['file']}: {targets - missing}/{targets} = {formatted}", flush=True)
    if output_csv is not None:
        with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run_ablation_studies_of_inference_all_domains(
    qid: str, *, seed: int | None = 42,
) -> dict[str, dict[str, Any] | None]:
    """Run the all-domain ablation for one QID under Results/conservative.

    Return details keyed by relative source path (None for no valid domains).
    Per-configuration archives are written by the all-domain ablation.
    """
    if not isinstance(qid, str) or re.fullmatch(r"Q[0-9]+", qid) is None:
        raise ValueError("qid must be a dataset identifier such as Q6256")
    directory = results_root / "conservative" / qid
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing results directory: {directory}")
    files = sorted(directory.rglob(f"qcn2_{qid}_*.json"))
    if not files:
        raise FileNotFoundError(f"No QCN result files found in {directory}")
    reports = {}
    for source in files:
        relative = source.relative_to(results_root).as_posix()
        print(f"Processing {relative}", flush=True)
        details = ablation_study_of_inference_all_domains(
            str(source), seed=seed, return_details=True,
        )
        reports[relative] = details
        print(f"{relative}: {details}", flush=True)
    return reports


def run_ablation_studies_of_inference(
    n: int = 10, *, seed: int | None = 42
) -> dict[str, float]:
    """Run conservative-only ten-round ablations on eligible QCNs under Results.

    Exclude two_stage, iterative, fix_point and strass directories. Legacy
    paths without a propagation-mode directory use conservative propagation.

    Skip files with fewer than 100 valid observed domains or fewer than 10 * n
    domains for disjoint samples. Print and return scores keyed by relative
    file path. The same seed makes each QCN's experiment reproducible.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")

    scores = {}
    reports = []
    required_domains = max(100, 10 * n)
    for relative_path, count in Number_of_relevant_observed_domains().items():
        path_parts = {part.lower() for part in (results_root / relative_path).resolve().parts}
        if path_parts & {"two_stage", "iterative", "fix_point", "strass"}:
            continue
        if count < required_domains:
            print(
                f"Skipping ablation for {relative_path}: {count} valid observed "
                f"domains (at least {required_domains} required)"
            )
            continue
        details = ablation_study_of_inference(
            str(results_root / relative_path), n=n, seed=seed, return_details=True
        )
        if details is not None:
            score = details["mrr"]
            scores[relative_path] = score
            reports.append({"file": relative_path, **details})
            print(
                f"{relative_path}: non atteints = {details['n_unreached']}/{details['n_masked']} "
                f"(cumul 10 epochs) | observed domains = {details['n_observed']} "
                f"| observed relations = {details['n_observed_relations']} "
                f"| missing relations = {details['n_missing_relations']}/{details['n_masked_relations']} "
                f"| weighted MRR = {score}"
            )
        else:
            print(f"{relative_path}: ablation study returned None")
    if reports:
        with (results_root / "ablation_inference_counts.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
            writer.writeheader()
            writer.writerows(reports)
    return scores


def evaluate_two_stage_inference(
    inferred_domains: dict, observed_domains: dict,
    occurences: float, total: int,
) -> dict[str, Any]:
    """Evaluate masked domains globally and by their inference stage.

    Reuse the support-weighted filtered MRR. Stage means are conditional on
    being inferred in that stage; empty groups have an undefined mean (None).
    Unrecovered domains are excluded from the global mean. Each supplied
    observed domain must have positive evidence qualifying under the thresholds.
    """
    scores = []
    stage_scores = {1: [], 2: []}
    for pair, observed in observed_domains.items():
        prediction = inferred_domains.get(pair, {})
        has_prediction = any(
            isinstance(evidence.get("inferred_score"), (int, float))
            and not isinstance(evidence["inferred_score"], bool)
            and math.isfinite(evidence["inferred_score"])
            for evidence in prediction.get("relations", {}).values()
        )
        if prediction.get("status") == "Inferred" and has_prediction:
            stage = prediction.get("inference_stage")
            if type(stage) is not int or stage not in stage_scores:
                raise ValueError(f"Missing or invalid inference_stage for {pair}")
            score = weighted_mean_reciprocal_rank_evaluation(
                {pair: prediction}, {pair: observed}, occurences, total,
            )
            stage_scores[stage].append(score)
        else:
            continue
        scores.append(score)
    return {
        "n_masked": len(observed_domains),
        "n_stage_1": len(stage_scores[1]),
        "n_stage_2": len(stage_scores[2]),
        "n_uninferred": len(observed_domains) - len(stage_scores[1]) - len(stage_scores[2]),
        "mrr_global": statistics.mean(scores) if scores else None,
        "mrr_stage_1": statistics.mean(stage_scores[1]) if stage_scores[1] else None,
        "mrr_stage_2": statistics.mean(stage_scores[2]) if stage_scores[2] else None,
    }


def ablation_study_two_stage(
    jsonfile: str, n: int = 10, *, seed: int | None = 42,
    reduced_composition: bool = True,
) -> dict[str, Any] | None:
    """Run ten disjoint rounds, masking n property pairs and their converses.

    Evaluate one direction per pair to avoid double counting. Every round
    starts from the original after_oracle. Always use TwoStage regardless of
    the folder name. Stage scores are pooled by domain across rounds, rather
    than averaging stage means from differently sized groups.
    """
    from .qcn_generator2 import propagateAndFilterTwoStage, support_counts

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    path = Path(jsonfile)
    thresholds = re.fullmatch(
        r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)", path.parent.name,
    )
    if thresholds is None:
        raise ValueError(f"Invalid threshold configuration folder: {path.parent.name}")
    occurrence, total = float(thresholds[1].replace("p", ".")), int(thresholds[2])
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    oracle = payload.get("after_oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError(f"Missing or invalid after_oracle in {path}")
    qcn = {}
    for key, domain in oracle.items():
        if "___" not in key:
            raise ValueError(f"Invalid QCN pair key: {key}")
        qcn[tuple(key.split("___", 1))] = domain
    eligible = {}
    for pair, domain in sorted(qcn.items()):
        if domain.get("status") != "Observed":
            continue
        if any(
            counts is not None and counts[0] > 0 and counts[1] >= total
            and counts[0] / counts[1] >= occurrence
            for evidence in domain["relations"].values()
            for counts in [support_counts(evidence.get("support"))]
        ):
            canonical = tuple(sorted(pair))
            if (pair[1], pair[0]) not in qcn:
                raise ValueError(f"Missing converse for {pair}")
            eligible.setdefault(canonical, pair)
    if len(eligible) < 10 * n:
        print(f"Skipping ablation for {path}: {len(eligible)} eligible property pairs; {10 * n} required")
        return None
    samples = random.Random(seed).sample(list(eligible.values()), 10 * n)
    all_predictions, all_observations, rounds = {}, {}, []
    for epoch in range(10):
        selected = samples[epoch * n:(epoch + 1) * n]
        masked = copy.deepcopy(qcn)
        for a, b in selected:
            for pair in ((a, b), (b, a)):
                masked[pair] = {
                    "status": None,
                    "relations": {relation: {"support": (0, 0)} for relation in ALLEN_RELATIONS},
                }
        propagated = propagateAndFilterTwoStage(masked, total, occurrence, reduced_composition)
        observed = {pair: qcn[pair] for pair in selected}
        predictions = {pair: propagated[pair] for pair in selected}
        rounds.append(evaluate_two_stage_inference(predictions, observed, occurrence, total))
        all_predictions.update(predictions)
        all_observations.update(observed)
    summary = evaluate_two_stage_inference(all_predictions, all_observations, occurrence, total)
    summary.update({"rounds": rounds, "seed": seed, "reduced_composition": reduced_composition})
    return summary


def run_two_stage_ablation_studies(
    n: int = 10, *, seed: int | None = 42,
) -> dict[str, dict[str, Any]]:
    """Evaluate QCN2 files under results_root and save a CSV plus per-round JSON."""
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    summaries = {}
    for path in sorted(results_root.rglob("qcn2_*.json")):
        summary = ablation_study_two_stage(str(path), n, seed=seed)
        if summary is None:
            continue
        relative = str(path.relative_to(results_root))
        summaries[relative] = summary
        print(f"{relative}: " + " | ".join(
            f"{key}={value}" for key, value in summary.items() if key != "rounds"
        ), flush=True)
    fields = ["file", "n_masked", "n_stage_1", "n_stage_2", "n_uninferred",
              "mrr_global", "mrr_stage_1", "mrr_stage_2", "seed", "reduced_composition"]
    with (results_root / "two_stage_ablation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path, summary in summaries.items():
            writer.writerow({"file": path, **{k: v for k, v in summary.items() if k != "rounds"}})
    (results_root / "two_stage_ablation.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return summaries


def weighted_mean_reciprocal_rank_evaluation(
    inferred_domains: dict,
    observed_domains: dict,
    occurences: float,
    total: int,
) -> float:
    """Average the support-weighted filtered reciprocal ranks over domains.

    Both mappings must use the same pair keys. Observed relations qualify when
    their support ratio >= occurences and evidence total >= total. Normalize
    those ratios into weights within each domain. Rank by inferred_score,
    excluding other qualifying observed relations and averaging tied ranks.
    Skip domains without valid predictions or without positive observed weight;
    return zero if no domain qualifies. Inputs are left unchanged.
    """
    from .qcn_generator2 import support_counts

    domain_scores = []
    for pair, domain in observed_domains.items(): # pour chaque pair masquée
        # Retain only relations meeting both thresholds.
        weights = {}
        for relation, evidence in domain["relations"].items():
            counts = support_counts(evidence.get("support")) # calcul du support
            if counts is None or counts[1] == 0 or counts[1] < total:
                continue
            ratio = counts[0] / counts[1]
            if ratio > 0 and ratio >= occurences:
                weights[relation] = counts[0] # garder le nombre d'occurrences de la relation

        # Normalize observed supports within the domain.
        weight_sum = sum(weights.values()) # somme des nbr d'occurences des relations dans le domaine
        if weight_sum == 0:
            continue
        # calculer le poids de la relation (servira dans la pondération du rang)
        weights = {relation: weight / weight_sum for relation, weight in weights.items()}

        # Récupérer le domaine inféré correspondant et trier par score inféré.
        inferred_relations = inferred_domains.get(pair, {}).get("relations", {})
        ranked_relations = sorted(
            (
                (relation, evidence["inferred_score"])
                for relation, evidence in inferred_relations.items()
                if isinstance(evidence.get("inferred_score"), (int, float))
                and not isinstance(evidence["inferred_score"], bool)
                and math.isfinite(evidence["inferred_score"])
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked_relations: # passer si le domaine n'est pas atteint par l'inference
            continue
        inferred_scores = dict(ranked_relations)

        # Filter out the other expected relations for each target's rank
        domain_score = 0.0
        for relation, weight in weights.items():
            if relation not in inferred_scores: # Si une relation est dans le GT mais pas inférée
                continue # equivalent a un RR = 0 
            score = inferred_scores[relation]
            competitors = [
                other_score
                for other_relation, other_score in ranked_relations
                if other_relation not in weights
            ]
            rank = (
                1
                + sum(other_score > score for other_score in competitors)
                + 0.5 * sum(other_score == score for other_score in competitors)
            )
            domain_score += weight / rank # on multiplie le rang par le weight
        domain_scores.append(domain_score)

    return sum(domain_scores) / len(domain_scores) if domain_scores else 0.0


    


def propagation_rate() -> dict[str, float | None]:
    """Print and return propagation rates for JSON files recursively under Results.

    Divide the number of Inferred domains after propagation by the number of
    None (or missing-status) domains after the oracle. Count directed domains
    separately. A zero denominator yields None; files missing either QCN
    section are skipped.
    """
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    rates: dict[str, float | None] = {}
    for json_path in sorted(results_root.rglob("*.json")):
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            continue
        after_oracle = payload.get("after_oracle")
        after_propagation = payload.get("after_propagation")
        if not isinstance(after_oracle, Mapping) or not isinstance(after_propagation, Mapping):
            continue

        unknown_count = sum(
            1 for domain in after_oracle.values()
            if isinstance(domain, Mapping) and domain.get("status") is None
        )
        inferred_count = sum(
            1 for domain in after_propagation.values()
            if isinstance(domain, Mapping) and domain.get("status") == "Inferred"
        )
        rate = inferred_count / unknown_count if unknown_count else None
        relative_path = str(json_path.relative_to(results_root))
        rates[relative_path] = rate
        formatted_rate = f"{rate:.2%}" if rate is not None else "undefined"
        database_match = re.match(r"qcn2_(Q[0-9]+)(?:_|$)", json_path.stem)
        database = database_match[1] if database_match else next(
            (part for part in reversed(json_path.resolve().parts[:-1])
             if re.fullmatch(r"Q[0-9]+", part)),
            "inconnue",
        )
        thresholds = re.fullmatch(
            r"occ_([0-9]+(?:[p.][0-9]+)?)_min_([0-9]+)_timeout_([0-9]+)",
            json_path.parent.name,
        )
        occurrence = f"{float(thresholds[1].replace('p', '.')):g}" if thresholds else "inconnu"
        minimum = str(int(thresholds[2])) if thresholds else "inconnu"
        print(
            f"BDD={database} | seuil occurrence={occurrence} | seuil support={minimum}: "
            f"{inferred_count}/{unknown_count} = {formatted_rate}"
        )

    return rates



def conservative_cpu_time_statistics() -> dict[str, Path]:
    """Write one TSV tempscpu_Qxxx.txt per dataset in Results/conservative.

    Include every existing configuration directory, even without a stats file.
    Missing stage timings are written as TO (timeout convention). Only the
    CPU TIME PER STAGE section is read, excluding cumulative/global timings.
    Configurations with no directory cannot be discovered by this function.
    Return the output paths indexed by dataset QID. Times are in seconds.
    """
    directory = results_root / "conservative"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing results directory: {directory}")

    configuration_re = re.compile(
        r"^occ_([0-9]+(?:p[0-9]+)?)_min_(\d+)_timeout_(\d+)$"
    )
    stage_re = re.compile(
        r"^Stage (after_oracle|after_propagation) \(s\)\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(?:s)?\s*$"
    )
    outputs = {}
    for dataset in sorted(directory.iterdir()):
        if not dataset.is_dir() or not re.fullmatch(r"Q\d+", dataset.name):
            continue
        rows = []
        for configuration in dataset.rglob("occ_*"):
            match = configuration_re.fullmatch(configuration.name)
            if not configuration.is_dir() or match is None:
                continue
            occurrence, minimum, timeout = match.groups()
            composition = configuration.parent.relative_to(dataset).as_posix()
            timings = {"after_oracle": "TO", "after_propagation": "TO"}
            stats_files = sorted(configuration.glob("*_stats.txt"))
            if len(stats_files) > 1:
                raise ValueError(f"Multiple statistics files in {configuration}")
            if stats_files:
                in_section = False
                for line in stats_files[0].read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("## "):
                        in_section = line == "## CPU TIME PER STAGE"
                        continue
                    stage = stage_re.fullmatch(line) if in_section else None
                    if stage:
                        timings[stage.group(1)] = stage.group(2)
            rows.append({
                "configuration": configuration.relative_to(dataset).as_posix(),
                "composition": composition,
                "tau_c": float(occurrence.replace("p", ".")),
                "tau": int(minimum),
                "timeout_s": int(timeout),
                "after_oracle_cpu_s": timings["after_oracle"],
                "after_propagation_cpu_s": timings["after_propagation"],
            })
        rows.sort(key=lambda row: (
            row["composition"], row["tau_c"], row["tau"], row["timeout_s"]
        ))
        output = directory / f"tempscpu_{dataset.name}.txt"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=[
                "configuration", "composition", "tau_c", "tau", "timeout_s",
                "after_oracle_cpu_s", "after_propagation_cpu_s",
            ])
            writer.writeheader()
            writer.writerows(rows)
        outputs[dataset.name] = output
        print(f"{output}: {len(rows)} configurations")
    return outputs


def conservative_domain_status_statistics() -> dict[str, dict[str, int | float | None]]:
    """Return domain counts and percentages per conservative result file.

    Keys are file paths relative to Results/conservative, identifying the KG,
    composition mode and configuration. Use all directed domains in
    ``after_propagation`` as the denominator. Statuses other than ``Observed``
    and ``Inferred`` (including null or missing) count as unknown.
    Percentages range from 0 to 100; they are None for an empty QCN.
    Files without an after-propagation QCN are skipped.
    """
    directory = results_root / "conservative"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing results directory: {directory}")

    statistics_by_configuration: dict[str, dict[str, int | float | None]] = {}
    for json_path in sorted(directory.rglob("*.json")):
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)
        domains = payload.get("after_propagation") if isinstance(payload, Mapping) else None
        if not isinstance(domains, Mapping):
            continue

        counts = {"observed": 0, "inferred": 0, "unknown": 0}
        for pair, domain in domains.items():
            if not isinstance(domain, Mapping):
                raise ValueError(f"Invalid domain for {pair} in {json_path}")
            category = {"Observed": "observed", "Inferred": "inferred"}.get(
                domain.get("status"), "unknown"
            )
            counts[category] += 1

        total = len(domains)
        row: dict[str, int | float | None] = {"total_domains": total}
        for category, count in counts.items():
            row[f"{category}_count"] = count
            row[f"{category}_percentage"] = 100 * count / total if total else None
        statistics_by_configuration[str(json_path.relative_to(directory))] = row

    return statistics_by_configuration


def number_of_unknown_domains_after_propagation() -> dict[str, int]:
    """Print and return unknown-domain counts for JSON files under Results.

    Count each directed domain in ``after_propagation`` whose status is neither
    ``Observed`` nor ``Inferred``, including null or missing statuses. Files
    without an after-propagation QCN are skipped.
    """
    if not results_root.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_root}")

    counts: dict[str, int] = {}
    for json_path in sorted(results_root.rglob("*.json")):
        with json_path.open("r", encoding="UTF-8") as handle:
            payload = json.load(handle)

        after_propagation = (
            payload.get("after_propagation") if isinstance(payload, Mapping) else None
        )
        if not isinstance(after_propagation, Mapping):
            continue

        count = sum(
            1
            for domain in after_propagation.values()
            if isinstance(domain, Mapping)
            and domain.get("status") not in ("Observed", "Inferred")
        )
        relative_path = str(json_path.relative_to(results_root))
        counts[relative_path] = count
        print(f"{relative_path}: {count}")

    return counts









def number_of_distinct_scores_indomains(
    kg_names: list[str] | None = None,
    *,
    mode: str = "New",
) -> dict[str, dict[int, int]]:
    '''
    Return counts of inferred domains by distinct inferred-score count (1 to 13).

    Counts are aggregated across all matching JSON files for each KG.
    Set mode to "reduced_composition" or "non_reduced_composition" to
    read the corresponding Results subdirectory; None selects all KGs.
    '''
    new_results_root = results_root / mode
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")
    counts_by_kg: dict[str, dict[int, int]] = {}
    for kg_name in sorted(kg_names):
        counts = dict.fromkeys(range(1, 14), 0)
        counts_by_kg[kg_name] = counts
        json_paths = sorted(
            {
                path
                for filename in (f"qcn2_{kg_name}.json", f"qcn2_{kg_name}_{mode}.json")
                for path in (new_results_root / kg_name).glob(f"occ_*/{filename}")
            }
        )
        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation")
            if not isinstance(after_propagation, Mapping):
                continue
            for domain in after_propagation.values():
                if not isinstance(domain, Mapping) or domain.get("status") != "Inferred":
                    continue
                relations = domain.get("relations")
                if not isinstance(relations, Mapping):
                    continue
                distinct_scores = {
                    relation["inferred_score"]
                    for relation in relations.values()
                    if isinstance(relation, Mapping)
                    and isinstance(relation.get("inferred_score"), (int, float))
                    and not isinstance(relation["inferred_score"], bool)
                }
                if len(distinct_scores) in counts:
                    counts[len(distinct_scores)] += 1

    return counts_by_kg




def numberOfNoneDomains(kg_names: list[str] | None = None) -> None:
    '''
    After propagation, count domains having a None status for each JSON file.
    '''
    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")
    for kg_name in sorted(kg_names):
        json_paths = sorted(
            (new_results_root / kg_name).glob(f"occ_*/qcn2_{kg_name}.json")
        )
        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation")
            if not isinstance(after_propagation, Mapping):
                continue
            count = sum(
                1
                for domain in after_propagation.values()
                if isinstance(domain, Mapping) and domain.get("status") is None
            )
            print(f"{json_path.relative_to(new_results_root)}: {count}")



def numberOfDomainslessof13(kg_names: list[str] | None = None) -> None:
    '''
    After propagation, count domains having less than 13 relations for each JSON file.
    '''
    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")
    for kg_name in sorted(kg_names):
        json_paths = sorted(
            (new_results_root / kg_name).glob(f"occ_*/qcn2_{kg_name}.json")
        )
        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation")
            if not isinstance(after_propagation, Mapping):
                continue
            count = sum(
                1
                for domain in after_propagation.values()
                if isinstance(domain, Mapping) and domain.get("status") == "Inferred" and len(domain.get("relations", {})) < 13
            )
            print(f"{json_path.relative_to(new_results_root)}: {count}")

def exploitableInferredDomains(kg_names: list[str] | None = None) -> None:
    '''
    After propagation, count exploitable inferred domains for each JSON file.
    '''
    new_results_root = results_root / "New"
    if not new_results_root.exists():
        raise FileNotFoundError(f"Missing New results directory: {new_results_root}")

    if kg_names is None:
        kg_names = [directory.name for directory in new_results_root.iterdir() if directory.is_dir()]
    if not kg_names:
        raise FileNotFoundError(f"No knowledge-graph directories found under {new_results_root}")
    for kg_name in sorted(kg_names):
        json_paths = sorted(
            (new_results_root / kg_name).glob(f"occ_*/qcn2_{kg_name}.json")
        )
        for json_path in json_paths:
            with json_path.open("r", encoding="UTF-8") as handle:
                payload = json.load(handle)

            after_propagation = payload.get("after_propagation")
            if not isinstance(after_propagation, Mapping):
                continue
            total_count = 0
            exploitable_domains = 0
            for domain in after_propagation.values():
                if not isinstance(domain, Mapping) or domain.get("status") != "Inferred":
                    continue

                total_count += 1
                relations = domain.get("relations")
                if not isinstance(relations, Mapping):
                    continue
                inferred_scores = sorted(
                    (
                        float(relation["inferred_score"])
                        for relation in relations.values()
                        if isinstance(relation, Mapping)
                        and isinstance(relation.get("inferred_score"), (int, float))
                        and not isinstance(relation.get("inferred_score"), bool)
                    ),
                    reverse=True,
                )
                if (
                    len(inferred_scores) >= 2
                    and inferred_scores[0] - inferred_scores[1] >= 0.3
                ):
                    exploitable_domains += 1

            print(
                f"{json_path.relative_to(new_results_root)}: "
                f"{exploitable_domains}/{total_count} exploitable inferred domains"
            )


                    
#if __name__ == "__main__":
    # after_propagation_domain_sizes_statistics()
    #exploitableInferredDomains()
    # parser = argparse.ArgumentParser(description="Check consistency of a QCN from a JSON file.")
    # parser.add_argument("jsonfile", type=str, help="Path to the QCN JSON file.")
    # parser.add_argument("--threshold", type=float, default=1.0, help="Score threshold for consistency check.")
    # parser.add_argument(
    #     "--incoherence-report",
    #     action="store_true",
    #     help="Generate a detailed report with contradictory triplets and incoherence path.",
    # )
    # parser.add_argument(
    #     "--max-triplets",
    #     type=int,
    #     default=300,
    #     help="Maximum number of contradictory triplets written in the report.",
    # )
    # parser.add_argument(
    #     "--trace-tail",
    #     type=int,
    #     default=30,
    #     help="Number of propagation steps kept before the collapse event.",
    # )
    # args = parser.parse_args()

    # if args.incoherence_report:
    #     report = incoherence_report(
    #         args.jsonfile,
    #         threshold=args.threshold,
    #         max_triplets=args.max_triplets,
    #         trace_tail=args.trace_tail,
    #     )
    #     print(f"Incoherence report written to: {report}")

    # is_consistent = checkConsistency(args.jsonfile, threshold=args.threshold)
    # if is_consistent:
    #     print("The QCN is consistent.")
    # else:
    #     print("The QCN is inconsistent.")


# Ordered threshold grids used in the paper (positions are equally spaced).
DOMAIN_SIZE_TAUS = {
    "Country": [0, 10, 59, 110, 160, 190],
    "Politician": [0, 14, 42, 128, 389, 1185],
    "Musical group": [0, 10, 30, 60, 100, 150],
}
DOMAIN_SIZE_QIDS = {"Q6256": "Country", "Q82955": "Politician", "Q215380": "Musical group"}


def load_domain_size_rows(input_path: str | Path, composition: str = "reduced_composition") -> list[dict[str, Any]]:
    """Load tables or domains classified by their after_propagation status.

    Directed arcs are counted separately. Observed sizes count strictly positive
    support numerators; inferred sizes count strictly positive inferred_score
    values. Status None is
    excluded. A None size marks absence from that status-specific population.
    Tables without status columns explicitly supply both size populations.
    Only occurrence=0 runs of the selected composition are loaded; duplicates
    are rejected rather than pooling configurations.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob("*")) if root.is_dir() else [root]
    tables = [p for p in paths if p.suffix.lower() in {".csv", ".tsv"}]
    rows = []
    required = {"dataset", "tau", "domain_id", "size_observed", "size_inferred"}
    for path in tables:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            if not required.issubset(reader.fieldnames or []):
                continue
            rows.extend(dict(row) for row in reader)
    # Prefer explicit tabular data when supplied, rather than counting it twice.
    if not rows:
        for path in paths:
            if path.suffix.lower() != ".json" or not path.stem.startswith("qcn2_"):
                continue
            match = re.fullmatch(r"qcn2_(Q\d+)(?:_(non_reduced_composition|reduced_composition))?", path.stem)
            if match is None or match[1] not in DOMAIN_SIZE_QIDS:
                continue
            variant = match[2] or next((p for p in path.parts if p in {"reduced_composition", "non_reduced_composition"}), None)
            if variant != composition:
                continue
            config = next((m for part in path.parts if (m := re.fullmatch(r"occ_([\d.p]+)_min_(\d+)_timeout_\d+", part))), None)
            dataset = DOMAIN_SIZE_QIDS[match[1]]
            if config is None or float(config[1].replace("p", ".")) != 0:
                continue
            tau = int(config[2])
            if tau not in DOMAIN_SIZE_TAUS[dataset]:
                continue
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            propagated = payload["after_propagation"]
            for pair, domain in propagated.items():
                status = domain.get("status")
                if status not in {"Observed", "Inferred", "Infered"}:
                    continue
                observed = None
                inferred = None
                if status == "Observed":
                    observed = sum(
                        1 for relation in domain["relations"].values()
                        if isinstance(relation.get("support"), (list, tuple))
                        and len(relation["support"]) == 2
                        and relation["support"][0] > 0
                    )
                else:
                    inferred = sum(
                        1 for relation in domain["relations"].values()
                        if isinstance(relation.get("inferred_score"), (int, float))
                        and not isinstance(relation["inferred_score"], bool)
                        and relation["inferred_score"] > 0
                    )
                rows.append(dict(dataset=dataset, tau=tau, domain_id=pair,
                                 size_observed=observed, size_inferred=inferred))
            # Release the large QCN before loading the next threshold.
            del payload, propagated
    if not rows:
        raise ValueError(f"No domain-size data found in {root}")
    seen = set()
    for row in rows:
        dataset = row["dataset"]
        if dataset not in DOMAIN_SIZE_TAUS:
            raise ValueError(f"Unknown dataset: {dataset!r}")
        for field in ("tau", "size_observed", "size_inferred"):
            value = row[field]
            if field != "tau" and value is None:
                continue
            if isinstance(value, bool) or not re.fullmatch(r"\d+", str(value)):
                raise ValueError(f"Invalid integer {field}: {value!r}")
            row[field] = int(value)
        if row["tau"] not in DOMAIN_SIZE_TAUS[dataset]:
            raise ValueError(f"Unexpected tau for {dataset}: {row['tau']}")
        if not all(row[field] is None or 1 <= row[field] <= 13 for field in ("size_observed", "size_inferred")):
            raise ValueError(f"Domain sizes must be in [1, 13]: {row}")
        if not row["domain_id"]:
            raise ValueError("Empty domain_id")
        key = dataset, row["tau"], row["domain_id"]
        if key in seen:
            raise ValueError(f"Duplicate domain/run: {key}; select one input configuration")
        seen.add(key)
    return rows


def calculate_domain_size_proportions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize each status population separately at every threshold.

    Empty populations
    are masked (undefined proportions), with n=0 and undefined diagnostics.
    """
    import numpy as np

    result = {}
    for dataset, taus in DOMAIN_SIZE_TAUS.items():
        columns = [[row for row in rows if row["dataset"] == dataset and row["tau"] == tau] for tau in taus]
        if not all(columns):
            raise ValueError(f"Missing domain data for {dataset}, tau(s) {[tau for tau, column in zip(taus, columns) if not column]}")
        matrices, counts, diagnostics = {}, {}, [[] for _ in taus]
        for label, field in (("observed", "size_observed"), ("inferred", "size_inferred")):
            population = [[row for row in column if row[field] is not None] for column in columns]
            counts[field] = [len(column) for column in population]
            matrices[field] = np.column_stack([
                np.bincount([row[field] for row in column], minlength=14)[1:14] / len(column)
                if column else np.full(13, np.nan)
                for column in population
            ])
            for index, column in enumerate(population):
                sizes = [row[field] for row in column]
                stats = (f"median={statistics.median(sizes):g}, universal={sizes.count(13)/len(sizes):.6f}"
                         if sizes else "median=N/A, universal=N/A")
                diagnostics[index].append(f"{label}: n={len(sizes)}, {stats}")
        result[dataset] = {**matrices, "counts": counts}
        for tau, stats in zip(taus, diagnostics):
            print(f"{dataset} | τ={tau} | " + " | ".join(stats))
    return result


def plot_domain_size_heatmap(
    proportions: dict[str, Any], output_path: str | Path, min_n: int = 100,
) -> Path:
    """Save the compact 1 x 3 layout, keeping status populations unchanged.

    The observed distribution at tau_0 supplies the single Obs. column.
    Empty populations remain masked; hatching marks counts below min_n.
    """
    import matplotlib as mpl
    import numpy as np
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    if min_n < 0:
        raise ValueError("min_n must be nonnegative")
    matrices = {
        dataset: np.column_stack((proportions[dataset]["size_observed"][:, 0],
                                  proportions[dataset]["size_inferred"]))
        for dataset in DOMAIN_SIZE_TAUS
    }
    finite_values = [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    vmax = max((values.max() for values in finite_values if values.size), default=1.0)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    style = {"font.family": "serif", "font.size": 8, "mathtext.fontset": "dejavuserif",
             "text.usetex": False, "pdf.fonttype": 42, "axes.titlesize": 8,
             "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 8,
             "hatch.linewidth": 0.4}
    with mpl.rc_context(style):
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7, 2.4))
        # Reserve space for vertical counts and the shared x label.
        fig.subplots_adjust(left=.075, right=.865, bottom=.34, top=.88, wspace=.09)
        try:
            for panel, (dataset, taus) in enumerate(DOMAIN_SIZE_TAUS.items()):
                ax = axes[panel]
                matrix = matrices[dataset]
                counts = [proportions[dataset]["counts"]["size_observed"][0],
                          *proportions[dataset]["counts"]["size_inferred"]]
                mesh = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto",
                                 interpolation="nearest", extent=(-.5, 6.5, .5, 13.5),
                                 cmap="cividis", norm=norm, rasterized=False)
                ax.set_title(dataset)
                ax.set_xticks(range(7), labels=["Obs.", *taus])
                ax.set_yticks(range(1, 14))
                ax.tick_params(axis="x", length=2, pad=2)
                ax.tick_params(axis="y", length=2 if panel == 0 else 0,
                               labelleft=panel == 0)
                if panel == 0:
                    ax.set_ylabel(r"Domain Size")
                for column, n in enumerate(counts):
                    # Vertical text keeps large n legible without overlapping.
                    ax.text(column, -.17, f"n={n}", transform=ax.get_xaxis_transform(),
                            ha="center", va="top", rotation=90, fontsize=6, clip_on=False)
                    if n < min_n:
                        ax.add_patch(Rectangle((column-.5, .5), 1, 13,
                                               facecolor="none", edgecolor="gray",
                                               linewidth=0, hatch="///", zorder=2))
                    for size_index, value in enumerate(matrix[:, column]):
                        if np.isfinite(value) and value >= .10:
                            rgb = mesh.cmap(norm(value))[:3]
                            # WCAG relative luminance of the rendered cell.
                            linear = [c / 12.92 if c <= .04045 else ((c+.055)/1.055)**2.4 for c in rgb]
                            luminance = sum(c*w for c, w in zip(linear, (.2126, .7152, .0722)))
                            ax.text(column, size_index+1, f"{value:.2f}", fontsize=5,
                                    ha="center", va="center",
                                    color="black" if luminance > .179 else "white", zorder=3)
                ax.axvline(.5, color="white", linewidth=2, zorder=4)
                ax.axhline(12.5, color="white", linewidth=.6, alpha=.8, zorder=4)
            fig.supxlabel(r"Threshold $\tau$ (inferred domains)", x=.47, y=.015, fontsize=8)
            colorbar_ax = fig.add_axes([.885, .34, .018, .54])
            colorbar = fig.colorbar(mesh, cax=colorbar_ax)
            colorbar.set_label("Proportion of domains")
            # Ticks are proportions, not gamma-transformed coordinates.
            ticks = [value for value in (0, .05, .1, .25, .5, .75) if value <= vmax]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([f"{value:g}" for value in ticks])
            fig.text(.885, .26, f"hatched: n < {min_n}", ha="left", va="top", fontsize=6)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, format="pdf", bbox_inches="tight")
        finally:
            plt.close(fig)
    return output

def export_eligible_observed_domain_sizes(
    conservative_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    composition: str = "reduced_composition",
) -> Path:
    """Export size counts/proportions for observed premises at each tau.

    Uses after_oracle and the conservative propagator's source filter:
    Observed status, positive support ratio, and support denominator >= tau.
    Only occurrence=0 runs are selected. Size counts positive-support relations
    in the eligible domain, not zero-support placeholders. Directed pairs are
    counted separately. Eligibility means usable as a premise, not proof that
    the pair participated in a successful inference. Empty populations have
    undefined proportions (N/A). Missing or duplicate configurations are errors.
    """
    from .qcn_generator2 import support_counts, support_ratio

    root = Path(conservative_dir) if conservative_dir is not None else results_root / "conservative"
    output = Path(output_path) if output_path is not None else root / "eligible_observed_domain_sizes.txt"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if composition not in {"reduced_composition", "non_reduced_composition"}:
        raise ValueError(f"Unknown composition: {composition}")
    configurations = {}
    for path in sorted(root.rglob("qcn2_*.json")):
        match = re.fullmatch(r"qcn2_(Q\d+)(?:_(non_reduced_composition|reduced_composition))?", path.stem)
        if match is None or match[1] not in DOMAIN_SIZE_QIDS:
            continue
        variant = match[2] or next((part for part in path.parts if part in {"reduced_composition", "non_reduced_composition"}), None)
        config = next((m for part in path.parts if (m := re.fullmatch(r"occ_([\d.p]+)_min_(\d+)_timeout_\d+", part))), None)
        if variant != composition or config is None or float(config[1].replace("p", ".")) != 0:
            continue
        dataset, tau = DOMAIN_SIZE_QIDS[match[1]], int(config[2])
        if tau not in DOMAIN_SIZE_TAUS[dataset]:
            continue
        key = dataset, tau
        if key in configurations:
            raise ValueError(f"Duplicate configuration for {key}")
        configurations[key] = path
    missing = [(dataset, tau) for dataset, taus in DOMAIN_SIZE_TAUS.items() for tau in taus
               if (dataset, tau) not in configurations]
    if missing:
        raise ValueError(f"Missing configurations: {missing}")
    lines = ["# Observed premises: support > 0 and total >= tau; occurrence threshold = 0.",
             "# Size = number of positive-support relations; directed pairs counted separately.",
             "dataset\ttau\tn_eligible\tsize\tcount\tproportion"]
    for dataset, taus in DOMAIN_SIZE_TAUS.items():
        for tau in taus:
            with configurations[dataset, tau].open(encoding="utf-8") as handle:
                payload = json.load(handle)
            counts = [0] * 14
            for domain in payload["after_oracle"].values():
                if domain.get("status") != "Observed":
                    continue
                positive, eligible = 0, False
                for relation in domain["relations"].values():
                    support = relation.get("support")
                    ratio = support_ratio(support)
                    if ratio is not None and ratio > 0:
                        positive += 1
                        eligible |= support_counts(support)[1] >= tau
                if eligible:
                    if not 1 <= positive <= 13:
                        raise ValueError(f"Invalid observed domain size: {positive}")
                    counts[positive] += 1
            del payload
            n = sum(counts)
            for size in range(1, 14):
                proportion = f"{counts[size] / n:.10f}" if n else "N/A"
                lines.append(f"{dataset}\t{tau}\t{n}\t{size}\t{counts[size]}\t{proportion}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def count_distinct_scores_in_universal_inferred_domains(
    domains: Mapping[str, Any],
) -> dict[int, int]:
    """Count inferred domains with exactly 13 positive-score relations.

    Distinctness uses exact numeric equality, without rounding. Missing,
    nonnumeric, nonfinite, boolean and nonpositive scores are excluded.
    Directed domains are counted separately, as stored in the QCN.
    """
    counts = dict.fromkeys(range(1, 14), 0)
    for domain in domains.values():
        if not isinstance(domain, Mapping) or domain.get("status") not in {"Inferred", "Infered"}:
            continue
        relations = domain.get("relations")
        if not isinstance(relations, Mapping):
            continue
        scores = [relation.get("inferred_score") for relation in relations.values()
                  if isinstance(relation, Mapping)]
        positive_scores = [score for score in scores
                           if isinstance(score, (int, float)) and not isinstance(score, bool)
                           and math.isfinite(score) and score > 0]
        if len(positive_scores) == 13:
            counts[len(set(positive_scores))] += 1
    return counts


def plot_conservative_distinct_score_histograms(
    conservative_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    composition: str = "reduced_composition",
) -> dict[str, dict[int, int]]:
    """Write one 300-dpi PNG per dataset and tau, plus a TXT of all counts.

    Traverse all QCN datasets and thresholds under conservative, selecting
    occurrence=0 and one composition variant to avoid mixing configurations.
    Domain size uses the same positive inferred_score rule as the heatmap.
    Return histograms keyed by source path relative to conservative.
    """
    root = Path(conservative_dir) if conservative_dir is not None else results_root / "conservative"
    destination = Path(output_dir) if output_dir is not None else root / "distinct_score_histograms"
    if not root.is_dir():
        raise FileNotFoundError(root)
    if composition not in {"reduced_composition", "non_reduced_composition"}:
        raise ValueError(f"Unknown composition: {composition}")
    configurations = {}
    for path in sorted(root.rglob("qcn2_*.json")):
        match = re.fullmatch(r"qcn2_(Q\d+)(?:_(non_reduced_composition|reduced_composition))?", path.stem)
        if match is None:
            continue
        variant = match[2] or next((part for part in path.parts if part in {"reduced_composition", "non_reduced_composition"}), None)
        config = next((m for part in path.parts if (m := re.fullmatch(r"occ_([\d.p]+)_min_(\d+)_timeout_\d+", part))), None)
        if variant != composition or config is None or float(config[1].replace("p", ".")) != 0:
            continue
        key = match[1], int(config[2])
        if key in configurations:
            raise ValueError(f"Duplicate dataset/tau configuration: {key}")
        configurations[key] = path
    if not configurations:
        raise ValueError(f"No matching conservative QCNs in {root}")
    destination.mkdir(parents=True, exist_ok=True)
    histograms = {}
    lines = ["dataset\ttau\tn_domains\tn_distinct_scores\tcount"]
    style = {"font.family": "serif", "font.size": 8, "text.usetex": False,
             "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"}
    for (qid, tau), path in sorted(configurations.items()):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        counts = count_distinct_scores_in_universal_inferred_domains(payload["after_propagation"])
        del payload
        histograms[str(path.relative_to(root))] = counts
        dataset = DOMAIN_SIZE_QIDS.get(qid, qid)
        n = sum(counts.values())
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
            try:
                ax.bar(list(counts), list(counts.values()), color="0.4", width=.8)
                ax.set_xticks(range(1, 14))
                ax.set_xlim(.5, 13.5)
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
                ax.set_xlabel("Number of distinct scores")
                ax.set_ylabel("Number of domains")
                ax.set_title(dataset + rf" — $\tau={tau}$ (n={n})")
                output = destination / f"distinct_scores_{qid}_tau_{tau}.png"
                fig.savefig(output, format="png", dpi=300, bbox_inches="tight")
            finally:
                plt.close(fig)
        for distinct, count in counts.items():
            lines.append(f"{dataset}\t{tau}\t{n}\t{distinct}\t{count}")
        print(f"{dataset} | τ={tau} | inferred domains of size 13: {n} | {output}")
    (destination / "distinct_score_histograms.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return histograms


DISTINCT_SCORE_HIGH_TAUS = {"Country": 190, "Politician": 389, "Musical group": 60}


def load_distinct_score_comparison(input_path: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Read tab/comma-separated counts and normalize the six selected series.

    Missing size bins are zero-filled. Each selected series must have a positive
    consistent n_domains, unique bins in 1..13, and counts summing to n_domains.
    """
    with Path(input_path).open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t" if "\t" in header else ",")
        required = {"dataset", "tau", "n_domains", "n_distinct_scores", "count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns: {sorted(required)}")
        series = {dataset: {} for dataset in DISTINCT_SCORE_HIGH_TAUS}
        for row in reader:
            dataset = row["dataset"]
            if dataset not in series:
                continue
            tau = int(row["tau"])
            if tau not in (0, DISTINCT_SCORE_HIGH_TAUS[dataset]):
                continue
            n, size, count = (int(row[field]) for field in ("n_domains", "n_distinct_scores", "count"))
            if n <= 0 or not 1 <= size <= 13 or count < 0:
                raise ValueError(f"Invalid histogram row: {row}")
            entry = series[dataset].setdefault(tau, {"n_domains": n, "counts": {}})
            if entry["n_domains"] != n or size in entry["counts"]:
                raise ValueError(f"Inconsistent population or duplicate bin: {row}")
            entry["counts"][size] = count
    for dataset, high_tau in DISTINCT_SCORE_HIGH_TAUS.items():
        for tau in (0, high_tau):
            if tau not in series[dataset]:
                raise ValueError(f"Missing histogram: {dataset}, tau={tau}")
            entry = series[dataset][tau]
            if sum(entry["counts"].values()) != entry["n_domains"]:
                raise ValueError(f"Counts do not sum to n_domains: {dataset}, tau={tau}")
            entry["proportions"] = [entry["counts"].get(size, 0) / entry["n_domains"] for size in range(1, 14)]
    return series


def plot_distinct_score_comparison(
    input_path: str | Path,
    output_path: str | Path = "distinct_scores.pdf",
) -> Path:
    """Compare tau=0 (filled bars) and high tau (outline bars) in three panels."""
    series = load_distinct_score_comparison(input_path)
    # Set one limit from all six series; per-panel limits freeze shared autoscaling.
    max_proportion = max(
        max(entry["proportions"])
        for dataset_series in series.values() for entry in dataset_series.values()
    )
    y_max = min(1.05, max_proportion * 1.25)
    style = {"font.family": "serif", "font.size": 8, "mathtext.fontset": "dejavuserif",
             "text.usetex": False, "pdf.fonttype": 42, "axes.titlesize": 8,
             "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7, 2), layout="constrained")
        try:
            for ax, (dataset, high_tau) in zip(axes, DISTINCT_SCORE_HIGH_TAUS.items()):
                for tau in (0, high_tau):
                    entry = series[dataset][tau]
                    ax.bar(range(1, 14), entry["proportions"], width=.8,
                           facecolor="0.65" if tau == 0 else "none",
                           edgecolor="0.65" if tau == 0 else "0.1",
                           linewidth=.5 if tau == 0 else 1,
                           label=rf"$\tau={tau}$ (n={entry['n_domains']})")
                ax.set_title(dataset)
                ax.set_xticks(range(1, 14))
                ax.set_xlim(.4, 13.6)
                ax.tick_params(axis="x", length=2, pad=2, labelsize=7)
                ax.legend(fontsize=6, frameon=False, loc="upper right")
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].set_ylim(0, y_max)
            axes[0].set_ylabel("Proportion of universal domains")
            fig.supxlabel("Number of distinct scores", fontsize=8)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, format="pdf", bbox_inches="tight")
        finally:
            plt.close(fig)
    return output


def domain_size_heatmap_main(argv: list[str] | None = None) -> None:
    """Generate the compact v2 domain-size figure."""
    parser = argparse.ArgumentParser(description="Domain-size heatmaps in ACM/TheWebConf format.")
    parser.add_argument("--input", type=Path, default=results_root / "conservative", help="Conservative directory or CSV/TSV file.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Output directory for figures and reports.")
    parser.add_argument("--composition", choices=("reduced_composition", "non_reduced_composition"), default="reduced_composition", help="QCN JSON composition variant to load.")
    parser.add_argument("--min-n", type=int, default=100, help="Hatch columns with counts below this threshold (default: 100).")
    parser.add_argument("--distinct-score-comparison", type=Path, metavar="TXT", help="Compare tau=0 and high tau from distinct_score_histograms.txt.")
    parser.add_argument("--distinct-score-histograms", action="store_true", help="Distinct-score histograms for inferred domains of size 13, by dataset and tau.")
    parser.add_argument("--eligible-observed-report", action="store_true", help="Export eligible observed-premise sizes to TXT without plotting.")
    args = parser.parse_args(argv)
    if args.min_n < 0:
        parser.error("--min-n must be nonnegative")
    try:
        if args.distinct_score_comparison is not None:
            output = plot_distinct_score_comparison(
                args.distinct_score_comparison, args.output_dir / "distinct_scores.pdf",
            )
            print(f"Figure saved: {output}")
            return
        if args.distinct_score_histograms:
            plot_conservative_distinct_score_histograms(
                args.input, args.output_dir / "distinct_score_histograms", args.composition,
            )
            return
        if args.eligible_observed_report:
            output = export_eligible_observed_domain_sizes(
                args.input, args.output_dir / "eligible_observed_domain_sizes.txt", args.composition,
            )
            print(f"Report saved: {output}")
            return
        rows = load_domain_size_rows(args.input, args.composition)
        summary = calculate_domain_size_proportions(rows)
        output = plot_domain_size_heatmap(
            summary, args.output_dir / "heatmap_domain_sizes_v2.pdf", min_n=args.min_n,
        )
        print(f"Figure saved: {output}")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    domain_size_heatmap_main()
