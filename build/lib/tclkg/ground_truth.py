from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .generator_common import PROJECT_ROOT


DEFAULT_KGS = ("Q6256", "Q215380")
SLOW_KGS = ("Q82955",)
SECTIONS = ("initial", "after_oracle", "after_propagation")
MANIFEST_VERSION = 1
FLOAT_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def result_path_for(kg: str, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "Results" / kg / f"qcn2_{kg}.json"


def manifest_path_for(kg: str, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "ground_truth" / "manifests" / f"{kg}.json"


def find_input_file(kg: str, project_root: Path = PROJECT_ROOT) -> Path | None:
    kg_dir = project_root / "data" / kg
    for filename in ("data.quintuplet", "train_cst_knowledge.quintuplet"):
        candidate = kg_dir / filename
        if candidate.exists():
            return candidate
    return None


def load_qcn_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"QCN result must be a JSON object: {path}")
    return data


def _canonical_support(value: Any) -> Any:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], int)
        or not isinstance(value[1], int)
        or value[0] < 0
        or value[1] < 0
        or value[0] > value[1]
    ):
        raise ValueError(f"Invalid support value: {value!r}")
    return [value[0], value[1]]


def _canonical_domain(domain: Mapping[str, Any]) -> dict[str, Any]:
    relations = domain.get("relations")
    if not isinstance(relations, Mapping):
        raise ValueError("QCN domain must contain a 'relations' mapping")

    canonical_relations: dict[str, Any] = {}
    for relation in sorted(relations):
        payload = relations[relation]
        if not isinstance(payload, Mapping):
            raise ValueError(f"Relation payload must be a mapping: {relation}")
        canonical_relations[relation] = {
            "support": _canonical_support(payload.get("support")),
        }
    return {"status": domain.get("status"), "relations": canonical_relations}


def canonicalize_section(section: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for pair in sorted(section, key=str):
        pair_key = f"{pair[0]}___{pair[1]}" if isinstance(pair, tuple) else str(pair)
        canonical[pair_key] = _canonical_domain(section[pair])
    return canonical


def canonical_section_hash(section: Mapping[str, Any]) -> str:
    canonical = canonicalize_section(section)
    return sha256_text(json.dumps(canonical, sort_keys=True, separators=(",", ":")))


def semantic_stats(section: Mapping[str, Any]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    support_counts = {
        "none": 0,
        "zero": 0,
        "positive": 0,
        "negative": 0,
    }
    positive_per_domain: list[int] = []
    domains_with_observed = 0
    domains_with_positive_support = 0
    total_relations = 0

    for domain in section.values():
        if not isinstance(domain, Mapping):
            raise ValueError("QCN domain must be a mapping")
        relations = domain.get("relations")
        if not isinstance(relations, Mapping):
            raise ValueError("QCN domain must contain a 'relations' mapping")

        domain_is_observed = domain.get("status") == "Observed"
        domain_positive_count = 0
        for payload in relations.values():
            if not isinstance(payload, Mapping):
                raise ValueError("Relation payload must be a mapping")
            total_relations += 1

            support = payload.get("support")
            if (
                not isinstance(support, (tuple, list))
                or len(support) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in support)
                or support[0] < 0
                or support[1] < 0
                or support[0] > support[1]
            ):
                raise ValueError(f"Invalid support value: {support!r}")
            occurrences, tested_pairs = support

            status_key = (
                "Observed" if domain_is_observed and occurrences > 0 else
                "None" if domain_is_observed else
                "Inferred"
            )
            status_counts[status_key] = status_counts.get(status_key, 0) + 1

            if tested_pairs == 0:
                support_counts["none"] += 1
            elif occurrences == 0:
                support_counts["zero"] += 1
            else:
                support_counts["positive"] += 1
                domain_positive_count += 1

        if domain_is_observed:
            domains_with_observed += 1
        if domain_positive_count > 0:
            domains_with_positive_support += 1
        positive_per_domain.append(domain_positive_count)

    domain_count = len(section)
    return {
        "domains": domain_count,
        "relations": total_relations,
        "status_counts": dict(sorted(status_counts.items())),
        "support_counts": support_counts,
        "domains_with_observed": domains_with_observed,
        "domains_with_positive_support": domains_with_positive_support,
        "positive_relations_per_domain": {
            "min": min(positive_per_domain) if positive_per_domain else 0,
            "max": max(positive_per_domain) if positive_per_domain else 0,
            "avg": (
                sum(positive_per_domain) / len(positive_per_domain)
                if positive_per_domain
                else 0.0
            ),
        },
    }


def transition_stats(
    after_oracle: Mapping[str, Any], after_propagation: Mapping[str, Any]
) -> dict[str, int]:
    none_after_oracle_total = 0
    none_to_inferred = 0
    observed_after_oracle_total = 0
    domains_without_positive_after_oracle = 0
    domains_without_positive_to_positive = 0
    initial_relation_total = 0
    propagation_relation_total = 0

    for pair, oracle_domain in after_oracle.items():
        oracle_relations = oracle_domain.get("relations", {})
        propagation_domain = after_propagation.get(pair, {})
        propagation_relations = propagation_domain.get("relations", {})
        if not isinstance(oracle_relations, Mapping) or not isinstance(
            propagation_relations, Mapping
        ):
            raise ValueError(f"Invalid QCN relations for pair: {pair}")

        oracle_is_observed = oracle_domain.get("status") == "Observed"
        propagation_is_observed = propagation_domain.get("status") == "Observed"
        initial_relation_total += len(oracle_relations)
        propagation_relation_total += len(propagation_relations)

        for relation, oracle_payload in oracle_relations.items():
            if not isinstance(oracle_payload, Mapping):
                raise ValueError(f"Invalid oracle payload for {pair}/{relation}")
            oracle_support = oracle_payload.get("support")
            oracle_relation_observed = (
                oracle_is_observed
                and isinstance(oracle_support, (list, tuple))
                and len(oracle_support) == 2
                and oracle_support[0] > 0
            )

            propagation_payload = propagation_relations.get(relation)
            if propagation_payload is None:
                propagation_status = None
            elif propagation_is_observed:
                propagation_status = "Observed"
            else:
                propagation_status = "Inferred"

            if not oracle_relation_observed:
                none_after_oracle_total += 1
                if propagation_status == "Inferred":
                    none_to_inferred += 1
            else:
                observed_after_oracle_total += 1

        if not oracle_is_observed:
            domains_without_positive_after_oracle += 1
            if propagation_is_observed:
                domains_without_positive_to_positive += 1

    return {
        "none_after_oracle_total": none_after_oracle_total,
        "none_to_inferred": none_to_inferred,
        "observed_after_oracle_total": observed_after_oracle_total,
        "after_oracle_relation_total": initial_relation_total,
        "after_propagation_relation_total": propagation_relation_total,
        "domains_without_positive_after_oracle": domains_without_positive_after_oracle,
        "domains_without_positive_to_positive": domains_without_positive_to_positive,
    }


def build_manifest(kg: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_path = result_path_for(kg, project_root)
    if not result_path.exists():
        raise FileNotFoundError(f"Missing QCN result: {result_path}")

    data = load_qcn_result(result_path)
    section_hashes: dict[str, str] = {}
    section_stats: dict[str, Any] = {}
    for section_name in SECTIONS:
        section = data.get(section_name)
        if not isinstance(section, Mapping):
            raise ValueError(
                f"Missing or invalid section '{section_name}' in {result_path}"
            )
        section_hashes[section_name] = canonical_section_hash(section)
        section_stats[section_name] = semantic_stats(section)

    input_file = find_input_file(kg, project_root)
    input_info = None
    if input_file is not None:
        input_info = {
            "path": str(input_file.relative_to(project_root)),
            "size_bytes": input_file.stat().st_size,
            "sha256": sha256_file(input_file),
        }

    lock_path = project_root / "uv.lock"
    return {
        "manifest_version": MANIFEST_VERSION,
        "kg": kg,
        "optional_slow": kg in SLOW_KGS,
        "input": input_info,
        "result": {
            "path": str(result_path.relative_to(project_root)),
            "size_bytes": result_path.stat().st_size,
            "raw_sha256": sha256_file(result_path),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "uv_lock_sha256": sha256_file(lock_path) if lock_path.exists() else None,
        },
        "learner": {
            "command": f"python -m tclkg.qcn_generator2 {kg} <timeout>",
            "property_support_threshold": 0.4,
            "float_tolerance": FLOAT_TOLERANCE,
        },
        "section_hashes": section_hashes,
        "section_stats": section_stats,
        "oracle_to_propagation_transition_stats": transition_stats(
            data["after_oracle"], data["after_propagation"]
        ),
    }


def write_manifest(kg: str, project_root: Path = PROJECT_ROOT) -> Path:
    manifest = build_manifest(kg, project_root)
    path = manifest_path_for(kg, project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _load_manifest(kg: str, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = manifest_path_for(kg, project_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def verify_manifest(kg: str, project_root: Path = PROJECT_ROOT) -> list[str]:
    expected = _load_manifest(kg, project_root)
    actual = build_manifest(kg, project_root)
    errors: list[str] = []

    for path in (
        ("manifest_version",),
        ("kg",),
        ("input", "sha256"),
        ("section_hashes",),
        ("section_stats",),
        ("oracle_to_propagation_transition_stats",),
    ):
        expected_value = _nested_get(expected, path)
        actual_value = _nested_get(actual, path)
        if expected_value != actual_value:
            errors.append(
                f"{kg}: mismatch at {'.'.join(path)}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def _nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _selected_kgs(args: argparse.Namespace) -> tuple[str, ...]:
    if args.kg:
        return tuple(args.kg)
    if args.include_slow:
        return DEFAULT_KGS + SLOW_KGS
    return DEFAULT_KGS


def generate(kgs: Iterable[str], project_root: Path = PROJECT_ROOT) -> None:
    for kg in kgs:
        path = write_manifest(kg, project_root)
        print(f"generated {path}")


def verify(kgs: Iterable[str], project_root: Path = PROJECT_ROOT) -> bool:
    ok = True
    for kg in kgs:
        errors = verify_manifest(kg, project_root)
        if errors:
            ok = False
            for error in errors:
                print(error)
        else:
            print(f"verified {kg}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify Learner 2 ground truth manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("generate", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--kg", action="append", help="KG to process. Can be repeated."
        )
        subparser.add_argument(
            "--include-slow",
            action="store_true",
            help="Include optional slow KGs such as Q82955.",
        )

    args = parser.parse_args()
    kgs = _selected_kgs(args)
    if args.command == "generate":
        generate(kgs)
    elif args.command == "verify":
        raise SystemExit(0 if verify(kgs) else 1)


if __name__ == "__main__":
    main()
