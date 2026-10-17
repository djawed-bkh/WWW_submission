from __future__ import annotations

import copy
import importlib.util
import json
from itertools import combinations_with_replacement, product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tclkg import generator_common
from tclkg import qcn_generator2 as qgen
from tclkg.oracle_cache import oracle_cache_path, read_oracle_cache

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "performance_benchmark", ROOT / "scripts/benchmark_performance.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
CASES = json.loads((ROOT / "ground_truth/performance_cases.json").read_text())["cases"]


@pytest.mark.parametrize(
    ("case", "settings"),
    [(case, settings) for case in CASES for settings in case["expected"]],
    ids=[
        f"seed{case['seed']}-{settings['mode']}-{settings['min_total']}-"
        f"{settings['occurrence']}-{settings['reduced']}"
        for case in CASES
        for settings in case["expected"]
    ],
)
def test_propagation_matches_frozen_original(
    case: dict[str, Any], settings: dict[str, Any]
) -> None:
    network = {
        tuple(pair.split("___")): domain for pair, domain in case["input"].items()
    }
    original = copy.deepcopy(network)
    function = {
        "conservative": qgen.propagateAndFilterConservative,
        "two_stage": qgen.propagateAndFilterTwoStage,
        "iterative": qgen.propagateAndFilterIterative,
        "strass": qgen.propagateAndFilterStrass,
    }[settings["mode"]]
    result = function(
        network, settings["min_total"], settings["occurrence"], settings["reduced"]
    )
    assert BENCHMARK.section_fingerprint(result) == settings["fingerprint"]
    assert network == original


def test_cached_oracle_skips_queries_and_preserves_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data" / "Mini"
    data.mkdir(parents=True)
    (data / "data.quintuplet").write_text(
        "e1\tP1\tv\t2000-01-01\t2000-01-02\n"
        "e1\tP2\tv\t2000-01-03\t2000-01-04\n"
        "e2\tP2\tv\t2000-01-01\t2000-01-02\n"
        "e2\tP3\tv\t2000-01-03\t2000-01-04\n"
    )
    monkeypatch.setattr(generator_common, "ROOT_DATA", data.parent)
    entities, properties = qgen.read_KG("Mini")
    properties = sorted(properties)
    network = qgen.build_entity_network(
        properties, top_k_fraction=1.0, entities=entities
    )
    cache = oracle_cache_path("Mini", data.parent, tmp_path / "cache", 1.0)
    fresh = qgen.learner2(
        copy.deepcopy(network), entities, properties, oracle_cache_path=cache
    )
    assert fresh is not None
    prepared = read_oracle_cache(cache)
    assert prepared is not None

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Cache hit must not call the oracle or rebuild its index")

    monkeypatch.setattr(qgen, "Query", forbidden)
    monkeypatch.setattr(qgen, "build_oracle_property_index", forbidden)
    for occurrence, minimum in ((0.0, 0), (0.5, 1), (1.0, 3)):
        cached = qgen.learner2(
            {}, {}, properties, minimum, occurrence, prepared_oracle=prepared
        )
        assert cached is not None
        for section in ("initial", "after_oracle"):
            assert cached[section] == fresh[section]
        expected = qgen.propagateAndFilterConservative(
            fresh["after_oracle"], minimum, occurrence
        )
        assert cached["after_propagation"] == expected
        cached["after_oracle"].clear()
        assert prepared["after_oracle"] == fresh["after_oracle"]
    (data / "data.quintuplet").write_text("e1\tP1\tv\t2000-01-01\t2000-01-03\n")
    assert oracle_cache_path("Mini", data.parent, tmp_path / "cache", 1.0) != cache
    assert not list(cache.parent.glob("tmp*"))


def test_export_and_in_memory_analysis_match_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    network = {("é", "b"): qgen.with_relation_data({"before": (1, 2)})}
    result = {"initial": network, "after_oracle": network, "after_propagation": network}
    monkeypatch.setattr(qgen, "results_root", tmp_path)
    qgen.save_qcn_to_file(result, "Mini")
    path = tmp_path / "Mini/conservative/qcn2_Mini_reduced_composition.json"
    actual = json.loads(path.read_text())
    expected = {
        section: {"é___b": json.loads(json.dumps(network["é", "b"]))}
        for section in result
    }
    assert actual == expected
    assert qgen.analyze_qcn_scores(
        "Mini", learning_result=result
    ) == qgen.analyze_qcn_scores("Mini")


def test_launcher_runs_one_process_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tclkg import launcher

    data = tmp_path / "inputs" / "Mini"
    data.mkdir(parents=True)
    (data / "data.quintuplet").write_text(
        "e1\tP1\tv\t2000-01-01\t2000-01-02\ne1\tP2\tv\t2000-01-03\t2000-01-04\n"
    )
    monkeypatch.setenv("TCLKG_DATA_DIR", str(data.parent))
    previous = None
    for name in ("first", "second"):
        output = tmp_path / "batch" / "Mini" / "conservative" / name
        run = launcher.Run("Mini", name, 0.0, 0, 30, output, output / "run.log")
        _, code = launcher._run_one(run)
        assert code == 0, run.log_path.read_text()
        serialized = json.loads(
            (output / "qcn2_Mini_reduced_composition.json").read_text()
        )
        if previous is not None:
            assert serialized == previous
            assert "Oracle cache hit" in run.log_path.read_text()
            assert "DÉBUT DU CHARGEMENT" not in run.log_path.read_text()
        previous = serialized
    assert len(list((tmp_path / "batch" / ".oracle-cache").glob("*.pickle"))) == 1


def test_parallel_launcher_prepares_cache_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from tclkg import launcher

    data = tmp_path / "inputs" / "Mini"
    data.mkdir(parents=True)
    (data / "data.quintuplet").write_text(
        "e\tP1\tv\t2000-01-01\t2000-01-02\ne\tP2\tv\t2000-01-03\t2000-01-04\n"
    )
    config = tmp_path / "sweep.json"
    config.write_text(
        json.dumps(
            {
                "databases": ["Mini"],
                "configurations": [
                    {
                        "name": str(i),
                        "occurrence_threshold": i / 3,
                        "min_total": i,
                        "timeout": 30,
                    }
                    for i in range(3)
                ],
            }
        )
    )
    monkeypatch.setenv("TCLKG_DATA_DIR", str(data.parent))
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--config",
            str(config),
            "--max-parallel",
            "2",
            "--output-root",
            str(output),
        ],
    )
    assert launcher.main() == 0
    logs = list(output.rglob("run.log"))
    assert len(logs) == 3
    assert sum("Oracle cache hit" in path.read_text() for path in logs) == 2
    assert len(list(output.rglob("*.pickle"))) == 1


def test_launcher_reports_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from tclkg import launcher

    def expired(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired("learner", 1)

    monkeypatch.setattr(launcher.subprocess, "run", expired)
    output = tmp_path / "batch/Mini/conservative/timeout"
    run = launcher.Run("Mini", "timeout", 0.0, 0, 1, output, output / "run.log")
    _, code = launcher._run_one(run)
    assert code == 124
    assert "Timeout" in run.log_path.read_text()


def test_result_lock_includes_scores_stages_and_all_metadata() -> None:
    network = {
        ("a", "b"): {
            "status": "Inferred",
            "inference_stage": 2,
            "relations": {"before": {"support": (0, 0), "inferred_score": 0.75}},
        }
    }
    fingerprint = BENCHMARK.section_fingerprint(network)
    for change in ("score", "stage", "support", "relation", "status"):
        changed = copy.deepcopy(network)
        domain = changed["a", "b"]
        if change == "score":
            domain["relations"]["before"]["inferred_score"] = np.nextafter(
                0.75, 1.0
            ).item()
        elif change == "stage":
            domain["inference_stage"] = 3
        elif change == "support":
            domain["relations"]["before"]["support"] = (0, 1)
        elif change == "relation":
            domain["relations"]["after"] = domain["relations"].pop("before")
        else:
            domain["status"] = None
        assert BENCHMARK.section_fingerprint(changed) != fingerprint


def test_result_lock_uses_same_order_for_memory_and_json_pairs() -> None:
    network = {
        ("P1", "P2"): qgen.with_relation_data({"before": (1, 1)}),
        ("P10", "P2"): qgen.with_relation_data({"after": (1, 1)}),
    }
    serialized = {"___".join(pair): domain for pair, domain in network.items()}
    assert BENCHMARK.section_fingerprint(serialized) == BENCHMARK.section_fingerprint(
        network
    )


def test_single_interval_fast_path_preserves_point_intervals_and_all_relations() -> (
    None
):
    intervals = list(combinations_with_replacement(range(4), 2))
    for (a_start, a_end), (b_start, b_end) in product(intervals, repeat=2):
        expected = dict.fromkeys(qgen.ALLEN_RELATIONS, 0)
        left = SimpleNamespace(date=SimpleNamespace(start=a_start, end=a_end))
        right = SimpleNamespace(date=SimpleNamespace(start=b_start, end=b_end))
        qgen.add_allen_relation_matches(expected, left, right)
        for dtype in ("int64", "datetime64[D]"):
            actual = dict.fromkeys(qgen.ALLEN_RELATIONS, 0)
            pairs = qgen.add_vectorized_allen_relation_matches(
                actual,
                np.array([a_start], dtype=dtype),
                np.array([a_end], dtype=dtype),
                np.array([b_start], dtype=dtype),
                np.array([b_end], dtype=dtype),
            )
            assert pairs == 1
            assert actual == expected


def test_filter_does_not_alias_inputs_and_keeps_domain_metadata() -> None:
    network = {
        ("a", "b"): {
            "status": "Observed",
            "metadata": ["keep"],
            "relations": {"before": {"support": [2, 3]}, "after": {"support": [0, 3]}},
        }
    }
    original = copy.deepcopy(network)
    result = qgen.filter_unobserved_relations_after_oracle(network)
    assert set(result["a", "b"]["relations"]) == {"before"}
    result["a", "b"]["metadata"].append("new")
    result["a", "b"]["relations"]["before"]["support"][0] = 0
    assert network == original
