from __future__ import annotations

import json
import csv

from tclkg import random_experiments as rexp


def test_check_consistency_loads_after_oracle_and_runs_classic_pc(tmp_path) -> None:
    qcn_path = tmp_path / "sample_qcn.json"
    qcn_payload = {
        "after_oracle": {
            "p1___p2": {"relations": {"before": {"support": [1, 1]}}},
            "p2___p1": {"relations": {"after": {"support": [1, 1]}}},
            "p2___p3": {"relations": {"before": {"support": [1, 1]}}},
            "p3___p2": {"relations": {"after": {"support": [1, 1]}}},
            "p1___p3": {"relations": {"before": {"support": [1, 5]}, "meets": {"support": [9, 10]}}},
            "p3___p1": {"relations": {"after": {"support": [1, 5]}, "met_by": {"support": [9, 10]}}},
        }
    }
    qcn_path.write_text(json.dumps(qcn_payload), encoding="utf-8")

    assert rexp.checkConsistency(str(qcn_path), threshold=0.5) is True


def test_check_consistency_returns_false_when_classic_pc_collapses(tmp_path) -> None:
    qcn_path = tmp_path / "inconsistent_qcn.json"
    qcn_payload = {
        "after_oracle": {
            "p1___p2": {"relations": {"before": {"support": [1, 1]}}},
            "p2___p1": {"relations": {"after": {"support": [1, 1]}}},
            "p2___p3": {"relations": {"before": {"support": [1, 1]}}},
            "p3___p2": {"relations": {"after": {"support": [1, 1]}}},
            "p1___p3": {"relations": {"after": {"support": [1, 1]}}},
            "p3___p1": {"relations": {"before": {"support": [1, 1]}}},
        }
    }
    qcn_path.write_text(json.dumps(qcn_payload), encoding="utf-8")

    assert rexp.checkConsistency(str(qcn_path), threshold=0.5) is False

def test_domain_size_statistics_reads_both_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rexp, "results_root", tmp_path)
    domains = {
        "a___b": {"status": "Inferred", "relations": {"before": {"inferred_score": 0.7}}},
        "a___c": {"status": "Inferred", "relations": {
            str(i): {"inferred_score": 0.7} for i in range(13)
        }},
        "b___c": {"status": "Inferred", "relations": {
            str(i): {"inferred_score": 0.7 if i else 0.2} for i in range(13)
        }},
        "c___d": {"status": None, "relations": {}},
    }
    for mode in ("reduced_composition", "non_reduced_composition"):
        directory = tmp_path / mode / "Q42" / "occ_0p3_min_10_timeout_600"
        directory.mkdir(parents=True)
        (directory / f"qcn2_Q42_{mode}.json").write_text(json.dumps({
            "after_oracle": {pair: {"status": None} for pair in domains},
            "after_propagation": domains if mode == "reduced_composition" else {},
        }))
    summary = rexp.after_propagation_domain_sizes_statistics()
    assert len(summary) == 2
    rows = {row["version"]: row for row in summary.values()}
    reduced = rows["reduced_composition"]
    assert reduced["n_domains"] == 4
    assert reduced["inferred_domains/total_None_after_oracle"] == "3/4"
    assert reduced["%domaines_informatifs"] == 200 / 3
    assert reduced["avg_domain_sizes"] == 9
    assert reduced["median_domain_sizes"] == 13
    assert reduced["min_domain_sizes"] == 1
    assert reduced["occurence_thresh"] == 0.3
    assert reduced["support_thresh"] == 10
    assert rows["non_reduced_composition"]["avg_domain_sizes"] is None
    assert (tmp_path / "after_propagation_domain_sizes_statistics.csv").is_file()
    assert rexp.after_propagation_domain_sizes_statistics(["Q99"]) == {}


def test_inference_scores_only_use_informative_domains(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rexp, "results_root", tmp_path)

    def domain(scores, status="Inferred"):
        return {"status": status, "relations": {
            str(i): {"inferred_score": score} for i, score in enumerate(scores)
        }}

    for mode in ("reduced_composition", "non_reduced_composition"):
        folder = tmp_path / mode / "Q42" / "occ_0p3_min_10_timeout_600"
        folder.mkdir(parents=True)
        propagated = {
            "a": domain([0.2, 0.8]),
            "b": domain([0.4] * 12 + [0.6]),
            "uniform": domain([1.0] * 13),
            "observed": domain([0.0], "Observed"),
            "empty": domain([]),
            "missing": domain([None]),
            "boolean": domain([True]),
            "infinite": domain([float("inf")]),
        } if mode == "reduced_composition" else {}
        (folder / f"qcn2_Q42_{mode}.json").write_text(json.dumps({
            "after_propagation": propagated,
        }))
    summary = rexp.after_propagation_domain_inference_scores_statistics()
    rows = {row["version"]: row for row in summary.values()}
    reduced = rows["reduced_composition"]
    assert reduced["min_inference_score"] == 0.2
    assert abs(reduced["avg_best_candidate_score"] - 0.7) < 1e-12
    assert abs(reduced["avg_domain_mean_score"] - (0.5 + 5.4 / 13) / 2) < 1e-12
    assert rows["non_reduced_composition"]["avg_best_candidate_score"] is None
    assert rows["non_reduced_composition"]["avg_domain_mean_score"] is None
    assert reduced["max_inference_score"] == 0.8
    assert rows["non_reduced_composition"]["min_inference_score"] is None
    with (tmp_path / "after_propagation_domain_inference_scores_statistics.csv").open() as handle:
        csv_rows = {row["version"]: row for row in csv.DictReader(handle)}
    assert len(csv_rows) == 2
    assert csv_rows["reduced_composition"]["min_inference_score"] == "0.2"
    assert abs(float(csv_rows["reduced_composition"]["avg_best_candidate_score"]) - 0.7) < 1e-12
    assert abs(float(csv_rows["reduced_composition"]["avg_domain_mean_score"]) - (0.5 + 5.4 / 13) / 2) < 1e-12
    assert csv_rows["non_reduced_composition"]["min_inference_score"] == ""
    assert "min_inference_score: 0.2" in capsys.readouterr().out
    assert rexp.after_propagation_domain_inference_scores_statistics(["Q99"]) == {}


def test_conservative_domain_status_statistics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rexp, "results_root", tmp_path)
    relative = "Q42/reduced_composition/occ_0_min_0_timeout_600/qcn2_Q42.json"
    path = tmp_path / "conservative" / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "after_oracle": {"a___b": {"status": "Observed"}},
        "after_propagation": {
            "a___b": {"status": "Observed"},
            "b___a": {"status": "Observed"},
            "a___c": {"status": "Inferred"},
            "c___a": {"status": None},
            "b___c": {},
            "c___b": {"status": "Unknown"},
        },
    }))
    (path.parent / "empty.json").write_text(json.dumps({"after_propagation": {}}))
    (path.parent / "metadata.json").write_text(json.dumps({"initial": {}}))
    outside = tmp_path / "other.json"
    outside.write_text(path.read_text())

    result = rexp.conservative_domain_status_statistics()

    assert len(result) == 2
    assert result[relative] == {
        "total_domains": 6,
        "observed_count": 2, "observed_percentage": 100 / 3,
        "inferred_count": 1, "inferred_percentage": 100 / 6,
        "unknown_count": 3, "unknown_percentage": 50,
    }
    empty = result[relative.rsplit("/", 1)[0] + "/empty.json"]
    assert empty["total_domains"] == 0
    assert all(empty[f"{category}_percentage"] is None
               for category in ("observed", "inferred", "unknown"))
