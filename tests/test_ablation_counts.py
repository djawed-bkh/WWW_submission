import csv
import json

from tclkg import random_experiments as rexp
from tclkg import qcn_generator2 as qgen
from tclkg.Latex_generator import ablation_tables


def test_relation_counts_distinguish_missing_relations_from_wrong_predictions():
    observed = {"a": {"relations": {
        "before": {"support": [5, 10]},
        "meets": {"support": [5, 10]},
        "after": {"support": [0, 10]},
        "during": {"support": [1, 10]},
    }}}
    for score, missing in [(0.0, 1), (False, 2), (float("nan"), 2), (float("inf"), 2)]:
        predicted = {"a": {"relations": {
            "before": {"inferred_score": score},
            "after": {"inferred_score": 1.0},
        }}}
        assert rexp._ablation_relation_counts(observed, predicted, 0.5, 10) == (2, missing)
    assert rexp._ablation_relation_counts(observed, {}, 0.5, 11) == (0, 0)
    for predictions in ({}, {"before": {"inferred_score": None}},
                        {"before": {"inferred_score": float("nan")}},
                        {"before": {"inferred_score": False}}):
        records = []
        assert rexp._ablation_relation_counts(
            observed, {"a": {"relations": predictions}}, 0.5, 10,
            missing_records=records,
        ) == (2, 0)
        assert records == []


def test_ablation_exports_counts_and_latex(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rexp, "results_root", tmp_path)
    folder = tmp_path / "reduced_composition/Q42/occ_0p5_min_10_timeout_600"
    folder.mkdir(parents=True)
    path = folder / "qcn2_Q42_reduced_composition.json"
    domain = {"status": "Observed", "relations": {
        "before": {"support": [5, 10]}, "meets": {"support": [5, 10]},
    }}
    path.write_text(json.dumps({"after_oracle": {f"p{i}___q{i}": domain for i in range(100)}}))
    rounds = 0

    def propagate(qcn, **kwargs):
        nonlocal rounds
        rounds += 1
        for pair in list(qcn):
            if qcn[pair]["status"] is None:
                if rounds <= 5:
                    qcn[pair] = {"relations": {"before": {"inferred_score": 1.0}}}
                else:
                    del qcn[pair]
        return qcn

    monkeypatch.setattr(qgen, "propagateAndFilterConservative", propagate)
    scores = rexp.run_ablation_studies_of_inference(n=1)
    assert list(scores.values()) == [0.5]
    with (tmp_path / "ablation_inference_counts.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert {k: int(row[k]) for k in row if k.startswith("n_")} == {
        "n_observed": 100, "n_observed_relations": 200,
        "n_masked": 10, "n_unreached": 5,
        "n_masked_relations": 20, "n_missing_relations": 5,
    }
    archive = tmp_path / "MRR/Q42/occ_0p5_min_10_timeout_600"
    summary = json.loads((archive / "summary.json").read_text())
    assert summary["mrr"] == 0.5
    assert summary["n_missing_relations"] == 5
    masked_rounds = json.loads((archive / "masked_rounds.json").read_text())
    inferred_rounds = json.loads((archive / "inferred_rounds.json").read_text())
    missing = json.loads((archive / "missing_relations.json").read_text())
    assert missing["n_masked_relations"] == 20
    assert missing["n_missing_relations"] == len(missing["missing_relations"]) == 5
    for record in missing["missing_relations"]:
        assert record["property_pair_occurrence"] == 10
        assert record["relation_occurrence"] == 5
        assert record["occurrence_ratio"] == 0.5
        assert record["round"] <= 5
        assert record["relation"] == "meets"
        original = masked_rounds[record["round"] - 1]
        assert record["property_pair"] in original["selected_pairs"]
    assert summary["seed"] == 42
    assert summary["source_json"] == str(path)
    assert summary["reduced_composition"] is True
    assert (archive / "results.csv").is_file()
    assert len(masked_rounds) == len(inferred_rounds) == 10
    assert not list(archive.glob("round_*.json"))
    selected = set()
    for i in range(1, 11):
        masked = masked_rounds[i - 1]
        inferred = inferred_rounds[i - 1]
        assert masked["round"] == inferred["round"] == i
        pair, = masked["selected_pairs"]
        assert pair not in selected
        selected.add(pair)
        assert len(masked["masked_qcn"]) == 100
        assert masked["ground_truth"][pair] == domain
        assert masked["masked_qcn"][pair]["status"] is None
        assert all(r["support"] == [0, 0]
                   for r in masked["masked_qcn"][pair]["relations"].values())
        untouched = next(key for key in masked["masked_qcn"] if key != pair)
        assert masked["masked_qcn"][untouched] == domain
        if i <= 5:
            assert inferred["inferred_qcn"][pair]["relations"] == {
                "before": {"inferred_score": 1.0}}
        else:
            assert pair not in inferred["inferred_qcn"]
    assert len(rexp.Number_of_relevant_observed_domains()) == 1
    log = tmp_path / "results.txt"
    log.write_text(capsys.readouterr().out)
    parsed = ablation_tables.parse_results(log)
    table = ablation_tables.build_latex_table("Q42", parsed["Q42"])
    assert "0.500 & 200 & 5/20" in table
    assert "Miss./Mask." in table
    other_configuration = archive.parent / "occ_0p9_min_10_timeout_600"
    other_configuration.mkdir()
    (other_configuration / "summary.json").write_text('{"mrr": 0.9}')
    (archive / "notes.txt").write_text("keep")
    (archive / "round_01_masked.json").write_text("{}")
    (archive / "round_01_inferred.json").write_text("{}")
    zero = rexp.ablation_study_of_inference(str(path), 0, return_details=True)
    assert zero["n_observed_relations"] == 200
    assert zero["n_masked_relations"] == zero["n_missing_relations"] == 0
    assert not any(folder.is_dir() for folder in archive.iterdir())
    assert not list(archive.glob("round_*.json"))
    assert json.loads((archive / "missing_relations.json").read_text())["missing_relations"] == []
    assert json.loads((archive / "summary.json").read_text())["n_masked"] == 0
    assert json.loads((archive / "masked_rounds.json").read_text()) == []
    assert json.loads((archive / "inferred_rounds.json").read_text()) == []
    assert (archive / "notes.txt").read_text() == "keep"
    assert json.loads((other_configuration / "summary.json").read_text()) == {"mrr": 0.9}


def test_legacy_ablation_results_remain_readable(tmp_path):
    log = tmp_path / "old.txt"
    log.write_text("non_reduced_composition/Q42/occ_0p5_min_10_timeout_600/qcn.json: weighted MRR = 0.75\n")
    data = ablation_tables.parse_results(log)
    assert "0.750 & -- & --" in ablation_tables.build_latex_table("Q42", data["Q42"])


def test_batch_only_runs_conservative_and_legacy_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(rexp, "results_root", tmp_path)
    paths = {
        f"Q42/{mode}/occ_0p5_min_10_timeout_600/qcn2_Q42.json": 100
        for mode in ("conservative", "two_stage", "iterative", "Iterative", "fix_point", "strass")
    }
    legacy = "reduced_composition/Q42/occ_0p5_min_10_timeout_600/qcn2_Q42.json"
    paths[legacy] = 100
    monkeypatch.setattr(rexp, "Number_of_relevant_observed_domains", lambda: paths)
    calls = []

    def ablation(path, **kwargs):
        calls.append(str(path))
        return dict(mrr=1.0, n_observed=100, n_observed_relations=100,
                    n_masked=100, n_unreached=0, n_masked_relations=100,
                    n_missing_relations=0)

    monkeypatch.setattr(rexp, "ablation_study_of_inference", ablation)
    scores = rexp.run_ablation_studies_of_inference()
    expected = {next(iter(paths)), legacy}
    assert set(scores) == expected
    assert set(calls) == {str(tmp_path / path) for path in expected}
