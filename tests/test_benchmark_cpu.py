from types import SimpleNamespace

import pytest

from tclkg import benchmark_cpu
from tclkg import qcn_generator2 as generator


def test_oracle_recomputed_five_times_without_cache(monkeypatch):
    entities = {"e": SimpleNamespace(triples_per_p={"a": [1], "b": [2]})}
    initial = {("a", "b"): {}, ("b", "a"): {}}
    calls = []
    indexes = []
    monkeypatch.setattr(generator, "read_KG", lambda database: (entities, ["a", "b"]))
    monkeypatch.setattr(generator, "build_entity_network", lambda **kwargs: initial)

    def build_index(loaded, properties):
        assert loaded is entities
        assert properties == ["a", "b"]
        index = object()
        indexes.append(index)
        return index

    def query(pair, loaded, index):
        calls.append((pair, index))
        return {"relations": {"before": {"support": [1, 1]}}}

    monkeypatch.setattr(generator, "build_oracle_property_index", build_index)
    monkeypatch.setattr(generator, "Query", query)
    ticks = iter(range(10))
    monkeypatch.setattr(benchmark_cpu.time, "process_time", lambda: next(ticks))

    result = benchmark_cpu.benchmark_oracle_cpu("Q42")

    assert result == {"runs_cpu_s": [1] * 5, "mean_cpu_s": 1}
    assert len(indexes) == 5
    assert calls == [(("a", "b"), index) for index in indexes]
    assert initial == {("a", "b"): {}, ("b", "a"): {}}


@pytest.mark.parametrize("selected", [None, ["WD27M"]])
def test_cli_selects_data_without_batch(tmp_path, monkeypatch, selected):
    import json
    from tclkg import generator_common

    data = tmp_path / "data"
    for database in ("Q6256", "WD27M"):
        source = data / database
        source.mkdir(parents=True)
        (source / "data.quintuplet").touch()
    (data / "unrelated").mkdir()
    monkeypatch.setattr(generator_common, "ROOT_DATA", data)
    monkeypatch.chdir(tmp_path)
    calls = []

    def benchmark(database, taus, occurrence, reduced):
        calls.append((database, taus, occurrence, reduced))
        return {"oracle": {"mean_cpu_s": 2}, "propagation_mean_cpu_s": {"tau_0": 3}}

    monkeypatch.setattr(benchmark_cpu, "benchmark_database_cpu", benchmark)
    argv = ["benchmark_cpu"]
    if selected:
        argv += ["--datasets", *selected, "--taus", "0", "10"]
    monkeypatch.setattr("sys.argv", argv)
    benchmark_cpu.main()
    result = json.loads((tmp_path / "temps_cpu.json").read_text())
    assert list(result) == (selected or ["Q6256", "WD27M"])
    assert calls == ([("WD27M", [0, 10], 0.0, True)] if selected else [
        ("Q6256", [0, 10, 59, 110, 160, 190], 0.0, True),
        ("WD27M", [0], 0.0, True),
    ])


def test_propagation_reuses_oracle_with_independent_copies(monkeypatch):
    network = {("a", "b"): {"original": True}}
    calls = []
    monkeypatch.setattr(benchmark_cpu, "_benchmark_oracle_cpu",
                        lambda database: ({"mean_cpu_s": 2}, network))

    def propagate(qcn, **kwargs):
        assert qcn == network
        calls.append(kwargs["domain_reliability_threshold"])
        qcn.clear()
        return qcn

    monkeypatch.setattr(generator, "propagateAndFilterConservative", propagate)
    ticks = iter(range(20))
    monkeypatch.setattr(benchmark_cpu.time, "process_time", lambda: next(ticks))
    result = benchmark_cpu.benchmark_database_cpu("WD27M", [0, 10])
    assert calls == [0] * 5 + [10] * 5
    assert network == {("a", "b"): {"original": True}}
    assert result["oracle"] == {"mean_cpu_s": 2}
    assert result["propagation_mean_cpu_s"] == {"occ_0_min_0": 1, "occ_0_min_10": 1}


def test_cli_rejects_missing_dataset_before_benchmark(tmp_path, monkeypatch, capsys):
    from tclkg import generator_common

    monkeypatch.setattr(generator_common, "ROOT_DATA", tmp_path)
    monkeypatch.setattr("sys.argv", ["benchmark_cpu", "--datasets", "missing"])
    with pytest.raises(SystemExit, match="2"):
        benchmark_cpu.main()
    assert "Bases absentes" in capsys.readouterr().err
