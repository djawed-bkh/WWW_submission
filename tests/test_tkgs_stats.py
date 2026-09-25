from types import SimpleNamespace

import pytest

from tclkg import tkgs_stats


def test_compute_property_pair_percentiles_uses_oracle_pair_counts(monkeypatch) -> None:
    entities = {
        "e1": SimpleNamespace(triples_per_p={"A": [1, 2], "B": [1], "C": [1]}),
        "e2": SimpleNamespace(triples_per_p={"A": [1], "B": [1, 2, 3]}),
    }
    monkeypatch.setattr(
        tkgs_stats,
        "read_KG",
        lambda _tkg, print_properties_count=False: (entities, {"A", "B", "C"}),
    )

    # Counts are A-B=5, A-C=2 and B-C=1.
    assert tkgs_stats.compute_property_pair_percentiles("test") == {
        "p10": pytest.approx(1.2),
        "p25": pytest.approx(1.5),
        "p50": pytest.approx(2.0),
        "p75": pytest.approx(3.5),
        "p90": pytest.approx(4.4),
    }


def test_compute_property_pair_percentiles_requires_two_properties(monkeypatch) -> None:
    monkeypatch.setattr(
        tkgs_stats,
        "read_KG",
        lambda _tkg, print_properties_count=False: ({}, {"A"}),
    )

    with pytest.raises(ValueError, match="at least two properties"):
        tkgs_stats.compute_property_pair_percentiles("test")


def test_compute_property_cooccurence_distribution_omits_empty_bins(monkeypatch) -> None:
    entities = {
        "e1": SimpleNamespace(triples_per_p={"A": [1], "B": [1]}),
        "e2": SimpleNamespace(triples_per_p={"A": [1], "B": [1]}),
        "e3": SimpleNamespace(triples_per_p={"A": [1], "B": [1]}),
        "e4": SimpleNamespace(triples_per_p={"C": [1], "D": [1]}),
    }
    monkeypatch.setattr(
        tkgs_stats,
        "read_KG",
        lambda _tkg, print_properties_count=False: (
            entities,
            {"A", "B", "C", "D"},
        ),
    )

    # A-B co-occurs 3 times and C-D once; no pair co-occurs exactly twice.
    assert tkgs_stats.compute_property_cooccurence_distribution("test") == {
        1: 1,
        3: 1,
    }


def test_compare_pair_counting_omits_pairs_with_ratio_one(monkeypatch) -> None:
    entities = {
        "e1": SimpleNamespace(triples_per_p={"A": [1], "B": [1], "C": [1, 2]}),
        "e2": SimpleNamespace(triples_per_p={"A": [1], "B": [1]}),
    }
    monkeypatch.setattr(
        tkgs_stats,
        "read_KG",
        lambda _tkg, print_properties_count=False: (entities, {"A", "B", "C"}),
    )

    assert tkgs_stats.compare_pair_counting("test") == [
        {"pair": ("A", "C"), "n_ij": 2, "m_ij": 1, "ratio": 2.0},
        {"pair": ("B", "C"), "n_ij": 2, "m_ij": 1, "ratio": 2.0},
    ]


def test_histogram_property_pair_cooccurence_uses_bins_of_ten(monkeypatch) -> None:
    monkeypatch.setattr(
        tkgs_stats,
        "compute_property_cooccurence_distribution",
        lambda _tkg: {1: 2, 10: 3, 19: 4, 20: 1},
    )

    axis = tkgs_stats.histogram_property_pair_cooccurence("test")

    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "0-9",
        "10-19",
        "20-29",
    ]
    assert [bar.get_height() for bar in axis.patches] == [2, 7, 1]
    assert axis.get_xlabel() == "Nombre de co-occurrences"
    assert axis.get_ylabel() == "Nombre de paires de proprietes"
