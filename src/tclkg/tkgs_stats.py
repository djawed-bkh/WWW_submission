import argparse
from collections import Counter
from itertools import combinations
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from .generator_common import read_kg as read_KG

def compute_property_pair_percentiles(tkg: str) -> dict[str, float]:
    """Return p10, p25, p50, p75 and p90 of property-pair evidence counts.

    For an unordered property pair ``(p1, p2)``, its evidence count follows the
    oracle's ``nbr_total`` semantics: for every entity containing both
    properties, add ``len(entity[p1]) * len(entity[p2])``.  Pairs with no
    common entity therefore contribute a zero to the distribution.

    Args:
        tkg: Knowledge-graph directory name understood by :func:`read_KG`.

    Raises:
        ValueError: If the knowledge graph contains fewer than two properties.
    """
    entities, properties = read_KG(tkg, print_properties_count=False)
    sorted_properties = sorted(properties)
    if len(sorted_properties) < 2:
        raise ValueError(
            f"Cannot compute property-pair percentiles for {tkg!r}: "
            "at least two properties are required."
        )

    pair_counts: list[int] = []
    for property1, property2 in combinations(sorted_properties, 2):
        pair_count = 0
        for entity in entities.values():
            property1_triples = entity.triples_per_p.get(property1)
            property2_triples = entity.triples_per_p.get(property2)
            if property1_triples and property2_triples:
                pair_count += len(property1_triples) * len(property2_triples)
        if pair_count > 0:
            pair_counts.append(pair_count)

    percentile_levels = (80, 85, 90, 95, 99)
    percentile_values = np.percentile(pair_counts, percentile_levels)
    return {
        f"p{level}": float(value)
        for level, value in zip(percentile_levels, percentile_values, strict=True)
    }


def compute_property_cooccurence_distribution(tkg: str) -> dict[int, int]:
    """Return a distribution of property co-occurrence counts.

    For an unordered property pair ``(p1, p2)``, its co-occurrence count is the
    number of entities that contain both properties. Pairs with no common
    entity and frequencies that were never observed are omitted, so every
    value in the returned dictionary is strictly positive.

    Args:
        tkg: Knowledge-graph directory name understood by :func:`read_KG`.

    Raises:
        ValueError: If the knowledge graph contains fewer than two properties.
    """
    entities, properties = read_KG(tkg, print_properties_count=False)
    sorted_properties = sorted(properties)
    if len(sorted_properties) < 2:
        raise ValueError(
            f"Cannot compute property-pair co-occurrence distribution for {tkg!r}: "
            "at least two properties are required."
        )

    cooccurence_counts: list[int] = []
    for property1, property2 in combinations(sorted_properties, 2):
        cooccurence_count = 0
        for entity in entities.values():
            if (
                property1 in entity.triples_per_p
                and property2 in entity.triples_per_p
            ):
                cooccurence_count += 1
        if cooccurence_count > 0:
            cooccurence_counts.append(cooccurence_count)

    if not cooccurence_counts:
        return {}

    frequencies = Counter(cooccurence_counts)
    return dict(sorted(frequencies.items()))



def compare_pair_counting(tkg: str) -> dict[str, list[dict]]: 
    """Compare fact-pair count (n_ij) vs entity count (m_ij) per property pair,
    returning only pairs whose multiplicity ratio differs from 1.0."""
    entities, properties = read_KG(tkg, print_properties_count=False)
    sorted_properties = sorted(properties)
    resultsdifferent = []
    resultsequals = []
    for property1, property2 in combinations(sorted_properties, 2):
        n_ij = 0   # comptage combinatoire actuel
        m_ij = 0   # comptage par entite
        for entity in entities.values():
            t1 = entity.triples_per_p.get(property1)
            t2 = entity.triples_per_p.get(property2)
            if t1 and t2:
                n_ij += len(t1) * len(t2)
                m_ij += 1
        if n_ij > 0 and n_ij != m_ij:
            resultsdifferent.append({
                "pair": (property1, property2),
                "n_ij": n_ij,
                "m_ij": m_ij,
                "ratio": n_ij / m_ij if m_ij else None,
            })
        elif n_ij > 0 and n_ij == m_ij:
            resultsequals.append({
                "pair": (property1, property2),
                "n_ij": n_ij,
                "m_ij": m_ij,
                "ratio": n_ij / m_ij if m_ij else None,
            })
    return {"different": resultsdifferent}



def histogram_property_pair_cooccurence(tkg: str):
    """Return a bar plot of property-pair co-occurrences in bins of 10.

    The x-axis contains co-occurrence intervals and the y-axis contains the
    number of unordered property pairs in each interval.
    """
    distribution = compute_property_cooccurence_distribution(tkg)
    figure, axis = plt.subplots()

    if not distribution:
        axis.set_xlabel("Nombre de co-occurrences")
        axis.set_ylabel("Nombre de paires de proprietes")
        axis.set_title(f"Co-occurrences des paires de proprietes - {tkg}")
        return axis

    bins: Counter[int] = Counter()
    for cooccurence_count, pair_count in distribution.items():
        bins[(cooccurence_count // 10) * 10] += pair_count

    bin_starts = sorted(bins)
    labels = [f"{start}-{start + 9}" for start in bin_starts]
    axis.bar(labels, [bins[start] for start in bin_starts])
    axis.set_xlabel("Nombre de co-occurrences")
    axis.set_ylabel("Nombre de paires de proprietes")
    axis.set_title(f"Co-occurrences des paires de proprietes (échelle logarithmique) - {tkg}")
    axis.tick_params(axis="x",labelsize=8, rotation=90)
    axis.set_yscale('log', base=2)
    figure.tight_layout()
    plt.savefig(f"cooccurrences_{tkg}.png", dpi=1000)
    with open(f"cooccurrences_{tkg}.txt", "w", encoding="utf-8") as histogram_file:
        histogram_file.write("cooccurrences_bin\tnombre_paires_proprietes\n")
        for start in bin_starts:
            histogram_file.write(f"{start}-{start + 9}\t{bins[start]}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calcule des statistiques de cooccurrence entre propriétés d'un TKG."
    )
    parser.add_argument(
        "tkg",
        help="Nom du knowledge graph à lire dans le répertoire data (ex. Q0001).",
    )
    parser.add_argument(
        "--metric",
        choices=("cooccurrence", "percentiles", "all"),
        default="cooccurrence",
        help="Statistique à afficher (défaut : cooccurrence).",
    )
    args = parser.parse_args()

    if args.metric in {"cooccurrence", "all"}:
        print(f"Distribution des cooccurrences — {args.tkg}")
        pprint(compute_property_cooccurence_distribution(args.tkg), sort_dicts=True)

    if args.metric in {"percentiles", "all"}:
        print(f"Percentiles des paires de propriétés — {args.tkg}")
        pprint(compute_property_pair_percentiles(args.tkg), sort_dicts=False)

    return 0






if __name__ == "__main__":
    raise SystemExit(main())
