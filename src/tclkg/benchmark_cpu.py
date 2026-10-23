"""Mesurer séparément cinq oracles par BDD et cinq propagations par configuration."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from itertools import combinations
from pathlib import Path


def _benchmark_oracle_cpu(database: str) -> tuple[dict, dict]:
    """Recalculer l'oracle sans cache, hors lecture et copie du réseau initial.

    Le chronométrage couvre la préparation de l'index, les requêtes et les
    domaines converses, comme la phase oracle de learner2. Aucun seuil de
    propagation ni mode de composition n'intervient.
    """
    from . import qcn_generator2 as generator

    entities, initial_properties = generator.read_KG(database)
    initial = generator.build_entity_network(
        properties=initial_properties,
        default_constraints=generator.ALLEN_RELATIONS,
        top_k_fraction=generator.PROPERTY_SUPPORT_THRESHOLD,
        entities=entities,
    )
    properties = sorted({prop for pair in initial for prop in pair})
    samples = []
    for run in range(1, 6):
        qcn = copy.deepcopy(initial)
        started = time.process_time()
        # Même préparation que dans la branche sans cache de learner2.
        kept_properties = set(properties)
        sum(
            len(triples)
            for entity in entities.values()
            for prop, triples in entity.triples_per_p.items()
            if prop in kept_properties
        )
        index = generator.build_oracle_property_index(entities, properties)
        for p1, p2 in combinations(properties, 2):
            domain = generator.Query((p1, p2), entities, index)
            if domain is not None:
                qcn[p1, p2] = domain
                qcn[p2, p1] = generator.fast_converse_domain(domain)
                if not generator.relation_entries(domain):
                    raise ValueError(f"Oracle incohérent pour {database}: {(p1, p2)}")
        elapsed = time.process_time() - started
        samples.append(elapsed)
        print(f"{database} | oracle {run}/5 : {elapsed:.6f} s CPU", flush=True)
        del index
    return {"runs_cpu_s": samples, "mean_cpu_s": statistics.mean(samples)}, qcn


def benchmark_oracle_cpu(database: str) -> dict:
    """Mesurer cinq oracles sans cache et retourner leurs temps CPU."""
    result, _ = _benchmark_oracle_cpu(database)
    return result


DEFAULT_TAUS = {
    "Q6256": [0, 10, 59, 110, 160, 190],
    "Q215380": [0, 10, 30, 60, 100, 150],
    "Q82955": [0, 14, 42, 128, 389, 1185],
}


def benchmark_database_cpu(database, taus, occurrence=0.0, reduced_composition=True):
    """Mesurer l'oracle puis propager des copies indépendantes du réseau obtenu."""
    from .qcn_generator2 import propagateAndFilterConservative

    oracle, network = _benchmark_oracle_cpu(database)
    result = {"oracle": oracle, "propagation_mean_cpu_s": {}, "propagation": {}}
    for tau in taus:
        samples = []
        for run in range(1, 6):
            qcn = copy.deepcopy(network)
            started = time.process_time()
            propagated = propagateAndFilterConservative(
                qcn, domain_reliability_threshold=tau,
                relation_occurrence_threshold=occurrence,
                reduced_composition=reduced_composition,
            )
            elapsed = time.process_time() - started
            samples.append(elapsed)
            print(f"{database} | tau={tau} | propagation {run}/5 : {elapsed:.6f} s CPU", flush=True)
            del qcn, propagated
        key = f"occ_{occurrence:g}_min_{tau}"
        result["propagation"][key] = {
            "tau": tau, "occurrence": occurrence,
            "reduced_composition": reduced_composition,
            "runs_cpu_s": samples, "mean_cpu_s": statistics.mean(samples),
        }
        result["propagation_mean_cpu_s"][key] = statistics.mean(samples)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("temps_cpu.json"),
                        help="Fichier JSON de sortie (défaut : ./temps_cpu.json).")
    parser.add_argument("--datasets", nargs="+", metavar="BDD",
                        help="Bases à mesurer (défaut : toutes les bases de data).")
    parser.add_argument("--taus", nargs="+", type=int,
                        help="Seuils communs à toutes les bases ; sinon seuils connus par base, ou 0.")
    parser.add_argument("--occurrence", type=float, default=0.0)
    parser.add_argument("--reduced-composition", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    from .generator_common import ROOT_DATA

    databases = sorted(
        path.name for path in ROOT_DATA.glob("*")
        if path.is_dir() and any(
            (path / filename).is_file()
            for filename in ("data.quintuplet", "train_cst_knowledge.quintuplet")
        )
    )
    if args.datasets:
        missing = sorted(set(args.datasets) - set(databases))
        if missing:
            parser.error(f"Bases absentes de {ROOT_DATA} : {', '.join(missing)}")
        databases = list(dict.fromkeys(args.datasets))
    if not databases:
        parser.error(f"Aucune base avec un fichier quintuplet dans {ROOT_DATA}")
    if args.taus is not None and any(tau < 0 for tau in args.taus):
        parser.error("Les seuils tau doivent être positifs ou nuls.")
    if not 0 <= args.occurrence <= 1:
        parser.error("--occurrence doit être compris entre 0 et 1.")
    output = args.output.expanduser()
    if output.exists():
        parser.error(f"Le fichier existe déjà : {output}. Choisis un autre --output.")
    print(f"Données : {ROOT_DATA}", flush=True)
    results = {}
    for database in databases:
        taus = list(dict.fromkeys(args.taus or DEFAULT_TAUS.get(database, [0])))
        print(f"BDD : {database} | seuils tau : {taus}", flush=True)
        results[database] = benchmark_database_cpu(
            database, taus, args.occurrence, args.reduced_composition,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Moyennes en secondes CPU : {output}", flush=True)


if __name__ == "__main__":
    main()
