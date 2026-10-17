"""Launch a parameter sweep for Learner 2, with bounded parallelism.

The default settings below can be edited directly.  A JSON configuration file
can also be supplied with ``--config``; see ``--write-example-config``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# DEFAULT_DATABASES = ("Q6256", "Q215380", "Q82955")
TIMEOUT = 86_400
MAX_PARALLEL = 6
REDUCED_COMPOSITION = True
PROPAGATION_MODE = "conservative"
# for testing
# DEFAULT_DATABASES = ("Q0000",)
# OCCURRENCE_THRESHOLDS = (0.5,)
# MIN_TOTALS = (2,)

# OCCURRENCE_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9)
# MIN_TOTALS = (1, 10, 50, 100, 150, 200)
configurations: dict[str, dict[str, Any]] = {
    "Q6256": {
        "occurrence_thresholds": [0.0, 0.3, 0.5, 0.8, 0.9],
        "min_total": [0, 10, 59, 110, 160, 190],
        "timeout": TIMEOUT,
    },
    "Q215380": {
        "occurrence_thresholds": [0.0, 0.3, 0.5, 0.8, 0.9],
        "min_total": [0, 10, 30, 60, 100, 150],
        "timeout": TIMEOUT,
    },
    "Q82955": {
        "occurrence_thresholds": [0.0, 0.3, 0.5, 0.8, 0.9],
        "min_total": [0, 14, 42, 128, 389, 1185],
        "timeout": TIMEOUT,
    },
    # "Q0000": {"occurrence_thresholds": [0.3, 0.5,0.8,0.9], "min_total": [1], "timeout": TIMEOUT},
}


def _configuration_name(
    occurrence_threshold: float, min_total: int, timeout: int
) -> str:
    threshold_label = f"{occurrence_threshold:g}".replace(".", "p")
    return f"occ_{threshold_label}_min_{min_total}_timeout_{timeout}"


def _default_configurations() -> dict[str, list[dict[str, Any]]]:
    """Expand the configured threshold grid independently for each database."""
    configurations_by_database: dict[str, list[dict[str, Any]]] = {}
    for database, settings in configurations.items():
        occurrence_thresholds = settings["occurrence_thresholds"]
        minimum_totals = settings["min_total"]
        timeout = int(settings["timeout"])
        configurations_by_database[database] = [
            {
                "name": _configuration_name(occurrence_threshold, minimum, timeout),
                "occurrence_threshold": occurrence_threshold,
                "min_total": minimum,
                "timeout": timeout,
                "reduced_composition": settings.get(
                    "reduced_composition", REDUCED_COMPOSITION
                ),
                "propagation_mode": settings.get("propagation_mode", PROPAGATION_MODE),
            }
            for occurrence_threshold in occurrence_thresholds
            for minimum in minimum_totals
        ]
    return configurations_by_database


@dataclass(frozen=True)
class Run:
    database: str
    name: str
    occurrence_threshold: float
    min_total: int
    timeout: int
    output_dir: Path
    log_path: Path
    reduced_composition: bool = REDUCED_COMPOSITION
    propagation_mode: str = "conservative"
    max_stages: int | None = None
    oracle_cache: bool = True


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"Nom de configuration invalide : {value!r}")
    return normalized


def _validate_configuration(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"configurations[{index}] doit être un objet JSON")  # noqa: TRY004 -- CLI configuration errors use ValueError.

    try:
        name = _safe_name(str(raw["name"]))
        occurrence_threshold = float(raw["occurrence_threshold"])
        min_total = int(raw["min_total"])
        timeout = int(raw["timeout"])
    except KeyError as error:
        raise ValueError(
            f"Champ manquant dans configurations[{index}] : {error.args[0]}"
        ) from error

    max_stages = raw.get("max_stages")
    if max_stages is not None and (type(max_stages) is not int or max_stages < 1):
        raise ValueError("max_stages doit être un entier positif")
    reduced_composition = raw.get("reduced_composition", REDUCED_COMPOSITION)
    propagation_mode = raw.get("propagation_mode", "conservative")
    if propagation_mode not in (
        "conservative",
        "two_stage",
        "iterative",
        "fix_point",
        "strass",
    ):
        raise ValueError(
            "propagation_mode doit être conservative, two_stage, iterative ou fix_point"
        )
    if not isinstance(reduced_composition, bool):
        raise ValueError(  # noqa: TRY004 -- CLI configuration errors use ValueError.
            "reduced_composition doit être un booléen JSON (true ou false)"
        )

    if not 0.0 <= occurrence_threshold <= 1.0:
        raise ValueError("occurrence_threshold doit être compris entre 0 et 1")
    if min_total < 0:
        raise ValueError("min_total doit être positif ou nul")
    if timeout <= 0:
        raise ValueError("timeout doit être strictement positif")

    return {
        "name": name,
        "occurrence_threshold": occurrence_threshold,
        "min_total": min_total,
        "timeout": timeout,
        "reduced_composition": reduced_composition,
        "propagation_mode": propagation_mode,
        "max_stages": max_stages,
    }


def _load_sweep(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return _default_configurations()

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("La configuration racine doit être un objet JSON")  # noqa: TRY004 -- CLI configuration errors use ValueError.

    raw_databases = payload.get("databases")
    raw_configurations = payload.get("configurations")
    if not isinstance(raw_databases, list) or not raw_databases:
        raise ValueError("'databases' doit être une liste non vide")
    if not isinstance(raw_configurations, list) or not raw_configurations:
        raise ValueError("'configurations' doit être une liste non vide")

    databases = [_safe_name(str(database)) for database in raw_databases]
    configurations = [
        _validate_configuration(configuration, index)
        for index, configuration in enumerate(raw_configurations)
    ]
    return {
        database: [dict(configuration) for configuration in configurations]
        for database in databases
    }


def _build_runs(
    configurations_by_database: dict[str, list[dict[str, Any]]],
    batch_dir: Path,
) -> list[Run]:
    runs: list[Run] = []
    for database, configurations in configurations_by_database.items():
        names = [
            (
                configuration.get("propagation_mode", "conservative"),
                str(configuration["name"]),
            )
            for configuration in configurations
        ]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Chaque configuration doit avoir un nom unique pour {database}"
            )
        for configuration in configurations:
            name = str(configuration["name"])
            propagation_mode = str(
                configuration.get("propagation_mode", "conservative")
            )
            if propagation_mode not in (
                "conservative",
                "two_stage",
                "iterative",
                "fix_point",
                "strass",
            ):
                raise ValueError(f"Mode de propagation invalide : {propagation_mode}")
            run_dir = batch_dir / database / propagation_mode / name
            runs.append(
                Run(
                    database=database,
                    name=name,
                    occurrence_threshold=float(configuration["occurrence_threshold"]),
                    min_total=int(configuration["min_total"]),
                    timeout=int(configuration["timeout"]),
                    reduced_composition=configuration.get(
                        "reduced_composition", REDUCED_COMPOSITION
                    ),
                    propagation_mode=propagation_mode,
                    max_stages=configuration.get("max_stages"),
                    output_dir=run_dir,
                    log_path=run_dir / "run.log",
                )
            )
    return runs


def _run_one(run: Run) -> tuple[Run, int]:
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "tclkg.launcher",
        "--_run-one",
        run.database,
        str(run.occurrence_threshold),
        str(run.min_total),
        str(run.timeout),
        str(run.output_dir),
    ]
    command.append(
        "--reduced-composition"
        if run.reduced_composition
        else "--no-reduced-composition"
    )
    command.extend(["--propagation-mode", run.propagation_mode])
    if run.max_stages is not None:
        command.extend(["--max-stages", str(run.max_stages)])
    if run.oracle_cache:
        command.extend(
            ["--oracle-cache-dir", str(run.output_dir.parents[2] / ".oracle-cache")]
        )
    environment = os.environ.copy()
    source_dir = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_dir, environment.get("PYTHONPATH")) if part
    )

    with run.log_path.open("w", encoding="utf-8") as log:
        log.write(f"Commande : {' '.join(command)}\n")
        log.write(f"Démarrage : {datetime.now().isoformat()}\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=run.timeout,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTimeout après {run.timeout}s\n")
            return_code = 124
        log.write(f"\nFin : {datetime.now().isoformat()}\n")
        log.write(f"Code de retour : {return_code}\n")
    return run, return_code


def _child_main(
    arguments: list[str],
    reduced_composition: bool = REDUCED_COMPOSITION,
    propagation_mode: str = "conservative",
    max_stages: int | None = None,
    oracle_cache_dir: Path | None = None,
) -> int:
    database, threshold, min_total, timeout, output_dir = arguments
    from . import qcn_generator2

    configuration_dir = Path(output_dir).resolve()
    staging_root = configuration_dir / ".generator-output"
    qcn_generator2.results_root = staging_root
    sys.argv = [
        "tclkg.qcn_generator2",
        database,
        threshold,
        min_total,
        timeout,
    ]
    sys.argv.append(
        "--reduced-composition" if reduced_composition else "--no-reduced-composition"
    )
    sys.argv.extend(["--propagation-mode", propagation_mode])
    if max_stages is not None:
        sys.argv.extend(["--max-stages", str(max_stages)])
    if oracle_cache_dir is not None:
        sys.argv.extend(["--oracle-cache-dir", str(oracle_cache_dir)])
    sys.argv.append("--_direct-learning")
    qcn_generator2.main()

    # Remonter les fichiers dans <BDD>/<mode>/<configuration>/<fichiers>.
    generated_database_dir = staging_root / database / propagation_mode
    if generated_database_dir.is_dir():
        for generated_path in generated_database_dir.iterdir():
            destination = configuration_dir / generated_path.name
            if destination.exists():
                raise FileExistsError(
                    f"Le fichier de sortie existe déjà : {destination}"
                )
            shutil.move(str(generated_path), str(destination))
        generated_database_dir.rmdir()
        generated_database_dir.parent.rmdir()
        staging_root.rmdir()
    return 0


def _example_payload() -> dict[str, Any]:
    configurations_by_database = _default_configurations()
    database = next(iter(configurations_by_database))
    return {
        "databases": [database],
        "configurations": configurations_by_database[database],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lance une batterie de configurations Learner 2 pour chaque BDD."
    )
    parser.add_argument("--config", type=Path, help="Fichier JSON décrivant le sweep.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL,
        help="Nombre maximal de configurations exécutées simultanément.",
    )
    parser.add_argument(
        "--kg", action="append", help="Ne lancer que cette BDD (répétable)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Afficher les runs sans les lancer."
    )
    parser.add_argument(
        "--oracle-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Réutiliser l’oracle entre les configurations d’une BDD.",
    )
    parser.add_argument("--oracle-cache-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "parameter_sweeps",
        help="Répertoire racine des résultats et logs.",
    )
    parser.add_argument(
        "--write-example-config",
        type=Path,
        help="Écrire un exemple JSON puis quitter.",
    )
    parser.add_argument("--_run-one", nargs=5, help=argparse.SUPPRESS)
    parser.add_argument(
        "--reduced-composition",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignorer les compositions à 13 relations (prioritaire sur la configuration).",
    )
    parser.add_argument(
        "--propagation-mode",
        choices=("conservative", "two_stage", "iterative", "fix_point", "strass"),
        default=None,
        help="Mode de propagation (prioritaire sur la configuration).",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=None,
        help="Nombre maximal de stages Strass (défaut : jusqu’à stabilisation).",
    )
    args = parser.parse_args()
    if args.max_stages is not None and args.max_stages < 1:
        parser.error("--max-stages doit être positif")

    if args._run_one:
        return _child_main(
            args._run_one,
            REDUCED_COMPOSITION
            if args.reduced_composition is None
            else args.reduced_composition,
            args.propagation_mode or "conservative",
            args.max_stages,
            args.oracle_cache_dir,
        )
    if args.write_example_config:
        args.write_example_config.write_text(
            json.dumps(_example_payload(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Configuration écrite dans {args.write_example_config}")
        return 0
    if args.max_parallel <= 0:
        parser.error("--max-parallel doit être strictement positif")

    configurations_by_database = _load_sweep(args.config)
    if args.max_stages is not None:
        for configurations in configurations_by_database.values():
            for configuration in configurations:
                configuration["max_stages"] = args.max_stages
    if args.propagation_mode is not None:
        for configurations in configurations_by_database.values():
            for configuration in configurations:
                configuration["propagation_mode"] = args.propagation_mode
    if args.reduced_composition is not None:
        for configurations in configurations_by_database.values():
            for configuration in configurations:
                configuration["reduced_composition"] = args.reduced_composition
    if args.kg:
        requested = set(args.kg)
        missing = requested.difference(configurations_by_database)
        if missing:
            parser.error(
                f"BDD absente de la configuration : {', '.join(sorted(missing))}"
            )
        configurations_by_database = {
            database: configurations
            for database, configurations in configurations_by_database.items()
            if database in requested
        }

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = args.output_root.resolve() / batch_id
    runs = _build_runs(configurations_by_database, batch_dir)
    runs = [replace(run, oracle_cache=args.oracle_cache) for run in runs]
    print(f"Batch : {batch_dir}")
    print(f"Runs : {len(runs)} ; parallélisme maximal : {args.max_parallel}")
    for run in runs:
        print(
            f"- {run.database}/{run.name}: seuil={run.occurrence_threshold:g}, "
            f"min_total={run.min_total}, timeout={run.timeout}s, "
            f"reduced_composition={run.reduced_composition}, propagation_mode={run.propagation_mode}, max_stages={run.max_stages}"
        )
    if args.dry_run:
        return 0

    failures = 0
    ready = runs
    remaining_by_database: dict[str, list[Run]] = {}
    if args.oracle_cache:
        # Finish one requested run per KG before scheduling the rest, so parallel
        # workers reuse a complete cache instead of all recomputing the oracle.
        first = {}
        for run in runs:
            if run.database not in first:
                first[run.database] = run
                remaining_by_database[run.database] = []
            else:
                remaining_by_database[run.database].append(run)
        ready = list(first.values())
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(_run_one, run): run for run in ready}
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                del futures[future]
                run, return_code = future.result()
                status = "OK" if return_code == 0 else f"ÉCHEC ({return_code})"
                failures += return_code != 0
                print(f"[{status}] {run.database}/{run.name} — {run.log_path}")
                for pending in remaining_by_database.pop(run.database, []):
                    futures[executor.submit(_run_one, pending)] = pending

    print(f"Terminé : {len(runs) - failures} succès, {failures} échec(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
