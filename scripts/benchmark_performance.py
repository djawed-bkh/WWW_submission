"""Freeze exact learner outputs and compare isolated, repeated performance runs.

Run ``uv run python scripts/benchmark_performance.py --help`` for usage.
The reference source and input checksums must be captured before optimization.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("initial", "after_oracle", "after_propagation")
MINIMUMS = {"Q6256": 10, "Q215380": 30, "Q82955": 128}
MAXIMUMS = {"Q6256": 190, "Q215380": 150, "Q82955": 1185}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def section_fingerprint(section: dict[Any, Any]) -> dict[str, Any]:
    """Hash every field, including inferred_score and inference_stage, exactly."""
    digest = hashlib.sha256()
    relations = 0
    for pair in sorted(
        section,
        key=lambda item: tuple(item.split("___", 1)) if isinstance(item, str) else item,
    ):
        key = "___".join(pair) if isinstance(pair, tuple) else pair
        domain = section[pair]
        encoded = json.dumps(
            [key, domain], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        relations += len(domain["relations"])
    return {
        "sha256": digest.hexdigest(),
        "domains": len(section),
        "relations": relations,
    }


def freeze_result(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8") as stream,
    ):
        serialized = {
            name: {"___".join(pair): domain for pair, domain in result[name].items()}
            for name in SECTIONS
        }
        json.dump(serialized, stream, separators=(",", ":"), allow_nan=False)
    destination.chmod(0o444)


def worker(args: argparse.Namespace) -> None:
    # Import only in the worker, after the parent chooses the reference/current source.
    from tclkg import qcn_generator2 as qgen

    artifact_root = args.artifacts.resolve()
    provenance = json.loads((artifact_root / "reference/provenance.json").read_text())
    source_input = ROOT / provenance["inputs"][args.worker]["path"]
    if sha256(source_input) != provenance["inputs"][args.worker]["sha256"]:
        raise ValueError(f"Input changed: {source_input}")
    qgen._compose_allen_scores_cached.cache_clear()
    kg = args.worker
    captured = io.StringIO()
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(
        prefix="www-perf-", dir=artifact_root
    ) as temporary:
        qgen.results_root = Path(temporary)
        with contextlib.redirect_stdout(captured):
            started = time.perf_counter()
            cpu_started = time.process_time()
            entities, properties = qgen.read_KG(kg)
            timings["load_wall_seconds"] = time.perf_counter() - started
            phase = time.perf_counter()
            network = qgen.build_entity_network(
                properties,
                qgen.ALLEN_RELATIONS,
                qgen.PROPERTY_SUPPORT_THRESHOLD,
                entities,
            )
            kept = sorted({p for pair in network for p in pair})
            timings["build_wall_seconds"] = time.perf_counter() - phase
            result = qgen.learner2(network, entities, kept, MINIMUMS[kg], 0.0)
            if result is None:
                raise RuntimeError(f"Learner failed for {kg}")
            phase = time.perf_counter()
            qgen.save_qcn_to_file(result, kg)
            timings["export_wall_seconds"] = time.perf_counter() - phase
            phase = time.perf_counter()
            if args.phase == "optimized":
                qgen.print_score_analysis(kg, learning_result=result)
            else:
                qgen.print_score_analysis(kg)
            timings["analysis_wall_seconds"] = time.perf_counter() - phase
            timings["pipeline_wall_seconds"] = time.perf_counter() - started
            timings["pipeline_cpu_seconds"] = time.process_time() - cpu_started
        timings.update(result["timing"])
        output = (
            Path(temporary) / kg / "conservative" / f"{qgen.build_output_stem(kg)}.json"
        )
        fingerprints = {name: section_fingerprint(result[name]) for name in SECTIONS}
        payload = {
            "kg": kg,
            "occurrence_threshold": 0.0,
            "min_total": MINIMUMS[kg],
            "timings": timings,
            "fingerprints": fingerprints,
            "output_bytes": output.stat().st_size,
            "source_module": qgen.__file__,
        }
        if args.freeze:
            folder = "reference" if args.phase == "baseline" else "optimized"
            snapshot = artifact_root / folder / "results" / f"{kg}.json.gz"
            freeze_result(result, snapshot)
            payload["snapshot_sha256"] = sha256(snapshot)
        Path(args.worker_result).write_text(json.dumps(payload, indent=2) + "\n")


def sweep_worker(args: argparse.Namespace) -> None:
    """Measure three real configurations, including disk cache publication/reads."""
    from tclkg import qcn_generator2 as qgen

    artifacts = args.artifacts.resolve()
    kg = args.worker
    provenance = json.loads((artifacts / "reference/provenance.json").read_text())
    source_input = ROOT / provenance["inputs"][kg]["path"]
    if sha256(source_input) != provenance["inputs"][kg]["sha256"]:
        raise ValueError(f"Input changed: {source_input}")
    configurations = ((0.0, 0), (0.5, MINIMUMS[kg]), (0.9, MAXIMUMS[kg]))
    fingerprints = {}
    cache_hits = 0
    oracle_calls = 0
    qgen._compose_allen_scores_cached.cache_clear()
    with (
        tempfile.TemporaryDirectory(prefix="www-sweep-", dir=artifacts) as temporary,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        started = time.perf_counter()
        cpu_started = time.process_time()
        for occurrence, minimum in configurations:
            cache_path = None
            prepared = None
            if args.phase == "optimized":
                cache_path = qgen.get_oracle_cache_path(
                    kg,
                    source_input.parent.parent,
                    Path(temporary) / "cache",
                    qgen.PROPERTY_SUPPORT_THRESHOLD,
                )
                prepared = qgen.read_oracle_cache(cache_path)
            if prepared is None:
                entities, properties = qgen.read_KG(kg)
                network = qgen.build_entity_network(
                    properties,
                    qgen.ALLEN_RELATIONS,
                    qgen.PROPERTY_SUPPORT_THRESHOLD,
                    entities,
                )
                properties = sorted({p for pair in network for p in pair})
            else:
                cache_hits += 1
                entities, network = {}, {}
                properties = prepared["properties"]
            extra = (
                {"prepared_oracle": prepared, "oracle_cache_path": cache_path}
                if args.phase == "optimized"
                else {}
            )
            result = qgen.learner2(
                network, entities, properties, minimum, occurrence, **extra
            )
            if result is None:
                raise RuntimeError(f"Learner failed: {kg}, {minimum}, {occurrence}")
            oracle_calls += result["oracle_stats"]["total_calls"]
            qgen.results_root = Path(temporary) / f"occ_{occurrence}_min_{minimum}"
            qgen.save_qcn_to_file(result, kg)
            if args.phase == "optimized":
                qgen.print_score_analysis(kg, learning_result=result)
            else:
                qgen.print_score_analysis(kg)
            # Fingerprinting is performed equally for both versions, outside
            # the elapsed interval accumulated for each configuration below.
            phase_end = time.perf_counter()
            cpu_end = time.process_time()
            hashes = {name: section_fingerprint(result[name]) for name in SECTIONS}
            fingerprint_wall = time.perf_counter() - phase_end
            fingerprint_cpu = time.process_time() - cpu_end
            started += fingerprint_wall
            cpu_started += fingerprint_cpu
            fingerprints[f"occ_{occurrence}_min_{minimum}"] = hashes
            del result
        timings = {
            "pipeline_wall_seconds": time.perf_counter() - started,
            "pipeline_cpu_seconds": time.process_time() - cpu_started,
        }
    Path(args.worker_result).write_text(
        json.dumps(
            {
                "kg": kg,
                "fingerprints": fingerprints,
                "timings": timings,
                "cache_hits": cache_hits,
                "oracle_calls": oracle_calls,
            },
            indent=2,
        )
        + "\n"
    )


def verify_provenance(artifacts: Path) -> None:
    provenance = json.loads((artifacts / "reference/provenance.json").read_text())
    for entry in provenance["inputs"].values():
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Locked input changed: {entry['path']}")
    for group in ("source_sha256", "original_results_sha256"):
        for name, expected in provenance[group].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Locked file changed: {name}")
    if sha256(ROOT / "uv.lock") != provenance["uv_lock_sha256"]:
        raise ValueError("The dependency lock changed")


def run(args: argparse.Namespace) -> None:
    artifacts = args.artifacts.resolve()
    verify_provenance(artifacts)
    suffix = "_sweep" if args.sweep else ""
    destination = artifacts / f"{args.phase}{suffix}.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace recorded measurements: {destination}"
        )
    reference = None
    if args.phase == "optimized":
        reference = json.loads((artifacts / f"baseline{suffix}.json").read_text())
        if not args.sweep:
            for kg in args.kg:
                expected = reference["runs"][kg][0]["snapshot_sha256"]
                if (
                    sha256(artifacts / "reference/results" / f"{kg}.json.gz")
                    != expected
                ):
                    raise ValueError(f"Reference snapshot changed for {kg}")
    environment = os.environ.copy()
    environment["TCLKG_DATA_DIR"] = str(artifacts / "data")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    source = (
        artifacts / "reference/source" if args.phase == "baseline" else ROOT / "src"
    )
    environment["PYTHONPATH"] = str(source)
    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scope": "load + build + learner2 + JSON/statistics export + score analysis; excludes process startup",
        "repeats": args.repeats,
        "runs": {},
        "summary": {},
    }
    for kg in args.kg:
        runs = []
        for iteration in range(args.repeats):
            result_path = artifacts / f".{args.phase}-{kg}-{iteration}.json"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                args.phase,
                "--artifacts",
                str(artifacts),
                "--worker",
                kg,
                "--worker-result",
                str(result_path),
            ]
            if args.sweep:
                command.append("--sweep")
            elif iteration == 0:
                command.append("--freeze")
            subprocess.run(command, env=environment, cwd=ROOT, check=True)
            measured = json.loads(result_path.read_text())
            result_path.unlink()
            expected_fingerprints = (
                reference["runs"][kg][0]["fingerprints"]
                if reference
                else runs[0]["fingerprints"]
                if runs
                else measured["fingerprints"]
            )
            if measured["fingerprints"] != expected_fingerprints:
                raise ValueError(
                    f"Exact output mismatch for {kg}, repetition {iteration + 1}"
                )
            runs.append(measured)
            elapsed = measured["timings"]["pipeline_wall_seconds"]
            print(
                f"{args.phase} {kg} {iteration + 1}/{args.repeats}: {elapsed:.3f}s, exact outputs verified",
                flush=True,
            )
            # Keep completed repetitions even if a later worker fails.
            report["runs"][kg] = runs
            destination.with_suffix(".partial.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        summary = {
            metric: statistics.median(run["timings"][metric] for run in runs)
            for metric in runs[0]["timings"]
        }
        if reference:
            previous = reference["summary"][kg]["pipeline_wall_seconds"]
            current = summary["pipeline_wall_seconds"]
            summary.update(
                speedup=previous / current,
                saved_seconds=previous - current,
                saved_percent=100 * (1 - current / previous),
            )
        report["summary"][kg] = summary
    destination.write_text(json.dumps(report, indent=2) + "\n")
    destination.with_suffix(".partial.json").unlink(missing_ok=True)
    if args.phase == "baseline":
        destination.chmod(0o444)
    print(destination)


def verify(artifacts: Path) -> None:
    """Re-read both immutable snapshots, checking every field against the lock."""
    verify_provenance(artifacts)
    baseline = json.loads((artifacts / "baseline.json").read_text())
    optimized = json.loads((artifacts / "optimized.json").read_text())
    manifest = json.loads((ROOT / "ground_truth/performance_manifest.json").read_text())
    for kg, runs in baseline["runs"].items():
        expected = runs[0]["fingerprints"]
        if expected != manifest["cases"][kg]["sections"]:
            raise ValueError(f"Reference differs from the portable manifest: {kg}")
        for folder, report in (("reference", baseline), ("optimized", optimized)):
            snapshot = artifacts / folder / "results" / f"{kg}.json.gz"
            if sha256(snapshot) != report["runs"][kg][0]["snapshot_sha256"]:
                raise ValueError(f"Snapshot modified: {snapshot}")
            with gzip.open(snapshot, "rt", encoding="utf-8") as stream:
                result = json.load(stream)
            actual = {name: section_fingerprint(result[name]) for name in SECTIONS}
            if actual != expected:
                raise ValueError(f"Exact result mismatch: {snapshot}")
            del result
        print(
            f"{kg}: all three networks, constraints, scores and metadata identical",
            flush=True,
        )
    for path in (artifacts / "baseline_sweep.json", artifacts / "optimized_sweep.json"):
        if not path.exists():
            return
    before = json.loads((artifacts / "baseline_sweep.json").read_text())
    after = json.loads((artifacts / "optimized_sweep.json").read_text())
    for kg, runs in before["runs"].items():
        if runs[0]["fingerprints"] != manifest["sweep_cases"][kg]:
            raise ValueError(
                f"Sweep reference differs from the portable manifest: {kg}"
            )
        for run in after["runs"][kg]:
            if run["fingerprints"] != runs[0]["fingerprints"]:
                raise ValueError(f"Sweep output mismatch: {kg}")
        print(f"{kg}: all three sweep configurations identical", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "optimized", "verify"))
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / "performance_artifacts"
    )
    parser.add_argument(
        "--kg", nargs="+", default=list(MINIMUMS), choices=list(MINIMUMS)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Measure three threshold configurations and cache reuse.",
    )
    parser.add_argument("--worker", choices=list(MINIMUMS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    parser.add_argument("--freeze", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.phase == "verify":
        verify(args.artifacts.resolve())
    elif args.worker:
        if args.sweep:
            sweep_worker(args)
        else:
            worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
