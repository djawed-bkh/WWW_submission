# XXXX-1 Submission

This repository contains the Learner 2 pipeline for temporal constraint learning on knowledge graphs.

## Run Learner 2

### 1. Prerequisites

- Linux/macOS shell (or equivalent terminal)
- Python 3.12+
- A virtual environment

### 2. Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Verify the CLI is available

```bash
python -m tclkg.run_experiments --help
python -m tclkg.qcn_generator2 --help
```

If these commands print usage text, your environment is ready.

## Execution Modes

### A) Run configured Learner 2 experiment batch

```bash
python -m tclkg.run_experiments
```

Optional: run only one KG from the configured list:

```bash
python -m tclkg.run_experiments --kg Q6256
```

Optional: limit parallel jobs:

```bash
python -m tclkg.run_experiments --max-parallel 2
```

The batch configuration is defined in:

- `src/tclkg/run_experiments.py` (`EXPERIMENTS_GEN2`)

### B) Run one Learner 2 experiment directly

```bash
python -m tclkg.qcn_generator2 Q6256 3600
```

Arguments:

- `kg`: one of `Q6256`, `Q215380`, `Q82955`
- `timeout`: timeout in seconds

## Input Data

By default, data is read from:

- `data/<KG>/data.quintuplet`

If missing, it falls back to:

- `data/<KG>/train_cst_knowledge.quintuplet`

You can override the data root with:

```bash
export TCLKG_DATA_DIR=/absolute/path/to/data
```

## Output Files

### Direct Learner 2 run (`tclkg.qcn_generator2`)

- JSON QCN output: `Results/<KG>/qcn2_<KG>.json`
- Statistics report: `Results/<KG>/qcn2_<KG>_stats.txt`

### Batch run (`tclkg.run_experiments`)

- Per-run logs: `logs_experiments/`

## Export a temporal network to SHACL

Install the locked environment, including the optional validation engine:

```bash
uv sync --locked --extra shacl
```

Export the final network without validating any data:

```bash
uv run tclkg-shacl-export \
  Results/Q6256/conservative/qcn2_Q6256_reduced_composition.json \
  --class-iri http://www.wikidata.org/entity/Q6256 \
  --output shapes.ttl
```

The same script is available as `uv run python -m tclkg.shacl_export`.
`--class-iri` is required because the TCL JSON does not store the target class.
The default stage is `after_propagation`; `--stage after_oracle` and
`--stage initial` select the other saved networks. A JSON object containing a
single network directly is also accepted. Pair keys use the learner's
`A___B` serialization, with the current `relations`/`support` payload format.

Add `--data` to export and run validation in the same invocation:

```bash
uv run tclkg-shacl-export \
  Results/Q6256/conservative/qcn2_Q6256_reduced_composition.json \
  --class-iri http://www.wikidata.org/entity/Q6256 \
  --output shapes.ttl \
  --data data/Q6256/data.quintuplet \
  --report validation-report.ttl
```

Without `--output`, the shapes are saved beside the JSON as `NETWORK.shacl.ttl`.
Without `--report`, validation writes `OUTPUT_STEM.report.ttl` beside the shapes.
Exit codes are **0** for successful export or conforming data, **1** for data
violations, and **2** for an invalid invocation, unsupported input, or I/O error.
A nonconforming graph still produces both the shapes and the validation report.
Export alone requires RDFLib, installed with the base project; validation
additionally requires the `shacl` extra.

Validation accepts either:

- A `.quintuplet` file using the same five tab-separated fields as TCL. It is
  treated as a class-specific input split: every subject is assigned to
  `--class-iri`. The converter uses the existing loader's day granularity,
  ordering of reversed endpoints, completion of a single missing endpoint, and
  deduplication. A subject that is a local identifier receives a deterministic
  `urn:tcl:entity:` IRI. If both endpoints are `None`, they remain absent and
  structural SHACL validation reports the incomplete assertion.
- An RDF graph already encoding temporal assertion records. Each assertion has
  type `https://example.org/tcl#TemporalAssertion`, the three RDF reification
  predicates `rdf:subject`, `rdf:predicate`, `rdf:object`, and the endpoint
  predicates `https://example.org/tcl#start` and `https://example.org/tcl#end`.
  Endpoints are UTC `xsd:dateTime` literals with `Z` or `+00:00`. Class membership
  must already be present in RDF; the script preserves it. Ordinary RDF triples
  alone do not encode the intervals needed for these temporal checks.

The data format is inferred from the suffix; `--data-format quintuplet` or an
RDFLib format such as `turtle`, `xml`, or `nt` overrides detection. Wikidata
`prop/direct/P…` predicates are normalized to `prop/P…` in both networks and
input assertions. The quintuplet dateTime conversion supports years 0001–9999;
out-of-range dates are reported with the source line number.

Allowed Allen relations form a disjunction for **every** assertion pair sharing
a subject and the two properties. Empty TCL domains represent unresolved
information and add no temporal constraint; universal domains likewise add no
restriction. Inferred relations are retained even when their observed support
is zero. Scores are not applied as validation thresholds. Equivalent converse
arcs are exported once; independently specified nonequivalent arcs are both
preserved. The structural shape remains active even when no temporal pair is
restricted, and admits point intervals to match the loader. The generic
construction is described in `latex/tcl_shacl_shapes.tex`.

Focused verification, also run by the SHACL CI workflow and pre-commit hooks:

```bash
uv run ruff check --fix src/tclkg/shacl_export.py tests/test_shacl_export.py
uv run ruff format src/tclkg/shacl_export.py tests/test_shacl_export.py
uv run pyright src/tclkg/shacl_export.py tests/test_shacl_export.py
uv run pytest tests/test_shacl_export.py
```

## Result equivalence before optimization

The current performance verification compares complete, locked outputs from the
original source with outputs from the optimized learner. It checks constraints,
supports, inferred scores, domain statuses and propagation stages exactly:

```bash
uv run python scripts/benchmark_performance.py verify
```

The older `tclkg.ground_truth` manifest helpers remain available for legacy
manifests. The full snapshots and portable regression cases used for this
optimization are described below.

## Performance snapshots and measurements

The performance work uses an independent, exact result lock. It includes every
domain field, relation, support count, `inferred_score`, and `inference_stage`.
Dictionary order and the JSON representation of tuples/lists are normalized;
floating-point values are compared exactly. Runtime fields are measured separately.

Local artifacts are kept in the ignored `performance_artifacts/` directory:

- `reference/provenance.json`: SHA-256 checksums of the original source, inputs,
  dependency lock, and six pre-existing training result files.
- `reference/source/`: the source captured before optimization.
- `reference/results/`: read-only compressed reference networks for all three KGs.
- `optimized/results/`: compressed networks produced by the optimized code.
- `baseline.json` and `optimized.json`: individual repetitions and median times.
- `baseline_sweep.json` and `optimized_sweep.json`: a three-configuration campaign.

The three full inputs were recovered from `data/Data_chunk*`. Reference artifacts
are created before changing the learner and are never regenerated during comparison.
The benchmark refuses to overwrite an existing report or snapshot. The commands used to create and verify the measurements are:

```bash
uv run python scripts/benchmark_performance.py baseline
uv run python scripts/benchmark_performance.py optimized
uv run python scripts/benchmark_performance.py baseline --sweep --kg Q215380
uv run python scripts/benchmark_performance.py optimized --sweep --kg Q215380
uv run python scripts/benchmark_performance.py verify
```

Each measurement uses three fresh sequential Python workers by default. Timings
include loading, network construction, learning, JSON/statistics export, and score
analysis; interpreter startup and verification hashing are excluded. The single
run measurements use conservative propagation, reduced composition, occurrence
threshold zero, and minimum totals 10 / 30 / 128 for Q6256 / Q215380 / Q82955.
The campaign uses `(occurrence, minimum)` values `(0, 0)`, `(0.5, 30)`, and
`(0.9, 150)` for Q215380. All three network states are checked on every repetition.

Portable regression cases captured from the original code are committed in
`ground_truth/performance_cases.json`. They cover 192 combinations across
conservative, two-stage, iterative and Strass propagation, both composition modes,
and evidence thresholds. Run their checks with:

```bash
uv run pytest tests/test_performance.py tests/test_ground_truth_unit.py tests/test_two_stage_propagation.py
```

`ground_truth/performance_manifest.json` also preserves the full-dataset input
checksums, original source checksums, and exact expected network fingerprints in
a small versioned file. Verification checks the local snapshots against it.

## Reusing the oracle in parameter sweeps

The launcher prepares one requested configuration per KG before scheduling the
remaining configurations. Later configurations reuse a private, atomic oracle
cache in the batch directory. Its key covers the input bytes, preparation source,
property selection, Python/NumPy versions, and `uv.lock`. Propagation thresholds
and modes remain independent. JSON output retains the same schema and uses compact
encoding; score analysis reads the already computed networks in memory.

```bash
TCLKG_DATA_DIR=performance_artifacts/data uv run python -m tclkg.launcher --kg Q215380 --max-parallel 2
TCLKG_DATA_DIR=performance_artifacts/data uv run python -m tclkg.launcher --kg Q215380 --no-oracle-cache
```

`--max-parallel` is enforced. A timeout covers the child process that loads,
learns, and saves the configuration, with exit code 124 on timeout. Direct learner
CLI calls use the same process boundary without transferring loaded graphs.
