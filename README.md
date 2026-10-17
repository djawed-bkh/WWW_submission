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

## Ground Truth Before Optimization

This repository includes semantic ground-truth manifests for Learner 2 outputs.
Use them before and after performance changes to ensure the optimized code keeps
the same results.

Fast checks:

```bash
pytest tests/test_ground_truth_unit.py tests/test_ground_truth_mini_pipeline.py
```

Mandatory full-result verification:

```bash
python -m tclkg.ground_truth verify
```

Optional slow verification, including `Q82955`:

```bash
python -m tclkg.ground_truth verify --include-slow
```

See `ground_truth/README.md` for manifest format, regeneration commands, and
the distinction between semantic correctness and runtime baselines.
