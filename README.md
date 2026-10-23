# Temporal Constraint Learning 

This artifact accompanies an anonymous research submission. It provides an
implementation for learning temporal constraints between knowledge graph
properties using Allen's interval relations.

The pipeline builds a qualitative constraint network (QCN), derives observed
relations in an oracle phase, and propagates temporal constraints. Saved outputs
include the initial network, the network after the oracle, and the network after
propagation.

## 1. Environment setup

Run the commands below from the root of the extracted artifact or repository.
Requirements: Python 3.12 or later and a Linux/macOS shell. Full datasets and
parameter sweeps can require substantial memory and disk space; start with one
configuration and one worker.

For a locked environment, use `uv`:

```bash
uv sync --locked
source .venv/bin/activate
```

Alternatively, install with standard Python tooling (this resolves dependencies
from `pyproject.toml` rather than using `uv.lock`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If obtaining the artifact through Git, install Git LFS and run `git lfs pull`
inside the repository to retrieve the large result files. A source ZIP downloaded
from a Git hosting service may contain LFS pointers instead of those results.
The learner can regenerate results from the input data.

## 2. Prepare the input data

The input archive is distributed as five parts, `data/Data_chunkaa` through
`data/Data_chunkae`. Reassemble and extract it into a separate directory so that
existing files remain intact:

```bash
mkdir -p reviewer_data
cat data/Data_chunk{aa,ab,ac,ad,ae} > reviewer_data/inputs.zip
python -m zipfile -e reviewer_data/inputs.zip reviewer_data
export TCLKG_DATA_DIR="$PWD/reviewer_data"
```

Set `TCLKG_DATA_DIR` in each new shell before running experiments. It must point
to the directory containing the dataset subdirectories:

```text
reviewer_data/
├── Q6256/data.quintuplet
├── Q215380/data.quintuplet
└── Q82955/data.quintuplet
```

| Dataset identifier | Entity class |
| --- | --- |
| `Q6256` | Country |
| `Q215380` | Musical group |
| `Q82955` | Politician |

Each input line has five tab-separated fields:

```text
subject    property    object    start    end
```

The separators must be literal tabs. The loader uses day-level temporal
intervals and accepts `None` for an unknown endpoint. For each dataset, it reads
`data.quintuplet`, falling back to `train_cst_knowledge.quintuplet` if the former
is absent. Other archive files contain the training, validation, and test splits.
To learn exclusively from a particular split, create a separate data root with
that split copied to `<dataset>/data.quintuplet` and point `TCLKG_DATA_DIR` to it.
Without the environment variable, the default data root is `data/`.

## 3. Run one configuration

Check that the command-line interfaces are available:

```bash
python -m tclkg.qcn_generator2 --help
python -m tclkg.run_experiments --help
```

Run conservative propagation on the Country dataset:

```bash
python -m tclkg.qcn_generator2 Q6256 0 10 3600 \
  --propagation-mode conservative --reduced-composition
```

The four positional arguments are:

| Argument | Example | Meaning |
| --- | --- | --- |
| Dataset | `Q6256` | Subdirectory under the input data root |
| Relation occurrence threshold | `0` | Minimum observed relation proportion used for inference, between 0 and 1 |
| Domain reliability threshold (`tau`) | `10` | Minimum property-pair occurrence count used for propagation |
| Timeout | `3600` | Maximum execution time in seconds |

Conservative propagation and reduced composition are enabled by default.
Reduced composition skips compositions yielding all 13 Allen relations. Use
`--no-reduced-composition` to disable this option. Other implemented propagation
modes are `two_stage`, `iterative` (also named `fix_point`), and `strass`.

The example writes:

```text
Results/Q6256/conservative/qcn2_Q6256_reduced_composition.json
Results/Q6256/conservative/qcn2_Q6256_reduced_composition_stats.txt
```

The JSON contains `initial`, `after_oracle`, `after_propagation`, and statistics.
Property-pair keys use the separator `___`. The text report summarizes network
sizes, reductions, and timings. Direct runs with the same dataset, mode, and
composition option reuse these output paths; use the batch launcher below to
keep configurations in separate directories. A timed-out direct run exits with
code `124` and may not produce final output files.

## 4. Run parameter sweeps

Preview the configured experiments without running the learner:

```bash
python -m tclkg.run_experiments --kg Q6256 --dry-run
```

Run all default configurations sequentially:

```bash
python -m tclkg.run_experiments --max-parallel 1
```

Use `--kg Q6256` to restrict the run to one dataset. Increase `--max-parallel`
only when sufficient memory is available. The default grid, defined in
`src/tclkg/launcher.py`, uses conservative propagation, reduced composition,
a timeout of 86,400 seconds per configuration, and these thresholds:

| Dataset | Occurrence thresholds | Domain reliability thresholds (`tau`) |
| --- | --- | --- |
| `Q6256` | 0, 0.3, 0.5, 0.8, 0.9 | 0, 10, 59, 110, 160, 190 |
| `Q215380` | 0, 0.3, 0.5, 0.8, 0.9 | 0, 10, 30, 60, 100, 150 |
| `Q82955` | 0, 0.3, 0.5, 0.8, 0.9 | 0, 14, 42, 128, 389, 1185 |

This produces 30 configurations per dataset (90 in total). Oracle results are
cached within a batch by default; use `--no-oracle-cache` to recompute them.
Each batch writes networks, statistics, and `run.log` files under:

```text
parameter_sweeps/<batch>/<dataset>/<mode>/<configuration>/
```

Use `--output-root PATH` to change the output root.

For a small, explicit experiment, save the following as `reviewer_config.json`:

```json
{
  "databases": ["Q6256"],
  "configurations": [
    {
      "name": "occ0_tau10",
      "occurrence_threshold": 0.0,
      "min_total": 10,
      "timeout": 3600,
      "reduced_composition": true,
      "propagation_mode": "conservative"
    }
  ]
}
```

Then run:

```bash
python -m tclkg.run_experiments \
  --config reviewer_config.json --max-parallel 1 --output-root reviewer_results
```

## 5. Measure oracle and propagation CPU time separately

```bash
python -m tclkg.benchmark_cpu \
  --datasets Q6256 Q215380 Q82955 --output reviewer_cpu.json
```

For each dataset, this recomputes the oracle five times without a cache, then
measures five independent propagations for each default `tau` listed above.
The occurrence threshold defaults to zero and reduced composition is enabled.
The JSON records individual measurements and arithmetic means in CPU seconds
using `time.process_time()`.

Oracle timing excludes input loading, initial network construction, and copying;
it includes index preparation, oracle queries, and converse domains. Propagation
timing excludes copying the oracle network. These are phase timings rather than
end-to-end elapsed runtimes. This benchmark does not impose an execution timeout.

To measure a smaller selection:

```bash
python -m tclkg.benchmark_cpu \
  --datasets Q6256 --taus 0 10 --occurrence 0 --output reviewer_cpu_small.json
```

The benchmark refuses to overwrite an existing output file. Use a new filename
for each measurement campaign. CPU times depend on the machine and environment.

## 6. Optional SHACL export

Install the validation extra with `uv sync --locked --extra shacl` or
`python -m pip install -e '.[shacl]'`. Export a learned network:

```bash
python -m tclkg.shacl_export \
  Results/Q6256/conservative/qcn2_Q6256_reduced_composition.json \
  --class-iri http://www.wikidata.org/entity/Q6256 \
  --output reviewer_shapes.ttl
```

The default exported stage is `after_propagation`; use `--stage after_oracle`
for the oracle network. Add `--data "$TCLKG_DATA_DIR/Q6256/data.quintuplet"`
and `--report reviewer_validation.ttl` to validate temporal assertions.

## 7. Verification and repository layout

With the locked development environment installed, run the portable regression
checks and CPU benchmark tests:

```bash
uv run pytest tests/test_performance.py tests/test_ground_truth_unit.py \
  tests/test_two_stage_propagation.py tests/test_benchmark_cpu.py
```

These checks do not require a complete parameter sweep. Historical performance
scripts may require local reference artifacts that are not part of the review
package; they are not needed to run the approach described above.

| Path | Contents |
| --- | --- |
| `src/tclkg/qcn_generator2.py` | Oracle, temporal constraint learning, and propagation |
| `src/tclkg/launcher.py` | Parameter grids and batch execution |
| `src/tclkg/benchmark_cpu.py` | Separate CPU measurements for oracle and propagation |
| `src/tclkg/shacl_export.py` | SHACL export and optional validation |
| `data/Data_chunk*` | Split input archive for the three datasets |
| `Results/` | Saved networks, reports, and analysis artifacts |
| `WWWfactChecking/` | Fact-checking experiment artifacts |
| `scripts/` | Analysis and reporting utilities |
| `ground_truth/`, `tests/` | Reference cases and regression checks |
| `pyproject.toml`, `uv.lock` | Package metadata and dependency configuration |

All example paths are relative to the project root unless explicitly stated.
Some command-line messages are in French; the commands and output formats are
as documented above.
