# Pre-failure phenotyping — K-Means (MIMIC-IV)

This directory contains Lucas's pre-onset analysis, separate from the project's **early prediction** models. The nine Python scripts are located in [`preonset_kmeans/`](preonset_kmeans/).

## Reproducible workflow

1. `01_build_corrected_preonset_cohort.py` — identifies the first corrected onset, constructs **strictly pre-onset** two-hour windows (24 five-minute time steps), selects one eligible ICU stay per patient, and writes intermediate files **locally**.
2. `02_fit_frozen_k2.py` — requires at least one directly observed measurement of each of the eight variables (heart rate, MAP, respiratory rate, SpO₂, lactate, platelets, PT, and PTT); extracts the most recent observed value, applies transformations, winsorization, and scaling fitted on the derivation subset, and runs K-Means (k=2). Cluster assignments and the frozen model artifact are saved locally only.
3. `03_primary_outcome_inference.py` — estimates risks and associations with outcomes.
4. `04_clinical_enrichment.py` — derives comorbidities and vasoactive exposures using official SQL concepts and locally available authorized data sources.
5. `05_confirmatory_inference.py` — performs covariate-balanced association analyses and evaluates incremental prognostic value.
6. `06_make_publication_tables_figures.py` — creates aggregate publication tables and figures.

Helper modules: `common.py`, `inference_utils.py`, and `phenotype_inference_utils.py`.

The commands below assume you are running them **from the repository root**. Execution requires authorized access to MIMIC-IV data and a Python environment with the project's dependencies. No individual-level clinical data are distributed in this package.

```bash
export SOURCE_ROOT=/path/to/authorized/inputs
export WORK_DIR=/path/to/private/intermediates
export MIMIC_ROOT=/path/to/mimiciv/3.1
export MIMIC_CODE=/path/to/mimic-code
python scripts/phenotyping/preonset_kmeans/01_build_corrected_preonset_cohort.py \
  --clean-dir "$SOURCE_ROOT/icu_clean_v2" \
  --labels-dir "$SOURCE_ROOT/step05_labels_v2" \
  --stay-subject-map "$SOURCE_ROOT/stay_subject_map.parquet" \
  --mimic-root "$MIMIC_ROOT" --out-dir "$WORK_DIR/cohort" --window-hours 2
python scripts/phenotyping/preonset_kmeans/02_fit_frozen_k2.py \
  --cohort-dir "$WORK_DIR/cohort" --out-dir "$WORK_DIR/model"
```

Run each script with `--help` to inspect the remaining steps; scripts 04 and 05 require additional clinical inputs and outputs from preceding steps. **Rerunning script 02 is not equivalent to externally validating the original frozen model.**

Public aggregate results: [`results/phenotyping/preonset_kmeans/`](../../../results/phenotyping/preonset_kmeans/). Do not publish `*.parquet` files, arrays, patient identifiers, patient hashes, internal logs, or binary models without a specific disclosure review.
