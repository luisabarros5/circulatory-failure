# Early prediction of circulatory failure in ICU patients

This repository contains the reviewed code for developing and externally validating models that estimate, every five minutes, the risk of a first circulatory-failure onset within the following eight hours.


## Models

- Event/stay-weighted LightGBM
- Causal Transformer with an eight-hour lookback
- Exploratory Transformer–tabular FusionMLP

All models use the same ordered set of 204 inputs: 68 clinical values, 68 observation indicators, and 68 elapsed-time-since-measurement variables. Vasoactive treatment variables are excluded from model inputs and may only contribute to endpoint construction.

## Endpoint

Circulatory failure is defined as arterial lactate ≥2 mmol/L together with MAP ≤65 mmHg or active vasoactive/inotropic support, persisting in at least two of three consecutive five-minute timesteps. The target is the first onset in `(t, t+8 h]`, using inputs available by `t−5 min`.

## Repository structure

```text
config/       endpoint and feature definitions
data/         access instructions only
figures/      aggregate figures only
models/       artifact-governance instructions only
notebooks/    four reader-facing workflow notebooks
results/      aggregate model and exploratory phenotyping results
scripts/      complete reviewed implementation
docs/         governance and reproducibility documentation
```

## Workflow

1. `01_data_preparation.ipynb`: cohort, labels, aligned features, split, and validation.
2. `02_model_training.ipynb`: LightGBM, causal Transformer, and exploratory FusionMLP.
3. `03_internal_evaluation.ipynb`: frozen internal evaluation, calibration, alert burden, and bootstrap.
4. `04_external_validation.ipynb`: frozen validation in NWICU and eICU.

The notebooks provide a concise scientific walkthrough. Detailed executable code is retained under `scripts/pipeline_steps/`.

## Exploratory phenotyping

The latest exploratory phenotyping analysis represents each positive ICU stay by a single clinical snapshot at the first circulatory-failure onset. A curated panel of 20 variables is median-imputed, selectively log-transformed, winsorized, standardized, reduced with PCA, and clustered with K-means. The best global geometric solution was PCA80 + k=2; a secondary within-cluster analysis yielded three clinically interpretable exploratory phenotypes.

Implementation is available under `scripts/phenotyping/onset_snapshot/`. Only aggregate, non-patient-level outputs are published under `results/phenotyping/onset_snapshot/`.

## Main aggregate results

On the MIMIC-IV internal test set, LightGBM achieved event/stay-level AUROC 0.9675 and AUPRC 0.9177.

In external event/stay evaluation, LightGBM achieved AUROC/AUPRC of 0.9416/0.6184 in NWICU and 0.9286/0.6212 in eICU. Full aggregate tables are available in `results/RESULTS.md` and `results/event_metrics.csv`.

## Data and model availability

Raw and derived patient-level data are not included. Access to MIMIC-IV and eICU requires credentialing through PhysioNet; NWICU access is governed separately. Model checkpoints and patient-, stay-, and hospital-level predictions are not distributed publicly.
