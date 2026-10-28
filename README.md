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
results/      aggregate model results and internal phenotyping analyses
scripts/      complete reviewed implementation
docs/         governance and reproducibility documentation
```

## Workflow

1. `01_data_preparation.ipynb`: cohort, labels, aligned features, split, and validation.
2. `02_model_training.ipynb`: LightGBM, causal Transformer, and exploratory FusionMLP.
3. `03_internal_evaluation.ipynb`: frozen internal evaluation, calibration, alert burden, and bootstrap.
4. `04_external_validation.ipynb`: frozen validation in NWICU and eICU.

The notebooks provide a concise scientific walkthrough. Detailed executable code is retained under `scripts/pipeline_steps/`.

## Pre-onset clinical phenotyping

In a complementary analysis to the eight-hour **early prediction** task, we investigated clinical heterogeneity during the **two hours preceding the first corrected onset of circulatory failure** in MIMIC-IV. This analysis is retrospective characterization of patients who developed the endpoint, **not** an externally validated early-warning model.

A prespecified panel of eight measurements—heart rate, mean arterial pressure, respiratory rate, SpO₂, lactate, platelet count, prothrombin time (PT), and activated partial thromboplastin time (PTT)—was summarized using the last directly observed value in that window. Laboratory features were selectively log-transformed; winsorization limits and robust scaling were fitted on the derivation subset. K-Means (k=2) was fitted on the derivation subset and evaluated using a frozen assignment rule in internal validation.

The primary time-eligible cohort comprised **13,262 patients**; requiring all eight observed measurements retained **4,606 (34.7%)**, with **3,268** in derivation and **1,338** in internal validation. The two clusters showed moderately separated, stable clinical profiles and different observed outcomes. Their association with prognosis should not be interpreted as causal or as evidence of biologically distinct mechanisms.

Additional audits found substantial **complete-case selection**, including overrepresentation of cardiovascular ICU patients, and **no consistent incremental discrimination** from cluster membership beyond its eight constituent variables in the evaluated prognostic models. **External validation of the phenotyping model has not been performed.** These limitations are part of the public results, not excluded from reporting.

- Implementation and reproducibility notes: [`scripts/phenotyping/`](scripts/phenotyping/)
- Aggregate phenotyping tables and audits: [`results/phenotyping/`](results/phenotyping/)

## Main aggregate results

On the MIMIC-IV internal test set, LightGBM achieved event/stay-level AUROC 0.9675 and AUPRC 0.9177.

In external event/stay evaluation, LightGBM achieved AUROC/AUPRC of 0.9416/0.6184 in NWICU and 0.9286/0.6212 in eICU. Full aggregate tables are available in `results/RESULTS.md` and `results/event_metrics.csv`.

## Data and model availability

Raw and derived patient-level data are not included. Access to MIMIC-IV and eICU requires credentialing through PhysioNet; NWICU access is governed separately. Model checkpoints and patient-, stay-, and hospital-level predictions are not distributed publicly.
