# First-onset snapshot phenotyping

This directory implements the current exploratory phenotyping pipeline for ICU stays that meet the circulatory-failure endpoint.

## Design

- one clinical vector per positive ICU stay;
- reference time: first circulatory-failure onset;
- 20 clinically interpretable variables;
- median imputation;
- selective `log1p` transformation;
- 0.5th-99.5th percentile winsorization;
- standardization;
- direct K-means and PCA + K-means sensitivity analyses;
- global seed-stability and within-macrophenotype analyses.

Identifiers are used only to align credentialed local tables. They are never used as clustering features and no patient-level artifact is included in the public repository.

## Scripts

1. `01_build_onset_snapshot.py` builds one row per positive stay.
2. `02_prepare_log_curated_onset_snapshot.py` prepares the curated 20-variable matrix.
3. `03_run_kmeans_no_pca.py` evaluates direct K-means for k=2..6.
4. `04_characterize_k2_k3_phenotypes.py` produces aggregate clinical summaries.
5. `05_run_pca_kmeans_snapshot.py` evaluates PCA80/PCA90/PCA95 with k=2..6.
6. `06_characterize_pca80_k2.py` characterizes the selected global solution.
7. `07_subcluster_macrophenotypes.py` evaluates within-macrophenotype structure.
8. `08_audit_imputation_availability.py` audits measurement availability at onset.
9. `09_validate_global_stability.py` evaluates stability across random seeds.
10. `10_characterize_c0_subclusters_original.py` summarizes C0 subclusters on the original clinical scale.

## Data governance

The scripts expect credentialed local inputs. Do not commit raw snapshots, metadata tables, scaled matrices, fitted preprocessing/PCA objects, cluster assignments, or any table containing patient or encounter identifiers. Only the aggregate outputs under `results/phenotyping/onset_snapshot/` are approved for public release.
