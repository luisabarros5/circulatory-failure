# First-onset snapshot phenotyping results

These files contain aggregate results from the current exploratory first-onset snapshot analysis.

## Cohort and representation

- 15,739 positive ICU stays;
- one clinical snapshot per stay at first circulatory-failure onset;
- 20 clinical variables;
- no onset-time variable used for clustering;
- no MiniRocket representation used in this analysis.

## Main result

PCA retaining 80% of variance produced 12 components. PCA80 + k=2 achieved the best global Silhouette score (0.2249):

- C0, multiorgan dysfunction: n=4,417 (28.1%);
- C1, preserved/vasopressor-supported: n=11,322 (71.9%).

Exploratory subclustering of C0 yielded:

- hepatic/metabolic pattern: n=1,416 (9.0% of the full cohort);
- renal/hypotensive pattern: n=3,001 (19.1% of the full cohort).

The C0 subclustering Silhouette was 0.1034. These patterns should therefore be interpreted as exploratory and clinically overlapping, not as validated discrete classes.

## Published files

The directory contains only aggregate metrics, cluster counts, clinical summaries, feature-level differences, onset observation coverage, and seed-stability statistics. No row represents an individual patient or ICU stay.

## Deliberately excluded

- raw and derived patient-level Parquet files;
- `subject_id`, `hadm_id`, `stay_id`, and timestamps;
- scaled patient matrices;
- PCA/preprocessing/scaling binary objects;
- per-stay cluster assignments;
- per-stay stability assignments.
