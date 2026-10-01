## Exploratory phenotyping

Aggregate phenotyping outputs are available under `results/phenotyping/`.

### First-onset clinical snapshot — current exploratory analysis

The current preferred exploratory analysis includes 15,739 positive ICU stays and uses one 20-variable clinical snapshot at the first circulatory-failure onset. The best global geometric solution was PCA80 + k=2 (12 components, 80.55% cumulative variance, Silhouette 0.2249), yielding a multiorgan-dysfunction group (n=4,417) and a preserved/vasopressor-supported group (n=11,322).

Exploratory subclustering of the multiorgan-dysfunction group produced hepatic/metabolic (n=1,416) and renal/hypotensive (n=3,001) patterns. Because within-group separation was modest (Silhouette 0.1034), these are reported as exploratory subphenotypes rather than discrete validated classes.

Aggregate metrics, cluster sizes, clinical summaries, feature differences, observation coverage, and seed-stability results are provided under `results/phenotyping/onset_snapshot/`. Patient-level snapshots, timestamps, identifiers, scaled matrices, PCA objects, cluster assignments, and other fitted binary artifacts are not distributed.

### Temporal MiniRocket

This analysis uses a temporal representation followed by MiniRocket embeddings, PCA, and K-means. Aggregate k-selection, cluster-level clinical summaries, feature summaries, rates, and run metrics are included in `results/phenotyping/temporal_minirocket/`.

This run predates the current endpoint revision and is retained for methodological and clinical comparison.

### Approach A — fixed 8-hour pre-onset

The full positive cohort included 15,739 ICU stays. The initial k=3 solution achieved a Silhouette score of 0.455.

In a sensitivity analysis restricted to the 9,449 stays with a complete 8-hour pre-onset window, the k=3 Silhouette decreased to 0.242, while k=2 achieved 0.402. This indicates that pre-onset history availability materially affects the original clustering structure.

### Approach B — normalized pre-onset

The normalized pre-onset approach included the same 15,739 positive ICU stays and produced a k=3 Silhouette score of 0.192.

Cluster sizes were 6,187, 4,340, and 5,212 stays. Observation-density analyses showed that earlier-onset patients had fewer directly observed clinical variables before failure, particularly laboratory measurements, indicating that measurement availability and imputation remain important methodological considerations.

These phenotyping results are exploratory and are included to support methodological and clinical review rather than to establish finalized clinical phenotypes.
