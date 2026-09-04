## Exploratory phenotyping

Aggregate phenotyping outputs are available under `results/phenotyping/`.

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
