# Onset-snapshot public-release audit

Audit date: 2026-10-01

## Included

- ten pipeline and validation scripts;
- aggregate feature-availability summaries;
- aggregate PCA/K-means metrics and cluster sizes;
- aggregate clinical summaries and feature differences;
- aggregate subclustering metrics;
- aggregate stability metrics across random seeds.

## Excluded

- raw and transformed patient-level snapshots;
- metadata containing encounter identifiers or onset times;
- all per-stay cluster assignments;
- scaled patient matrices;
- preprocessing, PCA, scaling, centroid, and local-scaling binary objects;
- per-stay stability assignments;
- macOS metadata.

## Automated checks

The release checker rejects patient-level binary formats, identifier-bearing filenames, identifier columns in CSV/TSV files, stored notebook outputs, local personal paths, database credentials, and files larger than 5 MiB. The packaged release passed these checks.
