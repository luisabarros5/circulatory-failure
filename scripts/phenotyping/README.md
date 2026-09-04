# Exploratory phenotyping

This directory contains the exploratory clustering pipeline for ICU stays meeting the current circulatory-failure definition (`cf_first_onset_primary`).

The phenotyping analysis is still under methodological review and should not be interpreted as a finalized phenotype solution.

## Approach A — fixed 8-h pre-onset window

For each positive ICU stay:

1. identify the first circulatory-failure onset;
2. extract up to 8 hours immediately before onset on a 5-minute grid;
3. transform the temporal representation with MiniRocket;
4. reduce the embedding with PCA (50 components);
5. cluster with K-means.

The initial k=3 solution used all 15,739 positive stays.

A sensitivity analysis was also performed using only the 9,449 stays with a complete 8-hour history before onset.

## Approach B — normalized pre-onset trajectory

This approach uses the complete available trajectory from ICU history start to first failure onset.

Each trajectory is mapped to a relative 0–100% time axis and resampled to 96 positions using a curated panel of 20 clinical variables.

The resulting representation is processed with MiniRocket, PCA (50 components), and K-means.

## Structure

- `core/`: MiniRocket and clustering utilities
- `cohort/`: positive-cohort construction and audits
- `approach_a_fixed_8h/`: fixed-window approach
- `approach_b_normalized_preonset/`: normalized-time approach
- `validation/`: sensitivity and robustness analyses

## Data availability

Patient-level data, identifiers, temporal tensors, embeddings and cluster assignments are not distributed.

Only code and aggregate results are included in the public repository.
