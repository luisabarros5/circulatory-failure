# Public results — Pre-onset phenotyping

The analysis currently documented here is Lucas's **pre-failure K-Means (k=2)** clustering, based on eight variables observed during the two hours immediately preceding the first corrected onset. Aggregated results and audits are available in [`preonset_kmeans/`](preonset_kmeans/).

This **clinical characterization and prognostic association** study should not be confused with the project's models predicting circulatory failure within the next eight hours. Earlier exploratory snapshot/MiniRocket analyses have been removed from this public presentation by project decision; they are not confirmatory results of this model.

Complete-case selection retains 4,606 of 13,262 temporally eligible patients (34.7%), with disproportionate representation of cardiovascular intensive care units, limiting generalizability. In frozen internal validation, adding cluster membership to the eight original variables did not demonstrate a consistent incremental improvement in discrimination. **No external validation of the clustering has been performed.**

This release contains only aggregate tables without patient identifiers. Clinical-enrichment files with very small cell/event counts have deliberately been excluded from public distribution.

For the public ICU-stratified mortality audit, rows with fewer than 11 patients in either comparison group have been omitted to avoid publishing very small strata; the retained table is not exhaustive across all unit types.
