# Public release checklist

- [ ] All notebooks have zero stored outputs and null execution counts.
- [ ] No raw or derived patient-level data are tracked.
- [ ] No patient, stay, admission, or hospital identifiers are tracked.
- [ ] No credentials, `.env`, database URIs, or personal filesystem paths are tracked.
- [ ] No model checkpoints, embeddings, or predictions are tracked.
- [ ] Only aggregate metrics and figures are included.
- [ ] Authors approved the repository license and citation metadata.
- [ ] `python scripts/check_public_release.py` passes before every push.
