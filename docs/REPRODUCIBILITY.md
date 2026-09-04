# Reproducibility

1. Create the Python environment from `requirements.txt`.
2. Copy `.env.example` to `.env` for local paths. Configure database credentials through a secure local secret mechanism; never store them in the repository.
3. Run notebooks 01–04 in order, following the detailed source scripts they enumerate.
4. Confirm the feature count and SHA-256 signature in `config/feature_names.json`.
5. Select models and operational thresholds using MIMIC-IV validation only.
6. Keep the internal test and both external cohorts frozen for final evaluation.

The external analyses prohibit retraining, scaler refitting, threshold tuning, and performance-guided hospital exclusion.
