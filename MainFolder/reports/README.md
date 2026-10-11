# Historical report templates

The original `metrics.json` contains `null` metrics, `ablation_table.csv` has
unmeasured result cells, and `error_analysis.md` is a blank analysis template.
They are preserved as repository history, not evidence of trained-model quality,
latency, ablations, or previous achievements.

The rebuilt trainer writes actual measured validation metrics to the ignored
checkpoint directory in `training_metadata.json` and `training_history.json`.
Metadata records the base-model revision, dataset SHA-256 fingerprints, seed,
source counts for each split, threshold and per-label precision/recall/F1/support.
Split groups use identical text after case/whitespace normalization across both
sources, so duplicate comments cannot cross training and validation. Conflicting
labels for duplicate text within a source are rejected. This does not separate
authors/threads, and a representative independent test set is still needed.

The best checkpoint is selected by the mean of both heads' validation macro-F1.
The selected validation figures therefore are **not independent test accuracy**.
Zero-denominator precision/recall/F1 use zero and support is reported. No real
training/evaluation dataset was supplied for this reconstruction, and no actual
project accuracy is claimed. Tiny artificial test fixtures only test the code.
