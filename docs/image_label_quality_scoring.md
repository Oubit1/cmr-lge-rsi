# Local image/label quality scoring

The scorer evaluates local image/label pair features and controlled synthetic
perturbations. A score or threshold is a configurable research policy, not a
clinical diagnosis or a validated guarantee for a new institution.

Configure local physician-reviewed pairs using
`configs/quality_scoring_datasets.example.yaml`, then rebuild the calibration,
patient-separated holdout and frozen benchmark in your private deployment.
No original patient pairs, fitted profile, clinical evaluation outcomes or
private benchmark hashes are included in this source release.

The framework separately records candidate validation, no-change acceptance,
actual label writes, backups, source/dependency hashes and committed receipts.
