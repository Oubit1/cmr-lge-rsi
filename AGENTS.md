# Contributor and agent boundaries

Maintain framework code, interfaces, scheduling, services, UI and verification.
Do not substitute framework maintenance for clinical adjudication. Domain tools
operate on disposable copies; only the explicitly configured executor applies
independently validated candidates with source hashes, backups and audit.

Never commit local datasets, patient identifiers, clinical manifests, provider
responses, secrets, database files or runtime snapshots. Use synthetic fixtures.
Keep role handoffs, independent review, frozen holdouts and operator pause/budget
semantics intact. Tests must not call paid providers or use private datasets.

Run `python -m unittest discover -s tests -q` and
`python scripts/audit_public_tree.py` before publishing.
