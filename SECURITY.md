# Privacy and deployment security

The dashboard exposes local data and control actions. Its direct label-apply
listener is loopback-only. Do not expose it directly to the public Internet.
For remote access use an authenticated private tunnel or reverse proxy with
TLS, authentication and the configured board-control token.

Keep credentials in local environment variables or an ignored secrets file.
Keep patient images, masks, DICOM headers, manifests, review text, prompts,
provider replies, audit logs, benchmark assets and database files out of Git.
Pseudonymizing an identifier alone does not anonymize medical data. Rebuild
sharing permissions and privacy review for each dataset/provider deployment.

This release contains synthetic fixtures only. It does not authorize sending
medical images or identifying metadata to an external model. Generated local
artifacts can contain sensitive information and must be reviewed before sharing.

Report a suspected security issue privately through the repository's security
advisory channel when enabled. Do not open a public issue containing patient
information, real credentials, private paths or raw logs.
