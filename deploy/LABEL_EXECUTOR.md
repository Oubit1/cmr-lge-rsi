# Production label executor deployment

> 2026-10-23 owner decision: the board now applies an explicitly confirmed
> staged edit directly on its loopback-only listener. This page documents the
> previous separate-executor deployment option; it is not required for the
> current board apply path. The trade-off is that board and label writer share
> the `operator` OS identity. Source-hash validation, backup, provenance and
> board audit remain mandatory. Autopilot is not authorized to trigger production
> apply, but same-UID code is no longer technically isolated from the writer.

The previous deployment defaulted to fail-closed label application. The board only staged
candidates and submitted apply requests to `/run/cmr-lge-label-executor/apply.sock`.
That architecture remains available as a reviewed option but is not the active route.

Host-admin installation is intentionally separate from application deployment:

1. Create system user `cmr-label-executor` and group `cmr-label-submitters`.
2. Add only interactive operators and the board service identity to
   `cmr-label-submitters`; do not add the Autopilot/Codex maintenance identity.
3. Copy a reviewed commit to root-owned `/opt/cmr-lge-label-executor` and make it
   non-writable by application users.
4. Change ownership of authoritative `cmr_checked` and `lge_checked` trees to
   `cmr-label-executor`; remove write ACLs for the Codex/Autopilot identity.
5. Grant the executor write ACLs only on `runtime/manual_annotations`,
   `labels/check`, and `label_iteration/manual_edits` as required by the staged
   apply transaction.
6. Install `deploy/cmr-lge-label-executor.service` as a system service, create
   `/var/log/cmr-lge-label-executor`, then enable and start it.
7. Run the boundary test below as the Codex/Autopilot identity. The first write
   must fail and the socket request must succeed only for an approved submitter.

```bash
test ! -w /srv/rsi/dataset/labels/cmr_checked/cmr_Myo_mask
test ! -w /srv/rsi/dataset/labels/lge_checked
test -S /run/cmr-lge-label-executor/apply.sock
```

Do not point the system service at the user-writable checkout. Updating the
executor is a deployment action: copy a reviewed commit into `/opt`, run tests,
and restart the system service under administrator control.
