# Public release verification

- Source-only export with a fresh Git history; no original runtime snapshots.
- Public-tree audit checks file types, tracked paths, private-network literals,
  common credential formats and name/age identifier patterns.
- Gitleaks v8.30.1 directory scan: zero findings.
- Private comparison against the original casebook identifiers: no original
  identifier matches in the public source. The comparison list is not distributed.
- Isolated existing-environment suite: 1901 tests passed, 1 skipped.
- Additional publication privacy and demo-overwrite tests: 3 passed.
- Generated synthetic demo: dashboard, editor, case listing and metadata routes
  all returned HTTP 200, and the state endpoint returned its structured snapshot.

- Clean Python 3.10 virtual environment installed from requirements.txt: 1904 tests
  passed, 1 skipped. The missing lxml test dependency was added before release.

GitHub CI repeats the checks on Python 3.10 and 3.12. Passing
software tests is not evidence of clinical accuracy or continuous RSI improvement.
