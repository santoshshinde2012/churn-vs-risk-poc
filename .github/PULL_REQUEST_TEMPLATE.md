## What this changes

<!-- One or two sentences. What is different after this PR? -->

## Why

<!-- What problem does it solve? If it is a disagreement with a documented finding,
     link the claim ID (e.g. CST-01) and bring the numbers. -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `pytest` passes
- [ ] `python run_all.py` completes
- [ ] `python scripts/verify_claims.py` passes

### If a number moved

- [ ] Updated the `expect` block in `docs/claims.yaml`
- [ ] Updated the prose that the claim's `docs:` field points at
- [ ] Re-rendered with `python scripts/verify_claims.py --render` and committed `docs/claims.md`
- [ ] Explained below *why* it moved

<!-- Which claims moved, and why: -->
