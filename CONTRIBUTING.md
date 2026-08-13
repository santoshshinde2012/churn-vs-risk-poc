# Contributing

Thanks for looking. This repo is a companion to an article, so it has an unusual
constraint: **the prose and the numbers have to stay in sync.** Most of the guidance below
exists to make that automatic rather than a matter of care.

## Getting set up

```bash
./setup.sh                      # CPython 3.10-3.13
.venv/bin/python run_all.py     # tests, five pipelines, ledger, dashboard  (~1 min)
```

## Before you open a pull request

```bash
.venv/bin/python -m ruff check .        # lint
.venv/bin/python -m pytest              # 49 tests
.venv/bin/python run_all.py             # full run, ledger, dashboard
git diff --exit-code docs/claims.md     # ledger re-rendered and committed?
```

CI runs exactly these on Python 3.10 through 3.13.

## The rule that catches people out

**If your change moves a number, the docs must move with it, in the same commit.**

Every figure quoted in the docs is registered in [`docs/claims.yaml`](docs/claims.yaml)
with a pointer to the metric that produces it. `scripts/verify_claims.py` resolves all 67 of
them against a real run, so a retuned generator or a changed seed fails the build and names
the claim that broke:

```
FBK-01: feedback_metrics.json#risk.fraud_censored_by_blocking -> 180 vs 242 +/- 60
```

When that happens, one of two things is true, and you have to decide which:

1. **The change was intended.** Update the `expect` block in `claims.yaml`, update the prose
   the claim's `docs:` field points at, re-render with `--render`, and say so in the PR.
2. **It was not.** You have found a real regression. Good.

Never widen a tolerance to make a failure go away without understanding it first. The
tolerances are already generous enough to absorb platform floating-point differences.

## What good changes look like

**Comments explain why, not what.** The code says what it does. A comment earns its place by
explaining a decision that is not obvious from reading it — why the blocklist excludes
devices with legitimate history, why the holdout is hashed rather than drawn, why the
threshold is derived rather than typed in. Several comments in this repo document a mistake
that was made and measured; those are the most valuable ones.

**Results are reported as measured.** Three of the article's claims are contradicted or
qualified by the code, and they are written up as such in
[`docs/findings.md`](docs/findings.md). If your change makes a result inconvenient, report
it. A companion repo that agrees with its article about everything has not been run.

**Invariants get tests, not comments.** Point-in-time correctness, holdout stability and
policy validation are all pinned by tests, because breaking any of them makes the metrics
look *better* and nothing else would notice.

## Adding a claim

1. Add an entry to `docs/claims.yaml` with a verdict, `tested_by`, a `docs:` anchor that
   exists, and at least one evidence pointer.
2. `qualified` and `contradicted` claims must carry a `caveat` explaining themselves — the
   verifier enforces this.
3. Run `python scripts/verify_claims.py --render` and commit `docs/claims.md`.

## Style

- Ruff config lives in [`pyproject.toml`](pyproject.toml); line length 96.
- The numbered pipeline scripts compute at module level on purpose — they are meant to be
  read top to bottom like a document. Reusable machinery belongs in `churnrisk/`.
- Figures use `churnrisk/viz.py` so every plot in the repo shares one style.
- Diagrams under `docs/images/*.svg` are hand-authored so they diff as text. Render them
  with `rsvg-convert -w 1040 file.svg -o /tmp/check.png` and actually look at the result
  before committing — three real bugs in this repo were found that way.

## Reporting problems

- Bugs and reproducibility issues: open an issue with the output of `python run_all.py` and
  your `pip freeze`.
- Security: see [SECURITY.md](SECURITY.md).
- Disagreements with a finding: those are the most welcome kind of issue. Bring the numbers.

## Licence

By contributing you agree that your contributions are licensed under
[Apache-2.0](LICENSE), the same as the rest of the project.
