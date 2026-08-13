# Security policy

## Scope

This is a self-contained demonstration project. It has no network calls, no authentication,
no persistence beyond local files, and it ships no personal data — the churn and transaction
datasets are generated at run time from fixed seeds.

That said, two things are worth knowing.

**The local HTTP endpoint.** `03_live_scoring_demo.py` starts an unauthenticated
`ThreadingHTTPServer` bound to `127.0.0.1:8091` for a few seconds. It is a demonstration of
serving mechanics, not a server: no authentication, no input validation beyond a column
list, no rate limiting. Do not expose it, and do not use it as a template for anything that
faces a network.

**The Databricks notebook.** `part_b_databricks.py` reads a workspace host and a token from
variables you supply. It never writes them anywhere. Do not commit them — `.gitignore`
covers `.env`, `.databrickscfg` and `*.pem`, but the notebook itself is a plain file and
will happily keep whatever you paste into it.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/santoshshinde2012/churn-vs-risk-poc/security/advisories/new)
rather than a public issue. A response should take a few days.

If you find that a dependency pinned in `requirements.txt` has a known vulnerability, a
pull request bumping it is welcome — please run `python run_all.py` first and note in the PR
whether any claim in `docs/claims.yaml` moved.
