"""Verify the claims ledger against a real run, and render the traceability matrix.

    python scripts/verify_claims.py            # verify
    python scripts/verify_claims.py --render    # verify, then rewrite docs/claims.md

Why this exists
---------------
Documentation that quotes numbers rots, and it rots silently: the code keeps
running, the prose just stops being true. Nobody notices, because nothing fails.

docs/claims.yaml states every claim the article makes, a verdict, and a pointer
to the exact metric that supports or refutes it. This script resolves every
pointer against outputs/metrics/*.json. If a seed moves, a threshold changes or
a generator is retuned, the build fails and names the claim that broke.

It also checks the parts of the ledger that are not numbers: that every script
named in `tested_by` exists, and that every `docs:` anchor resolves to a real
heading in a real file. A traceability matrix that points at a deleted section
is worse than none.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"
LEDGER = ROOT / "docs" / "claims.yaml"
ARTICLE = ROOT / "docs" / "article" / "churn-vs-risk-medium.md"
RENDERED = ROOT / "docs" / "claims.md"

# claims.md is read both on GitHub and inside the MkDocs site. Links to files
# outside docs/ have to be absolute, because the site has no ../ to walk up to.
REPO = "https://github.com/santoshshinde2012/churn-vs-risk-poc/blob/main/"

VERDICTS = {
    "confirmed":    ("Confirmed",    "🟢", "The run supports the claim as stated."),
    "qualified":    ("Qualified",    "🟡", "Supported, with a caveat that changes what you do."),
    "contradicted": ("Contradicted", "🔴", "The run does not support the claim as stated."),
    "demonstrated": ("Demonstrated", "🔵", "An engineering practice shown working."),
}


# --------------------------------------------------------------- resolving --
class Unresolved(Exception):
    """A pointer that does not lead anywhere. Always a ledger bug."""


def resolve(pointer: str, cache: dict):
    """Resolve `file.json#dotted.path` against outputs/metrics.

    Numeric segments index into lists. Keys containing dots are supported by
    falling back to a longest-match lookup, which is what makes pointers like
    `risk.pr_auc.naive: retrain on what was allowed` work.
    """
    if "#" not in pointer:
        raise Unresolved(f"pointer '{pointer}' has no '#'")
    filename, path = pointer.split("#", 1)

    if filename not in cache:
        source = METRICS / filename
        if not source.exists():
            raise Unresolved(f"no metrics file {filename}; run `python run_all.py` first")
        cache[filename] = json.loads(source.read_text())

    node, remaining = cache[filename], path
    while remaining:
        if isinstance(node, list):
            head, _, remaining = remaining.partition(".")
            try:
                node = node[int(head)]
            except (ValueError, IndexError) as error:
                raise Unresolved(f"{pointer}: bad list index '{head}'") from error
            continue
        if not isinstance(node, dict):
            raise Unresolved(f"{pointer}: '{remaining}' has nowhere to go")
        # Longest matching key first, so keys containing dots resolve.
        for key in sorted(node, key=len, reverse=True):
            if remaining == key:
                node, remaining = node[key], ""
                break
            if remaining.startswith(key + "."):
                node, remaining = node[key], remaining[len(key) + 1:]
                break
        else:
            raise Unresolved(f"{pointer}: no key matching '{remaining}'")
    return node


def check(actual, expect: dict) -> tuple[bool, str]:
    """Compare a resolved value against an expectation. Returns (ok, detail)."""
    if expect.get("is_null"):
        return actual is None, "expected null" if actual is not None else "null"
    if actual is None:
        return False, "resolved to null"
    if isinstance(actual, bool):
        actual = int(actual)
    try:
        actual = float(actual)
    except (TypeError, ValueError):
        return False, f"not numeric: {actual!r}"

    if "value" in expect:
        tolerance = float(expect.get("tol", 0.0))
        ok = abs(actual - float(expect["value"])) <= tolerance
        return ok, f"{actual:g} vs {expect['value']:g} +/- {tolerance:g}"
    if "min" in expect:
        return actual >= float(expect["min"]), f"{actual:g} >= {expect['min']:g}"
    if "max" in expect:
        return actual <= float(expect["max"]), f"{actual:g} <= {expect['max']:g}"
    return False, "expectation has no value/min/max/is_null"


# ---------------------------------------------------------------- checking --
def anchor_of(heading: str) -> str:
    """GitHub's heading-to-anchor slug."""
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s", "-", slug)


def doc_anchors(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {anchor_of(line.lstrip("#").strip())
            for line in path.read_text().splitlines() if line.startswith("#")}


def check_article(failures: list) -> None:
    """The article embeds result figures from a run. Keep those honest too.

    The result figures in docs/article/ come from outputs/plots via docs/images.
    Each carries a caption beginning "Measured" and a link to the script that
    produced it, which is the only thing telling a reader that a figure is a
    measurement rather than one of the hand-drawn concept diagrams beside it.

    A renamed figure, a dropped caption or a retired claim ID would leave the
    published article showing a broken image or citing something that no longer
    exists - and nobody would notice, because prose does not fail a build.
    """
    if not ARTICLE.exists():
        return
    text = ARTICLE.read_text()

    for reference in re.findall(r"\]\((\.\./images/[^)]+)\)", text):
        if not (ARTICLE.parent / reference).resolve().exists():
            failures.append(f"article: embeds a missing figure '{reference}'")

    known = {claim["id"] for claim in yaml.safe_load(LEDGER.read_text())["claims"]}
    for cited in re.findall(r"claims\.md#([a-z]{3}-\d{2})", text):
        if cited.upper() not in known:
            failures.append(f"article: cites claim '{cited.upper()}', which is not in the ledger")

    # Every embedded result figure must be followed by a *Measured ...* caption.
    embedded = re.findall(r"!\[[^\]]*\]\(\.\./images/[^)]+\)\n(.*)", text)
    for caption in embedded:
        if not caption.lstrip("*").startswith("Measured"):
            failures.append(f"article: a result figure is captioned {caption[:48]!r}; "
                            f"result captions must start with 'Measured' so a reader can "
                            f"tell a measurement from a drawing")


def verify(ledger: dict) -> tuple[list, list]:
    """Returns (results, failures). One result row per claim."""
    cache: dict = {}
    anchor_cache: dict = {}
    results, failures = [], []

    for claim in ledger["claims"]:
        cid = claim["id"]

        if claim["verdict"] not in VERDICTS:
            failures.append(f"{cid}: unknown verdict '{claim['verdict']}'")
        if claim["verdict"] in ("qualified", "contradicted") and not claim.get("caveat"):
            failures.append(f"{cid}: a '{claim['verdict']}' claim must explain itself "
                            f"in `caveat`")

        for script in str(claim.get("tested_by", "")).split(","):
            script = script.strip()
            if script and not (ROOT / script).exists():
                failures.append(f"{cid}: tested_by names a missing file '{script}'")

        target = claim.get("docs", "")
        if target:
            doc_path, _, anchor = target.partition("#")
            full = ROOT / doc_path
            if not full.exists():
                failures.append(f"{cid}: docs points at a missing file '{doc_path}'")
            else:
                if doc_path not in anchor_cache:
                    anchor_cache[doc_path] = doc_anchors(full)
                if anchor and anchor not in anchor_cache[doc_path]:
                    failures.append(f"{cid}: docs anchor '#{anchor}' not found in {doc_path}")

        checked = []
        for item in claim.get("evidence", []):
            try:
                actual = resolve(item["metric"], cache)
            except Unresolved as error:
                failures.append(f"{cid}: {error}")
                checked.append((item, False, "unresolved"))
                continue
            ok, detail = check(actual, item["expect"])
            if not ok:
                failures.append(f"{cid}: {item['metric']} -> {detail}")
            checked.append((item, ok, detail))

        results.append({"claim": claim, "checked": checked})

    check_article(failures)
    return results, failures


# --------------------------------------------------------------- rendering --
def render(ledger: dict, results: list) -> str:
    counts = dict.fromkeys(VERDICTS, 0)
    for row in results:
        counts[row["claim"]["verdict"]] += 1
    total = len(results)
    evidence_total = sum(len(row["checked"]) for row in results)

    out: list[str] = []
    add = out.append

    add("<!-- GENERATED FILE. Edit docs/claims.yaml, then run:")
    add("     python scripts/verify_claims.py --render -->")
    add("")
    add("# The claims ledger")
    add("")
    add(f'Every substantive claim in **"{ledger["meta"]["article"]}"**, with a verdict and a')
    add("pointer to the metric that settles it.")
    add("")
    add("This page is generated from [`claims.yaml`](claims.yaml) by")
    add(f"[`scripts/verify_claims.py`]({REPO}scripts/verify_claims.py), which resolves every")
    add("pointer against `outputs/metrics/*.json` from a real run. CI fails if any of them")
    add("stops holding, so a claim here cannot quietly become untrue.")
    add("")
    add(f"**{total} claims, {evidence_total} verified measurements.**")
    add("")
    add("| Verdict | | Count | Meaning |")
    add("| :--- | :-- | ---: | :--- |")
    for key, (label, icon, meaning) in VERDICTS.items():
        add(f"| {icon} | **{label}** | {counts[key]} | {meaning} |")
    add("")
    add("The three claims that are not confirmed are the reason this ledger exists. A")
    add("companion repo that agrees with its article about everything has not been run.")
    add("")

    add("## At a glance")
    add("")
    add("| ID | Theme | Claim | Verdict | Tested by |")
    add("| :--- | :--- | :--- | :-- | :--- |")
    for row in results:
        claim = row["claim"]
        label, icon, _ = VERDICTS[claim["verdict"]]
        summary = " ".join(claim["article_claim"].split())
        if len(summary) > 92:
            summary = summary[:89].rstrip() + "..."
        scripts = ", ".join(f"`{s.strip()}`" for s in str(claim["tested_by"]).split(","))
        add(f"| [{claim['id']}](#{claim['id'].lower()}) | {claim['theme']} | {summary} "
            f"| {icon} {label} | {scripts} |")
    add("")

    add("## The evidence")
    add("")
    for row in results:
        claim = row["claim"]
        label, icon, _ = VERDICTS[claim["verdict"]]
        add(f"### {claim['id']}")
        add("")
        add(f"**{icon} {label}** &nbsp;·&nbsp; {claim['theme']}"
            + (f" &nbsp;·&nbsp; {claim['figure']}" if claim.get("figure") else ""))
        add("")
        add("> " + " ".join(claim["article_claim"].split()))
        add("")
        if claim.get("caveat"):
            add(f"**Why {label.lower()}.** " + " ".join(claim["caveat"].split()))
            add("")
        add("| Measurement | Value | |")
        add("| :--- | :--- | :-- |")
        for item, ok, detail in row["checked"]:
            reads = " ".join(item.get("reads", item["metric"]).split())
            add(f"| {reads} | `{detail}` | {'✅' if ok else '❌'} |")
        add("")
        scripts = ", ".join(f"`{s.strip()}`" for s in str(claim["tested_by"]).split(","))
        where = f", {claim['where']}" if claim.get("where") else ""
        line = f"Runs in {scripts}{where}."
        if claim.get("docs"):
            # claims.md lives in docs/, so a repo-relative `docs/x.md#y` pointer
            # becomes a sibling link `x.md#y`.
            target = claim["docs"]
            href = target[5:] if target.startswith("docs/") else f"../{target}"
            line += f" Written up in [{target.split('#')[0]}]({href})."
        add(line)
        add("")

    return "\n".join(out) + "\n"


# -------------------------------------------------------------------- main --
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", action="store_true",
                        help="rewrite docs/claims.md from the ledger")
    args = parser.parse_args()

    if not METRICS.exists():
        print(f"no metrics at {METRICS.relative_to(ROOT)}; run `python run_all.py` first",
              file=sys.stderr)
        return 2

    ledger = yaml.safe_load(LEDGER.read_text())
    results, failures = verify(ledger)

    total_evidence = sum(len(row["checked"]) for row in results)
    print(f"claims ledger: {len(results)} claims, {total_evidence} measurements")
    by_verdict: dict[str, int] = {}
    for row in results:
        by_verdict[row["claim"]["verdict"]] = by_verdict.get(row["claim"]["verdict"], 0) + 1
    print("  " + "   ".join(f"{VERDICTS[k][0].lower()}: {v}" for k, v in sorted(by_verdict.items())))

    if failures:
        print(f"\n{len(failures)} problem(s):\n", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("\nEither the change was intended - update docs/claims.yaml and the prose "
              "it points at - or it was not.", file=sys.stderr)
        return 1

    print("every claim still holds against the current run.")

    if args.render:
        RENDERED.write_text(render(ledger, results))
        print(f"rendered {RENDERED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
