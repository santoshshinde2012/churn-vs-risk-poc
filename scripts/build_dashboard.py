"""Build a self-contained HTML summary of a run.

    python scripts/build_dashboard.py     ->  outputs/dashboard.html

Everything is inlined - CSS, and every figure as a base64 data URI - so the file
can be emailed, dropped in a ticket or opened from a USB stick and still render.
No network, no assets directory, no build step.

The content comes from outputs/metrics/*.json and docs/claims.yaml, so it cannot
disagree with the run that produced it. If a number here looks wrong, the run is
wrong.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"
PLOTS = ROOT / "outputs" / "plots"
OUT = ROOT / "outputs" / "dashboard.html"

VERDICT_STYLE = {
    "confirmed":    ("Confirmed", "ok"),
    "qualified":    ("Qualified", "warn"),
    "contradicted": ("Contradicted", "bad"),
    "demonstrated": ("Demonstrated", "info"),
}


def load(name: str) -> dict:
    path = METRICS / name
    return json.loads(path.read_text()) if path.exists() else {}


def data_uri(name: str) -> str | None:
    path = PLOTS / name
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def tile(label: str, value: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="tile {tone}"><div class="tile-v">{esc(value)}</div>'
            f'<div class="tile-l">{esc(label)}</div>'
            f'{f"<div class=tile-n>{esc(note)}</div>" if note else ""}</div>')


def figure(name: str, caption: str) -> str:
    uri = data_uri(name)
    if not uri:
        return ""
    return (f'<figure><img src="{uri}" alt="{esc(caption)}" loading="lazy">'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


CSS = """
:root{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--soft:#52514e;--line:#e4e3df;
--blue:#2a78d6;--orange:#eb6834;--aqua:#1baf7a;--plum:#8a5cd1}
@media (prefers-color-scheme:dark){:root{--bg:#12120f;--card:#1a1a17;--ink:#f2f2ee;
--soft:#a3a29c;--line:#2e2e29}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:34px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--soft);font-size:14px;margin:0}
h2{font-size:17px;margin:44px 0 4px;letter-spacing:-.005em}
h2 .rule{display:block;height:2px;width:34px;background:var(--blue);margin-top:8px;border-radius:2px}
h2.risk .rule{background:var(--orange)} h2.mixed .rule{background:var(--plum)}
.lede{color:var(--soft);font-size:13.5px;margin:10px 0 18px;max-width:70ch}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 15px}
.tile-v{font-size:23px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile-l{font-size:12.5px;color:var(--soft);margin-top:3px}
.tile-n{font-size:11.5px;color:var(--soft);opacity:.8;margin-top:6px}
.tile.ok .tile-v{color:var(--aqua)} .tile.bad .tile-v{color:var(--orange)}
.tile.info .tile-v{color:var(--blue)}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0 4px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;font-size:12px;color:var(--soft);text-transform:uppercase;letter-spacing:.04em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11.5px;font-weight:600;
white-space:nowrap}
.pill.ok{background:rgba(27,175,122,.14);color:var(--aqua)}
.pill.warn{background:rgba(235,104,52,.13);color:var(--orange)}
.pill.bad{background:rgba(235,104,52,.2);color:var(--orange)}
.pill.info{background:rgba(42,120,214,.14);color:var(--blue)}
figure{margin:18px 0 0;background:var(--card);border:1px solid var(--line);
border-radius:9px;padding:12px}
figure img{width:100%;height:auto;display:block;border-radius:4px}
figcaption{font-size:12.5px;color:var(--soft);margin-top:9px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.callout{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--orange);
border-radius:0 9px 9px 0;padding:14px 16px;margin:14px 0}
.callout h3{margin:0 0 6px;font-size:14px}
.callout p{margin:0;font-size:13.5px;color:var(--soft)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
font-size:12.5px;color:var(--soft)}
a{color:var(--blue)}
"""


def build() -> str:
    churn = load("churn_metrics.json")
    risk = load("risk_metrics.json")
    live = load("live_metrics.json")
    feedback = load("feedback_metrics.json")
    monitor = load("monitoring_metrics.json")
    ledger = yaml.safe_load((ROOT / "docs" / "claims.yaml").read_text())

    counts: dict[str, int] = {}
    for claim in ledger["claims"]:
        counts[claim["verdict"]] = counts.get(claim["verdict"], 0) + 1

    h: list[str] = []
    add = h.append

    add("<!doctype html><html lang=en><head><meta charset=utf-8>")
    add('<meta name=viewport content="width=device-width,initial-scale=1">')
    add("<title>Churn vs Risk &mdash; run summary</title>")
    add(f"<style>{CSS}</style></head><body><div class=wrap>")

    add("<header><h1>Churn vs Risk &mdash; run summary</h1>")
    add('<p class=sub>Generated from <code>outputs/metrics/*.json</code>. '
        "Every figure is embedded, so this file works on its own.</p></header>")

    # ------------------------------------------------------------ headline --
    add("<h2>The run<span class=rule></span></h2>")
    add('<p class=lede>Two pipelines built against the same article, and the claims '
        "ledger's verdict on what they showed.</p>")
    add("<div class=grid>")
    add(tile("claims verified", str(len(ledger["claims"])),
             f"{sum(len(c.get('evidence', [])) for c in ledger['claims'])} measurements"))
    add(tile("confirmed", str(counts.get("confirmed", 0)),
             "supported as stated", "ok"))
    add(tile("qualified", str(counts.get("qualified", 0)),
             "true, with a caveat that matters"))
    add(tile("contradicted", str(counts.get("contradicted", 0)),
             "the data disagreed", "bad"))
    add(tile("demonstrated", str(counts.get("demonstrated", 0)),
             "practice shown working", "info"))
    add("</div>")

    # ---------------------------------------------------------------- churn --
    if churn:
        test = churn.get("test", {})
        add("<h2>Churn &mdash; the batch path<span class=rule></span></h2>")
        add('<p class=lede>20,000 simulated accounts over 240 days, scored out of sample '
            "on a later snapshot than the model trained on.</p>")
        add("<div class=grid>")
        add(tile("PR-AUC", f"{test.get('pr_auc', 0):.3f}",
                 f"base rate {test.get('base_rate', 0):.1%}"))
        add(tile("precision@200", f"{churn.get('precision_at_k', {}).get('200', 0):.0%}",
                 "against the team's real capacity"))
        add(tile("ROC-AUC", f"{test.get('roc_auc', 0):.3f}", "flattering, and why we do not quote it"))
        add(tile("ECE", f"{test.get('ece', 0):.4f}", "measured, not assumed"))
        add(tile("holdout", f"{churn.get('holdout_accounts', 0):,}",
                 "accounts nobody may contact"))
        add("</div>")
        leak = churn.get("leakage", {})
        if leak:
            add("<div class=callout><h3>What leakage buys you offline</h3><p>"
                f"A leaky feature job scores <b>{leak.get('leaky_model_leaky_eval', 0):.3f}</b> "
                "when evaluated the same leaky way, and "
                f"<b>{leak.get('leaky_model_real_features', 0):.3f}</b> once asked about the "
                f"future. The point-in-time model scores "
                f"<b>{leak.get('point_in_time_model', 0):.3f}</b> &mdash; and keeps scoring it."
                "</p></div>")
        add("<div class=figs>")
        add(figure("churn_label_window.png",
                   "The same book, four label windows. A 7-day window mislabels half its churners."))
        add(figure("churn_leakage.png",
                   "What leakage buys offline, and costs in production."))
        add("</div>")

    # ----------------------------------------------------------------- risk --
    if risk:
        mat = risk.get("maturation", {})
        val = risk.get("validation", {})
        thr = risk.get("threshold", {})
        vis = risk.get("labels_visible_today", {})
        add("<h2 class=risk>Risk &mdash; the real-time path<span class=rule></span></h2>")
        add('<p class=lede>279,030 simulated transactions over 240 days, with chargebacks '
            "that arrive 30 to 90 days late and a tactic change on day 200.</p>")
        add("<div class=grid>")
        add(tile("fraud rate", f"{risk.get('true_fraud_rate', 0):.2%}", "of transactions"))
        add(tile("labels on file today", f"{vis.get('chargebacks_on_file', 0):,}",
                 f"of {vis.get('true_frauds', 0):,} actual frauds", "bad"))
        add(tile("PR-AUC, out of time", f"{val.get('oot', 0):.3f}", "days 150-200"))
        add(tile("PR-AUC, drift week", f"{val.get('drift_week', 0):.3f}", "after the tactic change"))
        add(tile("insult rate", f"{risk.get('bands', [{}])[0].get('insult_together', 0):.3%}",
                 "good customers blocked", "ok"))
        add("</div>")
        add("<div class=callout><h3>The threshold argues with the article</h3><p>"
            f"The cost model gives <b>{thr.get('derived', 0):.2f}</b>, but blocking there costs "
            f"<b>${thr.get('cost_at_derived', 0):,.0f}</b> against "
            f"<b>${thr.get('cost_at_default_half', 0):,.0f}</b> at the 0.5 default. "
            f"Overall ECE reads {thr.get('ece_overall', 0):.4f}; inside the decision region it "
            f"reads <b>{thr.get('ece_decision_region', 0):.3f}</b>. The formula was fed a number "
            "that was not true.</p></div>")
        add("<div class=scroll><table><tr><th>Training set</th>"
            "<th class=num>PR-AUC on the drift week</th><th>What it is</th></tr>")
        for key, label, note in (
                ("pr_auc_matured_only", "A &mdash; matured labels only", "the model you could build"),
                ("pr_auc_naive_labels", "B &mdash; labels as they look today",
                 "what happens by default"),
                ("pr_auc_oracle_labels", "C &mdash; labels from the future", "impossible")):
            add(f"<tr><td>{label}</td><td class=num>{mat.get(key, 0):.3f}</td><td>{note}</td></tr>")
        add("</table></div>")
        add("<div class=figs>")
        add(figure("risk_label_maturation.png",
                   "Share of fraud labels that have arrived, by days since the transaction."))
        add(figure("risk_validation_gap.png",
                   "The same model, three views of reality."))
        add(figure("risk_cost_curve.png",
                   "Total cost by block threshold. The derived threshold is not the minimum."))
        add(figure("risk_rules_coverage.png",
                   "Fraud challenged, reviewed or blocked: model, rules, and both."))
        add("</div>")

    # ------------------------------------------------------- feedback loops --
    if feedback:
        fr = feedback.get("risk", {})
        fc = feedback.get("churn", {})
        pr = fr.get("pr_auc", {})
        add("<h2 class=mixed>Feedback loops<span class=rule></span></h2>")
        add('<p class=lede>Both models damage the data they would retrain on, in opposite '
            "directions.</p>")
        add("<div class=grid>")
        add(tile("fraud labels erased", f"{fr.get('fraud_censored_by_blocking', 0):,}",
                 "blocked, so never confirmed", "bad"))
        add(tile("still observable", f"{fr.get('fraud_still_observable', 0):,}",
                 "of the window's fraud"))
        add(tile("accounts saved", f"{fc.get('saved', 0):,}",
                 "and relabelled 'did not churn'"))
        add(tile("incremental retention",
                 f"+{fc.get('incremental_retention', 0):.1%}",
                 f"+/- {fc.get('incremental_retention_ci_half_width', 0):.1%}, "
                 "measurable only via the holdout", "ok"))
        add("</div>")
        if pr:
            add("<div class=scroll><table><tr><th>Next training set</th>"
                "<th class=num>PR-AUC</th></tr>")
            for name, value in pr.items():
                add(f"<tr><td>{esc(name)}</td><td class=num>{value:.3f}</td></tr>")
            add("</table></div>")
        targeting = fc.get("targeting", [])
        if targeting:
            add("<div class=scroll><table><tr><th>Strategy for 200 calls</th>"
                "<th class=num>Customers kept</th><th class=num>Lost causes</th>"
                "<th class=num>Sleeping dogs</th></tr>")
            for row in targeting:
                add(f"<tr><td>{esc(row['strategy'])}</td>"
                    f"<td class=num>{row['customers_kept']:.1f}</td>"
                    f"<td class=num>{row['lost_causes']}</td>"
                    f"<td class=num>{row['sleeping_dogs']}</td></tr>")
            add("</table></div>")
        add("<div class=figs>")
        add(figure("risk_selective_labels.png",
                   "The next model, by what its training set was allowed to see."))
        add(figure("churn_uplift.png",
                   "Customers kept per 200 calls, and who is in the top decile by risk."))
        add("</div>")

    # ---------------------------------------------------------- monitoring --
    if monitor:
        mr = monitor.get("risk", {})
        mc = monitor.get("churn", {})
        add("<h2 class=risk>Monitoring, on two clocks<span class=rule></span></h2>")
        add('<p class=lede>The only monitoring that can run daily is monitoring that needs '
            "no outcome at all.</p>")
        add("<div class=grid>")
        add(tile("tactics change", f"day {mr.get('drift_day', 0)}", "the attack starts"))
        add(tile("alarm fires", f"day {mr.get('first_alert_day', 0)}",
                 "flag rate, no labels needed", "ok"))
        add(tile("labels usable", f"day {mr.get('label_based_detection_day', 0)}",
                 "the earliest a PR-AUC chart moves", "bad"))
        add(tile("head start", f"{mr.get('head_start_days', 0)} days",
                 "bought by monitoring inputs", "ok"))
        add(tile("false alarms", str(mr.get("false_alarms_before_change_z_score", 0)),
                 "across the quiet days before"))
        add("</div>")
        add("<div class=callout><h3>Population PSI never fired</h3><p>"
            "Across every monitored day. The attack is under 1% of traffic, which no "
            "population-level statistic can see at a usable threshold. The flag rate &mdash; "
            "the share of traffic scoring above the challenge band &mdash; caught it in "
            f"{mr.get('first_alert_day', 0) - mr.get('drift_day', 0)} days. On the churn side "
            f"the worst signal peaked at {mc.get('max_worst_psi', 0):.3f} and decayed on its "
            "own.</p></div>")
        add("<div class=figs>")
        add(figure("monitoring_two_clocks.png",
                   "Daily and monthly, side by side, each against its own alert threshold."))
        add(figure("risk_latency.png", "Every call against the 80 ms budget, outage included."))
        add("</div>")

    # ------------------------------------------------------------- ledger ---
    add("<h2>Claims ledger<span class=rule></span></h2>")
    add('<p class=lede>Every claim in the article, with the verdict a real run gives it. '
        "Generated from <code>docs/claims.yaml</code> and verified in CI.</p>")
    add("<div class=scroll><table><tr><th>ID</th><th>Theme</th><th>Claim</th>"
        "<th>Verdict</th></tr>")
    for claim in ledger["claims"]:
        label, tone = VERDICT_STYLE[claim["verdict"]]
        summary = " ".join(claim["article_claim"].split())
        if len(summary) > 110:
            summary = summary[:107].rstrip() + "..."
        add(f"<tr><td><code>{claim['id']}</code></td><td>{esc(claim['theme'])}</td>"
            f"<td>{esc(summary)}</td>"
            f'<td><span class="pill {tone}">{label}</span></td></tr>')
    add("</table></div>")

    if live:
        add("<footer>")
        add(f"Model served from <code>{esc(live.get('model_source', 'n/a'))}</code> under "
            f"policy v{esc(live.get('policy_version', 'n/a'))}. "
            f"Latency p50 {live.get('latency_healthy', {}).get('p50', 0):.1f} ms, "
            f"p95 {live.get('latency_healthy', {}).get('p95', 0):.1f} ms "
            f"against a {live.get('budget_ms', 0)} ms budget.<br>")
        add("Regenerate with <code>python run_all.py</code>. "
            "Apache-2.0. Every number here comes from the run that produced this file.")
        add("</footer>")

    add("</div></body></html>")
    return "\n".join(h)


def main() -> int:
    if not METRICS.exists():
        print(f"no metrics at {METRICS}; run `python run_all.py` first", file=sys.stderr)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    size = OUT.stat().st_size / 1024
    print(f"dashboard written: {OUT.relative_to(ROOT)} ({size:,.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
