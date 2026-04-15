from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
import platform
import re
import sys

from proof_verification.checks import CheckResult


EXTRA_CSS = """
    .badge {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      padding: .35rem .65rem;
      border-radius: 999px;
      font-size: .68rem;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
      border: 1px solid rgba(16,21,34,.08);
      background: rgba(255,255,255,.54);
      color: var(--muted);
    }
    .badge.pass {
      color: #184f3e;
      background: rgba(82, 163, 110, .12);
      border-color: rgba(24, 79, 62, .16);
    }
    .badge.fail {
      color: #7d2630;
      background: rgba(188, 72, 84, .12);
      border-color: rgba(125, 38, 48, .16);
    }
    .check-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
    .check-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      padding: 1rem;
    }
    .check-card h3 { font-size: 1.45rem; margin: .45rem 0 .35rem; }
    .check-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: .75rem;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: .4rem;
      padding: .32rem .55rem;
      border-radius: 999px;
      background: rgba(255,255,255,.66);
      border: 1px solid rgba(16,21,34,.06);
      color: var(--muted);
      font-size: .72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
    }
    .metric-table-wrap {
      margin-top: .8rem;
      overflow-x: auto;
      border-radius: 14px;
      border: 1px solid rgba(16,21,34,.08);
      background: rgba(255,255,255,.72);
    }
    table.metrics {
      width: 100%;
      border-collapse: collapse;
      font-size: .88rem;
    }
    table.metrics th,
    table.metrics td {
      padding: .7rem .8rem;
      text-align: left;
      border-bottom: 1px solid rgba(16,21,34,.07);
      vertical-align: top;
    }
    table.metrics th {
      color: var(--muted);
      font-size: .75rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      background: rgba(255,255,255,.58);
    }
    table.metrics tr:last-child td { border-bottom: 0; }
    .detail-list {
      list-style: none;
      margin: .8rem 0 0;
      padding: 0;
      display: grid;
      gap: .5rem;
    }
    .detail-list li {
      padding: .75rem .85rem;
      border-radius: 14px;
      background: rgba(255,255,255,.66);
      border: 1px solid rgba(16,21,34,.06);
      color: var(--muted);
    }
    .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem; }
    .summary-card {
      padding: .85rem;
      border-radius: 14px;
      background: rgba(255,255,255,.68);
      border: 1px solid rgba(16,21,34,.06);
    }
    .summary-card strong { display: block; font-size: 1.2rem; line-height: 1; margin-bottom: .25rem; }
    .summary-card span { display: block; font-size: .8rem; color: var(--muted); }
    .status-line {
      margin-top: .85rem;
      display: inline-flex;
      align-items: center;
      gap: .55rem;
      padding: .55rem .8rem;
      border-radius: 14px;
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(16,21,34,.08);
      font-weight: 700;
    }
    .status-line.pass { color: #184f3e; }
    .status-line.fail { color: #7d2630; }
    @media (max-width: 1080px) {
      .check-grid, .summary-grid { grid-template-columns: 1fr; }
    }
"""


def _extract_index_assets(index_path: Path) -> tuple[list[str], str, str]:
    text = index_path.read_text(encoding="utf-8")
    font_links = re.findall(
        r'(<link[^>]+(?:fonts\.googleapis|fonts\.gstatic)[^>]*>)',
        text,
        flags=re.IGNORECASE,
    )
    style_match = re.search(r"<style>(.*?)</style>", text, flags=re.DOTALL | re.IGNORECASE)
    script_matches = re.findall(
        r"(<script(?: async)?[^>]*>.*?</script>)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not style_match:
        raise ValueError(f"Could not extract style block from {index_path}")

    mathjax_scripts = []
    for script in script_matches:
        if "MathJax" in script or "tex-svg.js" in script:
            mathjax_scripts.append(script)
    return font_links, style_match.group(1), "\n".join(mathjax_scripts)


def _format_metric(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value}"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e3 or magnitude < 1e-3:
            return f"{value:.6e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _render_metric_rows(metrics: dict[str, object]) -> str:
    rows = []
    for key, value in metrics.items():
        label = escape(key.replace("_", " "))
        rows.append(
            f"<tr><td>{label}</td><td><code>{escape(_format_metric(value))}</code></td></tr>"
        )
    if not rows:
        return ""
    return (
        '<div class="metric-table-wrap"><table class="metrics">'
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_check_card(result: CheckResult) -> str:
    badge_class = "pass" if result.passed else "fail"
    badge_text = "Pass" if result.passed else "Fail"
    details = "".join(f"<li>{escape(item)}</li>" for item in result.details)
    math_blocks = "".join(
        f'<div class="math">\\[{block}\\]</div>' for block in result.math_blocks
    )
    metrics_html = _render_metric_rows(result.metrics)
    return f"""
      <article class="check-card">
        <div class="check-meta">
          <span class="pill">{escape(result.method)}</span>
          <span class="badge {badge_class}">{badge_text}</span>
        </div>
        <h3>{escape(result.title)}</h3>
        <p>{escape(result.summary)}</p>
        {math_blocks}
        <ul class="detail-list">{details}</ul>
        {metrics_html}
      </article>
    """


def _render_section(section_id: str, heading: str, copy: str, results: list[CheckResult]) -> str:
    cards = "".join(_render_check_card(result) for result in results)
    return f"""
      <section class="section" id="{escape(section_id)}">
        <div class="wrap">
          <div class="section-head">
            <div>
              <div class="kicker">Verification Block</div>
              <h2>{escape(heading)}</h2>
            </div>
            <p>{escape(copy)}</p>
          </div>
          <div class="check-grid">
            {cards}
          </div>
        </div>
      </section>
    """


def _build_summary(results: list[CheckResult]) -> dict[str, int]:
    by_category: dict[str, int] = defaultdict(int)
    passes = 0
    for result in results:
        by_category[result.category] += 1
        passes += int(result.passed)
    return {
        "total": len(results),
        "passed": passes,
        "exact symbolic": by_category["exact symbolic"],
        "numerical": by_category["numerical"],
        "artifact consistency": by_category["artifact consistency"],
    }


def write_results_json(results: list[CheckResult], output_path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "results": [result.to_dict() for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_html_report(results: list[CheckResult], output_path: Path, repo_root: Path) -> None:
    font_links, base_style, mathjax_scripts = _extract_index_assets(repo_root / "index.html")
    summary = _build_summary(results)
    exact = [result for result in results if result.category == "exact symbolic"]
    numerical = [result for result in results if result.category == "numerical"]
    artifacts = [result for result in results if result.category == "artifact consistency"]
    all_pass = all(result.passed for result in results)
    status_class = "pass" if all_pass else "fail"
    status_text = "All verification checks passed." if all_pass else "Some verification checks failed."
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    summary_cards = "".join(
        f"""
        <div class="summary-card">
          <strong>{escape(str(value))}</strong>
          <span>{escape(label.title())}</span>
        </div>
        """
        for label, value in summary.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Proof Verification Report</title>
  <meta
    name="description"
    content="Symbolic and numerical verification report for the Jacobian-velocity bounds manuscript."
  >
  {' '.join(font_links)}
  <style>
{base_style}
{EXTRA_CSS}
  </style>
  {mathjax_scripts}
</head>
<body>
  <header class="topbar">
    <div class="wrap navline">
      <a class="brand" href="#top"><span class="mark">J</span><span>Proof Verification</span></a>
      <nav class="nav" aria-label="Primary">
        <a href="#summary">Summary</a>
        <a href="#exact">Exact</a>
        <a href="#numerical">Numerical</a>
        <a href="#artifacts">Artifacts</a>
        <a href="../index.html">Project Site</a>
        <a class="btn primary" href="../jacobian_velocity_bounds_deployment_risk_covariate_drift.pdf">Read Manuscript</a>
      </nav>
    </div>
  </header>

  <main id="top">
    <section class="hero">
      <div class="wrap hero-grid">
        <div class="hero-copy">
          <div class="hero-top">
            <div class="eyebrow">Symbolic and Numerical Checks</div>
            <h1>Formal verification for the theorem chain and cached experiment bounds.</h1>
            <div class="hero-meta">
              Generated {escape(generated_at)}<br>
              Python {escape(platform.python_version())} on {escape(platform.platform())}
            </div>
            <p>
              This report verifies the paper's main mathematical steps in three ways: exact symbolic identities
              for the theorem package, numerical stress tests for the low-rank inequalities, and direct consistency
              checks against the cached experiment summaries already committed in the repository.
            </p>
          </div>
          <div class="hero-bottom">
            <div class="actions">
              <a class="btn primary" href="./verification_results.json">Open JSON Results</a>
              <a class="btn secondary" href="../index.html">Back to Project Site</a>
            </div>
            <div class="summary-grid">
              {summary_cards}
            </div>
            <div class="status-line {status_class}">{escape(status_text)}</div>
          </div>
        </div>
        <div class="repo-card" id="summary">
          <div class="kicker">Run Command</div>
          <h3>Reproduce the report</h3>
          <p>The verifier reads the manuscript-adjacent artifacts from the repository and writes both JSON and HTML outputs into <code>proof_verification/</code>.</p>
          <div class="command-block">
            <div class="command-line">python proof_verification/generate_report.py</div>
          </div>
          <ul class="repo-list">
            <li>
              <strong>Exact symbolic checks</strong>
              Sharpness of the Poincare bound, a deterministic equality case for the Jacobian-velocity theorem,
              the composition-chain identity, and the Bernoulli cross-entropy derivative bound.
            </li>
            <li>
              <strong>Numerical checks</strong>
              Randomized low-rank stress tests and a nontrivial expectation example verifying the full inequality chain.
            </li>
            <li>
              <strong>Artifact checks</strong>
              Row-by-row validation that the cached synthetic experiment summaries satisfy the stored upper bounds.
            </li>
          </ul>
        </div>
      </div>
    </section>

    {_render_section("exact", "Exact Symbolic Checks", "These checks use SymPy to simplify the main algebraic identities and sharp examples exactly.", exact)}
    {_render_section("numerical", "Numerical Stress Tests", "These checks verify the theorem chain and low-rank inequalities in settings where exact symbolic proof is less natural but dense numerical evaluation is informative.", numerical)}
    {_render_section("artifacts", "Cached Experiment Artifact Checks", "These checks validate the inequality columns stored in the committed CSV summaries that drive the synthetic theorem and directional-ablation figures.", artifacts)}
  </main>

  <footer class="footer">
    <div class="wrap footer-line">
      <div>
        <h3>Jacobian-Velocity Bounds</h3>
        <p>Proof verification report generated from the local repository state.</p>
      </div>
      <div>
        <a class="btn secondary" href="../jacobian_velocity_bounds_deployment_risk_covariate_drift.tex">Open LaTeX Source</a>
      </div>
    </div>
  </footer>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

