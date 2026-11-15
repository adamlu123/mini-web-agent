#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import build_cli_trace_viewer as viewer


VIEW_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-model OSWorld V2 trace viewer ZIP"
    )
    parser.add_argument(
        "--view",
        action="append",
        required=True,
        metavar="LABEL=SELECTION.tsv",
    )
    parser.add_argument("--results-root", type=Path, default=viewer.REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webp-quality", type=int, default=84)
    parser.add_argument("--workers", type=int, default=0, help="0 = one per view")
    parser.add_argument(
        "--task-workers",
        type=int,
        default=0,
        help="Workers per view; 0 apportions up to 64 total",
    )
    parser.add_argument(
        "--homepage-html",
        type=Path,
        help="Optional self-contained HTML report embedded on the homepage",
    )
    parser.add_argument(
        "--rubric-map",
        type=Path,
        help="Optional reconstructed evaluator rubric JSON keyed by view label",
    )
    parser.add_argument("--keep-build-dir", action="store_true")
    return parser.parse_args()


def parse_views(values: list[str]) -> list[tuple[str, Path]]:
    views = []
    labels = set()
    for value in values:
        label, separator, selection = value.partition("=")
        label = label.strip()
        if not separator or not selection or not VIEW_PATTERN.fullmatch(label):
            raise RuntimeError("--view must be a safe LABEL=SELECTION.tsv value")
        if label in labels:
            raise RuntimeError(f"Duplicate view label: {label}")
        labels.add(label)
        views.append((label, Path(selection).expanduser().resolve()))
    if len(views) < 2:
        raise RuntimeError("Comparison packages require at least two views")
    return views


def homepage(
    view_summaries: list[dict],
    report_href: str | None = None,
    full_report_href: str | None = None,
) -> str:
    total_tasks = sum(item["summary"]["taskCount"] for item in view_summaries)
    osworld = [
        item
        for item in view_summaries
        if item.get("homepageGroup") == "osworld" or item["label"] in {"5.5", "5.6"}
    ]
    supporting = [item for item in view_summaries if item not in osworld]

    def display_score(item: dict) -> float | None:
        summary = item["summary"]
        return (
            item.get("summaryScore")
            if item.get("summaryScore") is not None
            else summary["mean"]
        )

    osworld_rows = []
    for index, item in enumerate(osworld):
        score = display_score(item)
        cost = item.get("homepageCost")
        osworld_rows.append(
            f"""
            <tr class="model-row {'baseline' if index == 0 else 'frontier'}">
              <th><span class="model-mark" aria-hidden="true"></span>
                {html.escape(item["title"])}</th>
              <td>{score * 100:.2f}%</td>
              <td>{f'${cost:.2f}' if cost is not None else '—'}</td>
              <td>{item["summary"]["taskCount"]}</td>
              <td><a href="{html.escape(item["label"])}/index.html">Open trajectories →</a></td>
            </tr>
            """
        )
    comparison = ""
    if len(osworld) == 2:
        first_score, second_score = map(display_score, osworld)
        first_cost = osworld[0].get("homepageCost")
        second_cost = osworld[1].get("homepageCost")
        score_delta = (second_score - first_score) * 100
        cost_delta = (
            (1 - second_cost / first_cost) * 100
            if first_cost is not None and second_cost is not None
            else None
        )
        comparison = (
            f'<p class="delta-note"><strong>+{score_delta:.2f} percentage points</strong>'
            + (
                f" · <strong>${first_cost - second_cost:.2f} less per task "
                f"({cost_delta:.1f}%)</strong>"
                if cost_delta is not None
                else ""
            )
            + "</p>"
        )

    cards = []
    for item in supporting:
        score = display_score(item)
        metric = (
            f"{score * 100:.2f}%"
            if score is not None
            else f"{item['summary']['taskCount']} traces"
        )
        cards.append(
            f"""
            <article class="result-card">
              <span class="eyebrow">{item["summary"]["taskCount"]} trajectories</span>
              <h3>{html.escape(item["title"])}</h3>
              <strong>{metric}</strong>
              <p>{html.escape(item.get("summaryNote") or "")}</p>
              {f'<p class="claim">{html.escape(item["homepageClaim"])}</p>' if item.get("homepageClaim") else ''}
              <a href="{html.escape(item["label"])}/index.html">Review evidence →</a>
            </article>
            """
        )
    report = (
        f"""
        <section class="chart-section" id="price-performance">
          <div class="section-heading">
            <div>
              <span class="eyebrow">02 / Context</span>
              <h2>See where the runs land.</h2>
            </div>
            <a href="{html.escape(full_report_href or report_href)}"
               target="_blank" rel="noopener">
              Open full explorer ↗
            </a>
          </div>
          <iframe src="{html.escape(report_href)}"
                  title="Interactive OSWorld price-performance comparison"
                  loading="eager"></iframe>
          <p class="chart-caption">Costs use the documented pricing basis for
            each source; CUAWright values are reconstructed from trace tokens.
            Cross-harness differences are descriptive, not causal estimates.
            Some Anthropic values are digitized estimates.</p>
        </section>
        """
        if report_href
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAWright evaluations and traces</title>
  <style>
    :root {{ color: #20242c; background: #f3f0e9;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    a {{ color: #225fc5; }}
    header, main, footer {{ width: min(1180px, calc(100% - 40px)); margin: auto; }}
    header {{ padding: 64px 0 30px; border-bottom: 1px solid #d9dee6; }}
    .eyebrow {{ display: block; margin-bottom: 8px; color: #5d6879; font-size: 12px;
      font-weight: 750; letter-spacing: .08em; text-transform: uppercase; }}
    h1 {{ max-width: 900px; margin: 0; font-size: clamp(34px, 5vw, 58px);
      font-family: Georgia, ui-serif, serif; font-weight: 500;
      line-height: 1.06; letter-spacing: -.035em; }}
    .lede {{ max-width: 760px; margin: 20px 0 0; color: #4f5a6b;
      font-size: 18px; line-height: 1.55; }}
    .summary {{ margin-top: 26px; color: #596579; font: 12px ui-monospace,
      SFMono-Regular, Menlo, monospace; letter-spacing: .03em; text-transform: uppercase; }}
    main {{ padding: 38px 0 70px; }}
    h2 {{ margin: 0; font-family: Georgia, ui-serif, serif; font-size: 28px;
      font-weight: 500; }}
    .section-copy {{ max-width: 720px; margin: 8px 0 0; color: #596579;
      font-size: 14px; line-height: 1.5; }}
    .osworld {{ padding-bottom: 38px; border-bottom: 1px solid #d9dee6; }}
    .os-table {{ width: 100%; margin-top: 22px; border-collapse: collapse;
      border-top: 2px solid #20242c; border-bottom: 1px solid #cec8ba; }}
    .os-table th, .os-table td {{ padding: 15px 10px; border-bottom: 1px solid #cec8ba;
      text-align: right; font-variant-numeric: tabular-nums; }}
    .os-table thead th {{ color: #62685f; font-size: 10px; font-weight: 700;
      letter-spacing: .07em; text-transform: uppercase; }}
    .os-table th:first-child, .os-table td:first-child {{ text-align: left; }}
    .os-table tbody th {{ font-size: 14px; }}
    .os-table tbody td {{ font-size: 18px; font-weight: 700; }}
    .os-table tbody td:last-child {{ font-size: 12px; font-weight: 700; }}
    .os-table a, .result-card a {{ color: #225fc5; text-decoration: none; }}
    .model-mark {{ display: inline-block; width: 8px; height: 8px; margin-right: 8px;
      border-radius: 50%; background: #3159b8; }}
    .frontier .model-mark {{ background: #d94a38; }}
    .delta-note {{ margin: 16px 0 0; color: #424b43; font-size: 14px; }}
    .delta-note strong:first-child {{ color: #d94a38; }}
    .chart-section {{ margin-top: 42px; }}
    .section-heading {{ display: flex; align-items: end; justify-content: space-between;
      margin-bottom: 16px; }}
    .section-heading a {{ font-size: 13px; font-weight: 700; text-decoration: none; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #cec8ba;
      background: #fff; }}
    .chart-caption {{ max-width: 900px; margin: 10px 0 0; color: #667184;
      font-size: 12px; line-height: 1.5; }}
    .supporting {{ margin-top: 48px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
      margin-top: 18px; }}
    .result-card {{ display: flex; min-height: 250px; flex-direction: column;
      padding: 22px 18px 18px 0; border-top: 2px solid #20242c;
      background: transparent; }}
    .result-card h3 {{ margin: 0; font-size: 17px; }}
    .result-card > strong {{ margin: 20px 0 4px; font-size: 38px; letter-spacing: -.03em; }}
    .result-card p {{ margin: 4px 0; color: #596579; font-size: 13px; line-height: 1.45; }}
    .result-card .claim {{ color: #303946; }}
    .result-card a {{ margin-top: auto; }}
    .method {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px;
      margin-top: 32px; padding-top: 22px; border-top: 1px solid #d9dee6; }}
    .method h3 {{ margin: 0 0 7px; font-size: 14px; }}
    .method p {{ margin: 0; color: #596579; font-size: 13px; line-height: 1.55; }}
    footer {{ padding: 24px 0 50px; color: #667184; border-top: 1px solid #d9dee6;
      font-size: 12px; }}
    @media (max-width: 760px) {{
      header {{ padding-top: 44px; }}
      .summary {{ flex-wrap: wrap; }}
      .cards, .method {{ grid-template-columns: 1fr; }}
      .os-table {{ display: block; overflow-x: auto; }}
      iframe {{ height: 560px; }}
    }}
  </style>
</head>
<body>
  <header>
    <span class="eyebrow">CUAWright</span>
    <h1>How far can one terminal take a computer-use agent?</h1>
    <p class="lede">CUAWright gives the model one action—run a command—and lets
      it plan, inspect, click, code, and verify from there. Here are the results,
      the costs, and every trajectory behind them.</p>
    <div class="summary">{total_tasks} trajectories · 4 benchmark suites ·
      traces available by task</div>
  </header>
  <main>
    <section class="osworld">
      <span class="eyebrow">01 / OSWorld V2</span>
      <h2>More capability, at lower cost.</h2>
      <p class="section-copy">The same 105 tasks, run through two model
        generations. GPT-5.6 Sol reaches a higher mean reward while using fewer
        dollars per task at its original release price.</p>
      <table class="os-table">
        <thead><tr><th>Model</th><th>Mean partial reward</th>
          <th>Cost / task</th><th>Tasks</th><th>Run record</th></tr></thead>
        <tbody>{"".join(osworld_rows)}</tbody>
      </table>
      {comparison}
    </section>
    {report}
    <section class="supporting">
      <span class="eyebrow">03 / Beyond OSWorld</span>
      <h2>Three different tests of the same interface.</h2>
      <div class="cards">{"".join(cards)}</div>
      <div class="method">
        <div><h3>Different measures</h3><p>Each score belongs to its benchmark.
          OSWorld reward, WeaveBench PassRate, BenchCAD IoU, and CADGenBench
          aggregate scores should not be compared as one scale.</p></div>
        <div><h3>Inside each trace</h3><p>Provider-returned reasoning summaries,
          commands, host-recorded outputs, screenshots, specialist reports,
          verification records, and evaluator results.</p></div>
        <div><h3>Limits</h3><p>Hidden chain-of-thought and credentials are
          excluded. WeaveBench remains a candidate result pending upstream
          verification; CADGenBench exposes only its official aggregate.</p></div>
      </div>
    </section>
  </main>
  <footer>CUAWright makes agent behavior inspectable.
    <a href="https://github.com/TheoLeeCJ/cuawright-cua">Source on GitHub ↗</a>
  </footer>
</body>
</html>
"""


def embedded_report(source: str) -> str:
    style = """
  <style id="cuawright-homepage-embed">
    html, body { background: #fff !important; }
    .page { max-width: none !important; padding: 0 !important; }
    .page > header, .layout > aside, .summary-grid, .table-panel,
    .page > footer, .chart-toolbar { display: none !important; }
    .layout, .main-stack { display: block !important; }
    .chart-panel {
      border: 0 !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      padding: 0 !important;
    }
    .chart-shell { height: 680px !important; min-height: 680px !important;
      resize: none !important; }
    .chart-note { display: none !important; }
  </style>
"""
    if "</head>" not in source:
        raise RuntimeError("Homepage report does not contain </head>")
    return source.replace("</head>", style + "</head>", 1)


def build_view(
    label: str,
    selection_map: Path,
    build_dir: Path,
    results_root: Path,
    webp_quality: int,
    task_workers: int,
    rubric_map_path: Path | None,
) -> dict:
    rows = viewer.read_selection_map(selection_map)
    title = rows[0].get("viewer_title", "").strip() or label
    built = viewer.build_site(
        rows,
        build_dir / label,
        results_root,
        webp_quality,
        title=title,
        task_workers=task_workers,
        rubric_map=viewer.read_json(rubric_map_path, {}) if rubric_map_path else None,
        rubric_run_label=label,
    )
    summary_score = rows[0].get("summary_score", "").strip()
    homepage_cost = rows[0].get("homepage_cost", "").strip()
    return {
        "label": label,
        "title": title,
        "selectionMapSha256": viewer.hash_file(selection_map),
        "summary": built["catalog"]["summary"],
        "site": built["site"],
        "summaryScore": float(summary_score) if summary_score else None,
        "summaryNote": rows[0].get("summary_note", "").strip() or None,
        "homepageCost": float(homepage_cost) if homepage_cost else None,
        "homepageClaim": rows[0].get("homepage_claim", "").strip() or None,
        "homepageGroup": rows[0].get("homepage_group", "").strip() or None,
    }


def main() -> None:
    args = parse_args()
    views = parse_views(args.view)
    temporary = tempfile.TemporaryDirectory(prefix="osworld-trace-comparison-")
    build_dir = Path(temporary.name) / "site"
    build_dir.mkdir()
    requested_workers = args.workers or len(views)
    workers = max(1, min(len(views), requested_workers, os.cpu_count() or 1))
    available_task_workers = min(64, os.cpu_count() or 1)
    task_workers = args.task_workers or max(1, available_task_workers // workers)
    completed = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                build_view,
                label,
                selection_map,
                build_dir,
                args.results_root.resolve(),
                args.webp_quality,
                task_workers,
                args.rubric_map.resolve() if args.rubric_map else None,
            ): label
            for label, selection_map in views
        }
        for future in concurrent.futures.as_completed(futures):
            completed[futures[future]] = future.result()
    summaries = [completed[label] for label, _ in views]
    report_href = None
    full_report_href = None
    if args.homepage_html:
        report_path = args.homepage_html.expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"Homepage report not found: {report_path}")
        full_report_href = "performance.html"
        report_href = "performance-embed.html"
        source = report_path.read_text(encoding="utf-8")
        (build_dir / full_report_href).write_text(source, encoding="utf-8")
        (build_dir / report_href).write_text(embedded_report(source), encoding="utf-8")
    (build_dir / "index.html").write_text(
        homepage(summaries, report_href, full_report_href), encoding="utf-8"
    )
    manifest = {
        "format": "osworld-v2-cli-trace-comparison-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "homepageReport": report_href,
        "fullReport": full_report_href,
        "rubricMapSha256": (
            viewer.hash_file(args.rubric_map) if args.rubric_map else None
        ),
        "views": summaries,
    }
    (build_dir / "comparison-manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    site_stats = viewer.validate_site(build_dir)
    viewer.build_zip(build_dir, args.output)
    summary = {
        **site_stats,
        "zipPath": str(args.output),
        "zipBytes": args.output.stat().st_size,
        "zipSha256": viewer.hash_file(args.output),
        "views": summaries,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if args.keep_build_dir:
        destination = args.output.with_suffix("")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(build_dir, destination)
        print(f"Extracted site copied to {destination}")
    temporary.cleanup()


if __name__ == "__main__":
    main()
