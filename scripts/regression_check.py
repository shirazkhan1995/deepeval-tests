"""
CI regression gate for LLM evaluation.

Runs the full golden dataset evaluation and compares scores against the
committed baseline in baselines/scores.json.

Exit codes:
  0 — all checks pass (or first-run baseline was written)
  1 — one or more regressions detected

Configure tolerance via REGRESSION_TOLERANCE env var (default: 0.05).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from _eval_helpers import BASELINE_PATH, load_baseline, run_evaluation, write_baseline

REPORTS_DIR = Path(__file__).parent.parent / "reports"
TOLERANCE = float(os.environ.get("REGRESSION_TOLERANCE", "0.05"))


def _compare(
    baseline_scores: dict[str, dict[str, float | None]],
    current_scores: dict[str, dict[str, float | None]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (regressions, improvements, all_comparisons).
    Each entry: {test_id, metric, baseline, current, delta, status}
    """
    regressions, improvements, all_comparisons = [], [], []

    for test_id, baseline_metrics in baseline_scores.items():
        current_metrics = current_scores.get(test_id, {})

        for metric_name, baseline_score in baseline_metrics.items():
            current_score = current_metrics.get(metric_name)

            if current_score is None:
                # Missing score = treat as full regression
                delta = -baseline_score if baseline_score is not None else 0.0
                print(
                    f"  WARNING: metric '{metric_name}' missing for '{test_id}' "
                    "(error during evaluation?)"
                )
            else:
                delta = current_score - baseline_score

            delta = round(delta, 4)

            if delta < -TOLERANCE:
                status = "REGRESSED"
            elif delta > TOLERANCE:
                status = "IMPROVED"
            else:
                status = "OK"

            entry = {
                "test_id": test_id,
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "delta": delta,
                "status": status,
            }
            all_comparisons.append(entry)
            if status == "REGRESSED":
                regressions.append(entry)
            elif status == "IMPROVED":
                improvements.append(entry)

    return regressions, improvements, all_comparisons


def _write_json_report(
    regressions: list[dict],
    improvements: list[dict],
    all_comparisons: list[dict],
) -> None:
    total = len(all_comparisons)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "tolerance": TOLERANCE,
        "overall_status": "FAILED" if regressions else "PASSED",
        "summary": {
            "total_checks": total,
            "regressions": len(regressions),
            "improvements": len(improvements),
            "unchanged": total - len(regressions) - len(improvements),
        },
        "regressions": regressions,
        "improvements": improvements,
        "all_scores": _group_by_test_id(all_comparisons),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "regression_diff.json", "w") as f:
        json.dump(report, f, indent=2)


def _group_by_test_id(comparisons: list[dict]) -> dict:
    grouped: dict = {}
    for c in comparisons:
        tid = c["test_id"]
        grouped.setdefault(tid, {})
        grouped[tid][c["metric"]] = {
            "baseline": c["baseline"],
            "current": c["current"],
            "delta": c["delta"],
            "status": c["status"],
        }
    return grouped


def _write_md_report(
    regressions: list[dict],
    improvements: list[dict],
    all_comparisons: list[dict],
) -> None:
    status_line = (
        f"**FAILED** — {len(regressions)} regression(s) detected"
        if regressions
        else "**PASSED** — no regressions"
    )
    lines = [
        "# Model Regression Report",
        "",
        f"**Run date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Tolerance:** {TOLERANCE}",
        f"**Overall status:** {status_line}",
        "",
    ]

    if regressions:
        lines += [
            "## Regressions",
            "",
            "| Test ID | Metric | Baseline | Current | Delta |",
            "|---------|--------|----------|---------|-------|",
        ]
        for r in regressions:
            lines.append(
                f"| {r['test_id']} | {r['metric']} "
                f"| {r['baseline']:.4f} | {r['current']:.4f} | {r['delta']:+.4f} |"
            )
        lines.append("")

    if improvements:
        lines += [
            "## Improvements",
            "",
            "| Test ID | Metric | Baseline | Current | Delta |",
            "|---------|--------|----------|---------|-------|",
        ]
        for imp in improvements:
            lines.append(
                f"| {imp['test_id']} | {imp['metric']} "
                f"| {imp['baseline']:.4f} | {imp['current']:.4f} | {imp['delta']:+.4f} |"
            )
        lines.append("")

    lines += [
        "## Full Score Comparison",
        "",
        "| Test ID | Metric | Baseline | Current | Delta | Status |",
        "|---------|--------|----------|---------|-------|--------|",
    ]
    for c in all_comparisons:
        cur = f"{c['current']:.4f}" if c["current"] is not None else "N/A"
        lines.append(
            f"| {c['test_id']} | {c['metric']} "
            f"| {c['baseline']:.4f} | {cur} | {c['delta']:+.4f} | {c['status']} |"
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "regression_diff.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    print(f"Running evaluation (tolerance={TOLERANCE})...")
    current_scores = run_evaluation()

    # First-run bootstrap
    if not BASELINE_PATH.exists():
        print("No baseline found — writing initial baseline.")
        write_baseline(current_scores)
        print(f"Initial baseline written to {BASELINE_PATH}")
        print("Commit baselines/scores.json to lock it in.")
        sys.exit(0)

    baseline = load_baseline()
    baseline_scores = baseline["scores"]

    regressions, improvements, all_comparisons = _compare(baseline_scores, current_scores)

    _write_json_report(regressions, improvements, all_comparisons)
    _write_md_report(regressions, improvements, all_comparisons)

    total = len(all_comparisons)
    print(f"\nResults: {total} checks — "
          f"{len(regressions)} regressed, {len(improvements)} improved, "
          f"{total - len(regressions) - len(improvements)} unchanged")

    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(
                f"  [{r['test_id']}] {r['metric']}: "
                f"{r['baseline']:.4f} → {r['current']:.4f}  (Δ {r['delta']:+.4f})"
            )
        print(f"\nDiff report: {REPORTS_DIR / 'regression_diff.md'}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
