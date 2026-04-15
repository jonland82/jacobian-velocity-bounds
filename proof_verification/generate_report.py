from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from proof_verification.checks import run_all_checks
from proof_verification.report import write_html_report, write_results_json


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "proof_verification"
    html_path = output_dir / "verification_report.html"
    json_path = output_dir / "verification_results.json"

    results = run_all_checks(repo_root)
    write_results_json(results, json_path)
    write_html_report(results, html_path, repo_root)

    passed = sum(int(result.passed) for result in results)
    total = len(results)
    print(f"Generated {html_path}")
    print(f"Generated {json_path}")
    print(f"Verification checks passed: {passed}/{total}")


if __name__ == "__main__":
    main()
