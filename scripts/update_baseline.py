"""
Promote current evaluation scores to the committed baseline.

Run this after a deliberate model upgrade to accept the new score levels:

    python scripts/update_baseline.py

Then commit baselines/scores.json to lock in the new baseline.
"""

from _eval_helpers import BASELINE_PATH, run_evaluation, write_baseline


def main() -> None:
    print("Running full evaluation to generate new baseline...")
    scores = run_evaluation()

    write_baseline(scores)

    total_checks = sum(len(m) for m in scores.values())
    print(f"Baseline written to {BASELINE_PATH}")
    print(f"  {len(scores)} test cases, {total_checks} metric scores recorded")
    print()
    print("Next step: review baselines/scores.json, then commit it.")


if __name__ == "__main__":
    main()
