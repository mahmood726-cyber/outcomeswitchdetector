"""Pipeline runner — fetches trials, detects switches, scores severity, aggregates."""

import json
import os
import sys

from src.fetcher import parse_api_response
from src.switch_detector import detect_switches
from src.severity_scorer import score_all
from src.aggregator import aggregate_by_sponsor, aggregate_by_condition, aggregate_by_era


def run_pipeline(input_path=None, output_path=None):
    """Run the full OutcomeSwitchDetector pipeline.

    Parameters
    ----------
    input_path : str, optional
        Path to a JSON file containing raw trial data.
        If None, uses fixture data.
    output_path : str, optional
        Path to write dashboard_data.json.
        If None, writes to data/processed/dashboard_data.json.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if input_path is None:
        input_path = os.path.join(base_dir, "data", "fixtures", "sample_trials.json")

    if output_path is None:
        output_path = os.path.join(base_dir, "data", "processed", "dashboard_data.json")

    # 1. Load and parse
    print(f"Loading trials from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_trials = json.load(f)

    trials = [parse_api_response(raw) for raw in raw_trials]
    print(f"Parsed {len(trials)} trials")

    # 2. Detect switches and score severity
    analyzed = []
    total_switches = 0
    for trial in trials:
        if not trial["hasResults"]:
            continue
        switches = detect_switches(trial)
        switches = score_all(switches)
        total_switches += len(switches)
        analyzed.append({"trial": trial, "switches": switches})

    print(f"Analyzed {len(analyzed)} trials with results, found {total_switches} switches")

    # 3. Aggregate
    by_sponsor = aggregate_by_sponsor(analyzed)
    by_condition = aggregate_by_condition(analyzed)
    by_era = aggregate_by_era(analyzed)

    # 4. Build dashboard data
    dashboard_data = {
        "summary": {
            "total_trials": len(analyzed),
            "trials_with_switches": sum(1 for a in analyzed if a["switches"]),
            "total_switches": total_switches,
            "switch_rate": sum(1 for a in analyzed if a["switches"]) / len(analyzed) if analyzed else 0.0,
        },
        "by_sponsor": by_sponsor,
        "by_condition": by_condition,
        "by_era": by_era,
        "trials": [
            {
                "nctId": a["trial"]["nctId"],
                "sponsor": a["trial"]["sponsor"],
                "condition": a["trial"]["condition"],
                "phase": a["trial"]["phase"],
                "switches": a["switches"],
            }
            for a in analyzed
        ],
    }

    # 5. Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Dashboard data written to: {output_path}")

    return dashboard_data


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(input_file, output_file)
