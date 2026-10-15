"""Sponsor / phase / condition / era rollups for detected switches."""

import re
from collections import defaultdict

# Descriptive date boundary only. NOT a determination of FDAAA applicability.
FDAAA_CUTOFF_YEAR = 2007


def _get_nct_year(trial):
    """Descriptive submission cohort based on actual CT.gov date, never NCT ID.

    Registration around 2007 alone never establishes FDAAA 801 obligations.
    Missing or malformed dates remain unknown, rather than being guessed.
    """
    from datetime import date

    raw = trial.get("studyFirstSubmitDate", "")
    try:
        registered = date.fromisoformat(raw)
    except (ValueError, TypeError):
        return "unknown_submission_date"
    return "submitted_before_2007" if registered.year < 2007 else "submitted_2007_or_later"


def aggregate_by_sponsor(analyzed_trials):
    """Roll up switch counts by sponsor.

    Parameters
    ----------
    analyzed_trials : list of dict
        Each dict has 'trial' (normalized trial dict) and 'switches' (list of switch dicts).

    Returns
    -------
    dict : {sponsor: {total_trials, trials_with_switches, switches_by_category, severity_counts}}
    """
    rollup = defaultdict(lambda: {
        "total_trials": 0,
        "trials_with_switches": 0,
        "switches_by_category": defaultdict(int),
        "severity_counts": defaultdict(int),
    })

    for item in analyzed_trials:
        trial = item["trial"]
        switches = item["switches"]
        sponsor = trial.get("sponsor", "Unknown")

        rollup[sponsor]["total_trials"] += 1
        if switches:
            rollup[sponsor]["trials_with_switches"] += 1

        for s in switches:
            rollup[sponsor]["switches_by_category"][s["category"]] += 1
            rollup[sponsor]["severity_counts"][s.get("severity", "LOW")] += 1

    # Convert defaultdicts to regular dicts for serialization
    result = {}
    for sponsor, data in rollup.items():
        result[sponsor] = {
            "total_trials": data["total_trials"],
            "trials_with_switches": data["trials_with_switches"],
            "switches_by_category": dict(data["switches_by_category"]),
            "severity_counts": dict(data["severity_counts"]),
        }
    return result


def aggregate_by_condition(analyzed_trials):
    """Roll up switch counts by condition.

    Parameters
    ----------
    analyzed_trials : list of dict

    Returns
    -------
    dict : {condition: {total_trials, trials_with_switches, switches_by_category}}
    """
    rollup = defaultdict(lambda: {
        "total_trials": 0,
        "trials_with_switches": 0,
        "switches_by_category": defaultdict(int),
    })

    for item in analyzed_trials:
        trial = item["trial"]
        switches = item["switches"]
        condition = trial.get("condition", "Unknown")

        rollup[condition]["total_trials"] += 1
        if switches:
            rollup[condition]["trials_with_switches"] += 1

        for s in switches:
            rollup[condition]["switches_by_category"][s["category"]] += 1

    result = {}
    for condition, data in rollup.items():
        result[condition] = {
            "total_trials": data["total_trials"],
            "trials_with_switches": data["trials_with_switches"],
            "switches_by_category": dict(data["switches_by_category"]),
        }
    return result


def aggregate_by_era(analyzed_trials):
    """Roll up *candidate discrepancies* by actual pre/2007+ submission date.

    Parameters
    ----------
    analyzed_trials : list of dict

    Returns
    -------
    dict : {era: {total_trials, trials_with_switches, switch_rate, total_switches}}
    """
    rollup = defaultdict(lambda: {
        "total_trials": 0,
        "trials_with_switches": 0,
        "total_switches": 0,
    })

    for item in analyzed_trials:
        trial = item["trial"]
        switches = item["switches"]
        era = _get_nct_year(trial)

        rollup[era]["total_trials"] += 1
        if switches:
            rollup[era]["trials_with_switches"] += 1
        rollup[era]["total_switches"] += len(switches)

    result = {}
    for era, data in rollup.items():
        total = data["total_trials"]
        result[era] = {
            "total_trials": total,
            "trials_with_switches": data["trials_with_switches"],
            "total_switches": data["total_switches"],
            "switch_rate": data["trials_with_switches"] / total if total > 0 else 0.0,
        }
    return result
