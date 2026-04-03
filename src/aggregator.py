"""Sponsor / phase / condition / era rollups for detected switches."""

import re
from collections import defaultdict

# FDAAA 2007 cutoff — trials after this date must report results
FDAAA_CUTOFF_YEAR = 2007


def _get_nct_year(nct_id):
    """Estimate enrollment era from NCT ID numeric portion.

    NCT IDs are assigned sequentially, so higher numbers = later trials.
    For our analysis, we use a simple heuristic:
    NCT00000001-NCT00500000 ~ pre-2007
    NCT00500001+ ~ post-FDAAA

    In production, use actual startDate from the protocol.
    """
    match = re.search(r'NCT(\d+)', nct_id)
    if not match:
        return "unknown"
    num = int(match.group(1))
    if num <= 500000:
        return "pre-FDAAA"
    return "post-FDAAA"


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
    """Roll up switch counts by era (pre/post FDAAA 2007).

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
        era = _get_nct_year(trial.get("nctId", ""))

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
