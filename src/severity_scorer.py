"""Discrepancy severity scoring: HIGH / MEDIUM / LOW."""


# Severity mapping by switch category
SEVERITY_MAP = {
    "PRIMARY_ADDED": "HIGH",
    "PRIMARY_REMOVED": "HIGH",
    "PROMOTION": "HIGH",
    "DEMOTION": "HIGH",
    "TIMEFRAME_CHANGE": "MEDIUM",
    "MEASURE_MODIFIED": "LOW",
}


def score_severity(switch):
    """Score the severity of a single switch.

    Parameters
    ----------
    switch : dict
        A switch dict with 'category' key.

    Returns
    -------
    str : 'HIGH', 'MEDIUM', or 'LOW'
    """
    category = switch.get("category", "")
    return SEVERITY_MAP.get(category, "LOW")


def score_all(switches):
    """Score severity for a list of switches, adding 'severity' key to each.

    Parameters
    ----------
    switches : list of dict
        Switch dicts from detect_switches().

    Returns
    -------
    list of dict : Same switches with 'severity' key added.
    """
    for s in switches:
        s["severity"] = score_severity(s)
    return switches
