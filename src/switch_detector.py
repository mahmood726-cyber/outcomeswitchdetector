"""Registered vs reported outcome comparison — detects 6 categories of switching."""

import re
from .endpoint_parser import normalize_endpoint, compute_similarity


# Similarity thresholds
SAME_THRESHOLD = 0.85
MODIFIED_THRESHOLD = 0.60


def _extract_measure_texts(outcomes):
    """Extract measure text from outcome dicts."""
    return [o.get("measure", "") for o in outcomes]


def _extract_timeframes(outcomes):
    """Extract timeframe text from outcome dicts."""
    return [o.get("timeFrame", "") for o in outcomes]


def _find_best_match(target, candidates):
    """Find the best matching candidate for target.

    Returns (index, similarity) or (-1, 0.0) if no match.
    """
    best_idx = -1
    best_sim = 0.0

    for i, candidate in enumerate(candidates):
        sim = compute_similarity(target, candidate)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    return best_idx, best_sim


def _timeframe_differs(tf1, tf2):
    """Check if two timeframes are substantively different after normalization."""
    norm1 = normalize_endpoint(tf1)
    norm2 = normalize_endpoint(tf2)

    # Extract numeric parts
    nums1 = re.findall(r'\d+', norm1)
    nums2 = re.findall(r'\d+', norm2)

    if nums1 != nums2:
        return True

    # Check unit difference
    units1 = re.findall(r'(?:month|week|day|year)s?', norm1)
    units2 = re.findall(r'(?:month|week|day|year)s?', norm2)

    if units1 != units2:
        return True

    return False


def detect_switches(trial):
    """Compare registered (protocol) vs reported (results) outcomes.

    Parameters
    ----------
    trial : dict
        Normalized trial dict with protocolPrimary, protocolSecondary,
        resultsPrimary, resultsSecondary fields.

    Returns
    -------
    list of dict
        Each dict has: category, endpoint, severity (set later), detail
    """
    switches = []

    proto_primary = _extract_measure_texts(trial.get("protocolPrimary", []))
    proto_secondary = _extract_measure_texts(trial.get("protocolSecondary", []))
    results_primary = _extract_measure_texts(trial.get("resultsPrimary", []))
    results_secondary = _extract_measure_texts(trial.get("resultsSecondary", []))

    proto_primary_tf = _extract_timeframes(trial.get("protocolPrimary", []))
    results_primary_tf = _extract_timeframes(trial.get("resultsPrimary", []))

    # Track which results endpoints have been matched
    matched_results_primary = set()
    matched_results_secondary = set()

    # 1. Check each protocol primary against results
    for i, pp in enumerate(proto_primary):
        # Check if it's in results primary
        idx_rp, sim_rp = _find_best_match(pp, results_primary)

        if sim_rp >= SAME_THRESHOLD and idx_rp not in matched_results_primary:
            matched_results_primary.add(idx_rp)
            # Check timeframe change
            if i < len(proto_primary_tf) and idx_rp < len(results_primary_tf):
                if _timeframe_differs(proto_primary_tf[i], results_primary_tf[idx_rp]):
                    switches.append({
                        "category": "TIMEFRAME_CHANGE",
                        "endpoint": pp,
                        "detail": f"Timeframe changed from '{proto_primary_tf[i]}' to '{results_primary_tf[idx_rp]}'"
                    })
            continue

        # Check if it was demoted to secondary
        idx_rs, sim_rs = _find_best_match(pp, results_secondary)
        if sim_rs >= SAME_THRESHOLD and idx_rs not in matched_results_secondary:
            matched_results_secondary.add(idx_rs)
            switches.append({
                "category": "DEMOTION",
                "endpoint": pp,
                "detail": f"Primary endpoint '{pp}' demoted to secondary in results"
            })
            continue

        # Check if it's a modified primary
        if sim_rp >= MODIFIED_THRESHOLD and idx_rp not in matched_results_primary:
            matched_results_primary.add(idx_rp)
            switches.append({
                "category": "MEASURE_MODIFIED",
                "endpoint": pp,
                "detail": f"Primary endpoint modified: '{pp}' -> '{results_primary[idx_rp]}' (similarity={sim_rp:.2f})"
            })
            continue

        # Not found anywhere — PRIMARY_REMOVED
        switches.append({
            "category": "PRIMARY_REMOVED",
            "endpoint": pp,
            "detail": f"Primary endpoint '{pp}' not found in results"
        })

    # 2. Check each protocol secondary against results
    for j, ps in enumerate(proto_secondary):
        # Check if it was promoted to primary
        idx_rp, sim_rp = _find_best_match(ps, results_primary)
        if sim_rp >= SAME_THRESHOLD and idx_rp not in matched_results_primary:
            matched_results_primary.add(idx_rp)
            switches.append({
                "category": "PROMOTION",
                "endpoint": ps,
                "detail": f"Secondary endpoint '{ps}' promoted to primary in results"
            })
            continue

        # Check if it's in results secondary (normal)
        idx_rs, sim_rs = _find_best_match(ps, results_secondary)
        if sim_rs >= SAME_THRESHOLD and idx_rs not in matched_results_secondary:
            matched_results_secondary.add(idx_rs)
            continue

        # Modified secondary — don't flag (only track primary changes)

    # 3. Check for added primary endpoints
    for k, rp in enumerate(results_primary):
        if k not in matched_results_primary:
            switches.append({
                "category": "PRIMARY_ADDED",
                "endpoint": rp,
                "detail": f"Primary endpoint '{rp}' in results but not in protocol"
            })

    return switches
