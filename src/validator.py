"""COMPare benchmark harness — validates detected switches against ground truth."""

from .endpoint_parser import compute_similarity


MATCH_THRESHOLD = 0.85


def _match_switch(detected, expected):
    """Check if a detected switch matches an expected switch.

    Match criteria: same category AND endpoint similarity >= threshold.
    """
    if detected.get("category") != expected.get("category"):
        return False

    sim = compute_similarity(
        detected.get("endpoint", ""),
        expected.get("endpoint", "")
    )
    return sim >= MATCH_THRESHOLD


def validate_against_benchmark(benchmark_trials, detector_fn=None):
    """Compare detected switches against ground truth expectedSwitches.

    Parameters
    ----------
    benchmark_trials : list of dict
        Each dict has: nctId, protocolPrimary, protocolSecondary,
        resultsPrimary, resultsSecondary, expectedSwitches.
    detector_fn : callable, optional
        Function that takes a trial dict and returns list of switches.
        If None, uses switch_detector.detect_switches.

    Returns
    -------
    tuple (concordance: float, details: list of dict)
        concordance: proportion of benchmark trials where detection matches ground truth
        details: per-trial validation results
    """
    if detector_fn is None:
        from .switch_detector import detect_switches
        detector_fn = detect_switches

    details = []
    correct = 0

    for bt in benchmark_trials:
        # Build trial dict for the detector
        trial = {
            "nctId": bt.get("nctId", ""),
            "protocolPrimary": [{"measure": m, "timeFrame": ""} for m in bt.get("protocolPrimary", [])],
            "protocolSecondary": [{"measure": m, "timeFrame": ""} for m in bt.get("protocolSecondary", [])],
            "resultsPrimary": [{"measure": m, "timeFrame": ""} for m in bt.get("resultsPrimary", [])],
            "resultsSecondary": [{"measure": m, "timeFrame": ""} for m in bt.get("resultsSecondary", [])],
        }

        detected = detector_fn(trial)
        expected = bt.get("expectedSwitches", [])

        # Check match: each expected should have a detected counterpart and vice versa
        matched_detected = set()
        matched_expected = set()

        for ei, exp in enumerate(expected):
            for di, det in enumerate(detected):
                if di not in matched_detected and _match_switch(det, exp):
                    matched_detected.add(di)
                    matched_expected.add(ei)
                    break

        true_positives = len(matched_expected)
        false_positives = len(detected) - len(matched_detected)
        false_negatives = len(expected) - len(matched_expected)

        is_correct = (false_positives == 0 and false_negatives == 0)
        if is_correct:
            correct += 1

        details.append({
            "nctId": bt.get("nctId", ""),
            "expected_count": len(expected),
            "detected_count": len(detected),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "correct": is_correct,
            "detected_switches": detected,
            "expected_switches": expected,
        })

    concordance = correct / len(benchmark_trials) if benchmark_trials else 0.0

    return concordance, details
