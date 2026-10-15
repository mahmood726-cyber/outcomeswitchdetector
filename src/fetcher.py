"""CT.gov API v2 response parser — converts raw JSON into normalized trial dicts."""

import json
import re


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def _extract_outcomes(outcomes_list):
    """Extract measure text from a list of outcome dicts."""
    if not outcomes_list:
        return []
    results = []
    for item in outcomes_list:
        measure = item.get("measure") or item.get("title", "")
        timeframe = item.get("timeFrame", "")
        results.append({
            "measure": measure.strip(),
            "timeFrame": timeframe.strip()
        })
    return results


def _extract_results_outcomes(outcome_measures, outcome_type):
    """Extract outcomes from resultsSection by type (PRIMARY or SECONDARY)."""
    if not outcome_measures:
        return []
    results = []
    for item in outcome_measures:
        if item.get("type", "").upper() == outcome_type.upper():
            title = item.get("title", "").strip()
            timeframe = item.get("timeFrame", "").strip()
            results.append({
                "measure": title,
                "timeFrame": timeframe
            })
    return results


def parse_api_response(raw):
    """Convert a CT.gov API v2 JSON study object into a normalized trial dict.

    Parameters
    ----------
    raw : dict
        A single study object from the CT.gov API v2 response.

    Returns
    -------
    dict with keys: nctId, status, phase, sponsor, sponsorClass, condition,
        protocolPrimary, protocolSecondary, resultsPrimary, resultsSecondary, hasResults
    """
    protocol = raw.get("protocolSection", {})
    results_section = raw.get("resultsSection", {})

    # Identification
    nct_id = _safe_get(protocol, "identificationModule", "nctId", default="UNKNOWN")

    # Status and dated registration metadata (never infer era from NCT number).
    status = _safe_get(protocol, "statusModule", "overallStatus", default="UNKNOWN")
    first_submitted = _safe_get(protocol, "statusModule", "studyFirstSubmitDate", default="")

    # Phase
    phases = _safe_get(protocol, "designModule", "phases", default=[])
    phase = phases[0] if phases else "N/A"

    # Sponsor
    sponsor = _safe_get(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name", default="Unknown")
    sponsor_class = _safe_get(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class", default="OTHER")

    # Condition
    conditions = _safe_get(protocol, "conditionsModule", "conditions", default=[])
    condition = conditions[0] if conditions else "Unknown"

    # Protocol outcomes
    protocol_primary = _extract_outcomes(
        _safe_get(protocol, "outcomesModule", "primaryOutcomes", default=[])
    )
    protocol_secondary = _extract_outcomes(
        _safe_get(protocol, "outcomesModule", "secondaryOutcomes", default=[])
    )

    # Results outcomes
    has_results = raw.get("hasResults", False)
    outcome_measures = _safe_get(results_section, "outcomeMeasuresModule", "outcomeMeasures", default=[])

    results_primary = _extract_results_outcomes(outcome_measures, "PRIMARY")
    results_secondary = _extract_results_outcomes(outcome_measures, "SECONDARY")

    # If resultsSection exists but hasResults isn't explicitly set, infer
    if results_section and outcome_measures:
        has_results = True

    return {
        "nctId": nct_id,
        "status": status,
        "studyFirstSubmitDate": first_submitted,
        "phase": phase,
        "sponsor": sponsor,
        "sponsorClass": sponsor_class,
        "condition": condition,
        "protocolPrimary": protocol_primary,
        "protocolSecondary": protocol_secondary,
        "resultsPrimary": results_primary,
        "resultsSecondary": results_secondary,
        "hasResults": has_results
    }


def fetch_completed_with_results(max_pages=1, page_size=20):
    """Fetch completed trials with results from CT.gov API v2.

    This function requires network access. For offline testing, use
    parse_api_response() with fixture data.

    Parameters
    ----------
    max_pages : int
        Maximum number of pages to fetch.
    page_size : int
        Number of studies per page.

    Returns
    -------
    list of normalized trial dicts
    """
    import requests

    if not isinstance(max_pages, int) or max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if not isinstance(page_size, int) or not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "filter.overallStatus": "COMPLETED",
        "aggFilters": "results:with",  # Official v2 results filter (not filter.resultsReported)
        "pageSize": page_size,
        "format": "json"
    }

    trials = []
    page_token = None
    seen_tokens = set()
    seen_ids = set()

    for _ in range(max_pages):
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies")
        if not isinstance(studies, list):
            raise ValueError("ClinicalTrials.gov response missing studies list")
        for study in studies:
            trial = parse_api_response(study)
            if trial["status"] != "COMPLETED" or not trial["hasResults"]:
                raise ValueError("API returned a study outside completed-with-results filter")
            nct_id = trial["nctId"]
            if not re.fullmatch(r"NCT\d{8}", nct_id):
                raise ValueError("API returned a missing or invalid NCT identifier")
            if nct_id in seen_ids:
                raise ValueError("Duplicate NCT identifier across API pages: " + nct_id)
            seen_ids.add(nct_id)
            trials.append(trial)

        new_token = data.get("nextPageToken")
        if new_token and (new_token in seen_tokens or new_token == page_token):
            raise ValueError("Repeated ClinicalTrials.gov pagination token")
        page_token = new_token
        if not page_token:
            break
        seen_tokens.add(page_token)

    return trials
