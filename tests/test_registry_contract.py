"""Contract tests for source evidence and safe non-historical interpretation."""
from copy import deepcopy

import pytest

from src.fetcher import fetch_completed_with_results, parse_api_response
from src.aggregator import aggregate_by_era
from src.switch_detector import detect_switches
from run_analysis import run_pipeline


def api_study(nct, *, result=True, status="COMPLETED", date="2020-01-01"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "statusModule": {"overallStatus": status, "studyFirstSubmitDate": date},
            "outcomesModule": {"primaryOutcomes": [{"measure": "Mortality", "timeFrame": "12 months"}]},
            "designModule": {"phases": ["PHASE3"]},
        },
        "hasResults": result,
        "resultsSection": {
            "outcomeMeasuresModule": {"outcomeMeasures": [
                {"title": "Mortality", "type": "PRIMARY", "timeFrame": "12 months"}
            ]}
        } if result else {},
    }


def test_results_filter_uses_actual_v2_parameter(monkeypatch):
    observed = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"studies": [api_study("NCT00000001")]}
    def fake_get(url, params=None, timeout=None):
        observed.append(params.copy())
        return Response()
    monkeypatch.setattr("requests.get", fake_get)
    trials = fetch_completed_with_results(max_pages=1, page_size=5)
    assert len(trials) == 1
    assert observed[0]["aggFilters"] == "results:with"
    assert "filter.resultsReported" not in observed[0]
    assert trials[0]["studyFirstSubmitDate"] == "2020-01-01"


def test_api_rejects_wrong_study_status_or_results(monkeypatch):
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"studies": [api_study("NCT00000001", result=False)]}
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="outside"):
        fetch_completed_with_results(max_pages=1, page_size=5)


def test_invalid_page_size_is_rejected_before_request():
    with pytest.raises(ValueError, match="page_size"):
        fetch_completed_with_results(max_pages=1, page_size=1001)


def test_submission_dates_not_nct_id_determine_era():
    old = parse_api_response(api_study("NCT99999999", date="2004-06-09"))
    recent = parse_api_response(api_study("NCT00000001", date="2016-07-01"))
    unknown = parse_api_response(api_study("NCT00000002", date=""))
    found = aggregate_by_era([{"trial": x, "switches": []} for x in [old, recent, unknown]])
    assert set(found) == {
        "submitted_before_2007", "submitted_2007_or_later", "unknown_submission_date"
    }


def test_similar_primary_outcomes_match_distinct_results_without_duplicate_claims():
    trial = {
        "protocolPrimary": [
            {"measure": "Mortality from cardiovascular disease"},
            {"measure": "Mortality from cardiovascular causes"},
        ],
        "protocolSecondary": [],
        "resultsPrimary": [
            {"measure": "Mortality from cardiovascular disease"},
            {"measure": "Mortality from cardiovascular causes"},
        ],
        "resultsSecondary": [],
    }
    assert detect_switches(trial) == []


def test_detected_changes_are_unverified_candidate_discordances():
    trial = {
        "protocolPrimary": [{"measure": "Overall survival", "timeFrame": "12 months"}],
        "protocolSecondary": [],
        "resultsPrimary": [{"measure": "Heart rate", "timeFrame": "12 months"}],
        "resultsSecondary": [],
    }
    changes = detect_switches(trial)
    assert changes
    assert all(c["requires_archived_protocol_history"] for c in changes)
    assert all(c["verification_status"] == "UNVERIFIED_CURRENT_RECORD_DISCORDANCE" for c in changes)


def test_pipeline_skips_posted_results_missing_primary_outcomes(tmp_path):
    record = api_study("NCT00000001")
    record["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"] = []
    source = tmp_path / "raw.json"
    output = tmp_path / "out.json"
    import json
    source.write_text(json.dumps([record]))
    result = run_pipeline(input_path=str(source), output_path=str(output))
    assert result["summary"]["total_trials"] == 0
    assert result["summary"]["unassessable_trials_no_primary_results"] == 1
    assert result["summary"]["evidence_status"] == "UNVERIFIED_CURRENT_RECORD_DISCORDANCE"
