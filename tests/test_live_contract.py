"""Network-dependent CT.gov v2 completed-with-posted-results smoke test."""
from src.fetcher import fetch_completed_with_results


def test_real_ctgov_results_filter_contract():
    trials = fetch_completed_with_results(max_pages=1, page_size=3)
    assert len(trials) == 3
    ids = {t["nctId"] for t in trials}
    assert len(ids) == 3
    for trial in trials:
        assert trial["status"] == "COMPLETED"
        assert trial["hasResults"] is True
        assert trial["nctId"].startswith("NCT")
        assert len(trial["nctId"]) == 11
    print("ClinicalTrials.gov v2: verified three completed studies with posted results", sorted(ids))
