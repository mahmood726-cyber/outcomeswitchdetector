"""25 tests for OutcomeSwitchDetector."""

import json
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fetcher import parse_api_response
from src.endpoint_parser import normalize_endpoint, compute_similarity
from src.switch_detector import detect_switches
from src.severity_scorer import score_severity, score_all
from src.aggregator import aggregate_by_sponsor, aggregate_by_condition, aggregate_by_era
from src.validator import validate_against_benchmark


# ============================================================
# FETCHER TESTS (2)
# ============================================================

class TestFetcher:
    def test_parse_api_response_with_results(self, clean_trial):
        """Parse a trial with results section."""
        trial = parse_api_response(clean_trial)
        assert trial["nctId"] == "NCT00000001"
        assert trial["status"] == "COMPLETED"
        assert trial["phase"] == "PHASE3"
        assert trial["sponsor"] == "AcmePharma"
        assert trial["sponsorClass"] == "INDUSTRY"
        assert trial["condition"] == "Heart Failure"
        assert trial["hasResults"] is True
        assert len(trial["protocolPrimary"]) == 1
        assert len(trial["resultsPrimary"]) == 1

    def test_parse_api_response_without_results(self):
        """Parse a trial without results section."""
        raw = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT99999999"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "TestSponsor", "class": "ACADEMIC"}
                },
                "conditionsModule": {"conditions": ["Asthma"]},
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "FEV1 change", "timeFrame": "12 weeks"}
                    ],
                    "secondaryOutcomes": []
                }
            },
            "hasResults": False
        }
        trial = parse_api_response(raw)
        assert trial["nctId"] == "NCT99999999"
        assert trial["hasResults"] is False
        assert len(trial["resultsPrimary"]) == 0
        assert len(trial["resultsSecondary"]) == 0
        assert len(trial["protocolPrimary"]) == 1


# ============================================================
# ENDPOINT PARSER TESTS (5)
# ============================================================

class TestEndpointParser:
    def test_abbreviation_expansion(self):
        """Abbreviations like LVEF should be expanded."""
        result = normalize_endpoint("Change in LVEF from baseline")
        assert "left ventricular ejection fraction" in result

    def test_timeframe_normalization(self):
        """52 weeks should normalize to 12 months."""
        result = normalize_endpoint("Change at 52 weeks")
        assert "12 months" in result

    def test_case_normalization(self):
        """All output should be lowercase."""
        result = normalize_endpoint("Overall Survival at 36 Months")
        assert result == result.lower()

    def test_similarity_identical(self):
        """Identical endpoints should have similarity 1.0."""
        sim = compute_similarity(
            "Change in LVEF from baseline",
            "Change in LVEF from baseline"
        )
        assert abs(sim - 1.0) < 1e-9

    def test_similarity_synonym(self):
        """Synonymous endpoints should have high similarity."""
        sim = compute_similarity(
            "Change in LVEF from baseline",
            "Change in Left Ventricular Ejection Fraction From Baseline"
        )
        assert sim >= 0.85


# ============================================================
# SWITCH DETECTOR TESTS (6)
# ============================================================

class TestSwitchDetector:
    def test_no_switches(self, clean_trial):
        """Clean trial should have no switches detected."""
        trial = parse_api_response(clean_trial)
        switches = detect_switches(trial)
        assert len(switches) == 0

    def test_primary_added(self, addition_trial):
        """Additional primary in results should be detected."""
        trial = parse_api_response(addition_trial)
        switches = detect_switches(trial)
        added = [s for s in switches if s["category"] == "PRIMARY_ADDED"]
        assert len(added) == 1
        assert "HbA1c" in added[0]["endpoint"] or "hba1c" in added[0]["endpoint"].lower()

    def test_promotion(self, promotion_trial):
        """Secondary promoted to primary should be detected."""
        trial = parse_api_response(promotion_trial)
        switches = detect_switches(trial)
        promotions = [s for s in switches if s["category"] == "PROMOTION"]
        assert len(promotions) == 1
        assert "progression" in promotions[0]["endpoint"].lower() or "pfs" in promotions[0]["endpoint"].lower()

    def test_demotion(self, promotion_trial):
        """Primary demoted to secondary should be detected."""
        trial = parse_api_response(promotion_trial)
        switches = detect_switches(trial)
        demotions = [s for s in switches if s["category"] == "DEMOTION"]
        assert len(demotions) == 1
        assert "overall survival" in demotions[0]["endpoint"].lower() or "os" in demotions[0]["endpoint"].lower()

    def test_primary_removed(self):
        """Primary endpoint removed from results should be detected."""
        trial = {
            "protocolPrimary": [
                {"measure": "Overall Survival", "timeFrame": "36 months"},
                {"measure": "Cardiac mortality", "timeFrame": "36 months"}
            ],
            "protocolSecondary": [
                {"measure": "LVEF change", "timeFrame": "12 months"}
            ],
            "resultsPrimary": [
                {"measure": "Overall Survival", "timeFrame": "36 months"}
            ],
            "resultsSecondary": [
                {"measure": "LVEF change", "timeFrame": "12 months"}
            ],
        }
        switches = detect_switches(trial)
        removed = [s for s in switches if s["category"] == "PRIMARY_REMOVED"]
        assert len(removed) == 1
        assert "cardiac mortality" in removed[0]["endpoint"].lower()

    def test_timeframe_change(self):
        """Timeframe change in primary endpoint should be detected."""
        trial = {
            "protocolPrimary": [
                {"measure": "Change in blood pressure", "timeFrame": "12 weeks"}
            ],
            "protocolSecondary": [],
            "resultsPrimary": [
                {"measure": "Change in blood pressure", "timeFrame": "24 weeks"}
            ],
            "resultsSecondary": [],
        }
        switches = detect_switches(trial)
        tf_changes = [s for s in switches if s["category"] == "TIMEFRAME_CHANGE"]
        assert len(tf_changes) == 1


# ============================================================
# SEVERITY SCORER TESTS (4)
# ============================================================

class TestSeverityScorer:
    def test_high_severity_added(self):
        """PRIMARY_ADDED should be HIGH severity."""
        assert score_severity({"category": "PRIMARY_ADDED"}) == "HIGH"

    def test_medium_severity_timeframe(self):
        """TIMEFRAME_CHANGE should be MEDIUM severity."""
        assert score_severity({"category": "TIMEFRAME_CHANGE"}) == "MEDIUM"

    def test_low_severity_modified(self):
        """MEASURE_MODIFIED should be LOW severity."""
        assert score_severity({"category": "MEASURE_MODIFIED"}) == "LOW"

    def test_high_severity_promotion(self):
        """PROMOTION should be HIGH severity."""
        assert score_severity({"category": "PROMOTION"}) == "HIGH"


# ============================================================
# AGGREGATOR TESTS (3)
# ============================================================

class TestAggregator:
    def _make_analyzed(self, sample_trials):
        """Helper: parse and analyze sample trials."""
        analyzed = []
        for raw in sample_trials:
            trial = parse_api_response(raw)
            switches = detect_switches(trial)
            switches = score_all(switches)
            analyzed.append({"trial": trial, "switches": switches})
        return analyzed

    def test_aggregate_by_sponsor(self, sample_trials):
        """Sponsor rollup should contain all 3 sponsors."""
        analyzed = self._make_analyzed(sample_trials)
        result = aggregate_by_sponsor(analyzed)
        assert "AcmePharma" in result
        assert "BetaCorp" in result
        assert "GammaMed" in result
        # AcmePharma should have 0 switches
        assert result["AcmePharma"]["trials_with_switches"] == 0
        # BetaCorp should have switches
        assert result["BetaCorp"]["trials_with_switches"] == 1

    def test_aggregate_by_condition(self, sample_trials):
        """Condition rollup should have distinct conditions."""
        analyzed = self._make_analyzed(sample_trials)
        result = aggregate_by_condition(analyzed)
        assert "Heart Failure" in result
        assert result["Heart Failure"]["trials_with_switches"] == 0

    def test_aggregate_by_era(self, sample_trials):
        """Era rollup should classify all fixture trials as pre-FDAAA."""
        analyzed = self._make_analyzed(sample_trials)
        result = aggregate_by_era(analyzed)
        # All fixture NCT IDs are < 500000
        assert "pre-FDAAA" in result
        assert result["pre-FDAAA"]["total_trials"] == 3


# ============================================================
# VALIDATOR TESTS (2)
# ============================================================

class TestValidator:
    def test_concordance(self, benchmark_trials):
        """Concordance should be computed correctly against benchmark."""
        concordance, details = validate_against_benchmark(benchmark_trials)
        assert isinstance(concordance, float)
        assert 0.0 <= concordance <= 1.0
        # At least the clean trial (NCT99000001) should be correct
        clean = [d for d in details if d["nctId"] == "NCT99000001"]
        assert len(clean) == 1
        assert clean[0]["correct"] is True

    def test_details_structure(self, benchmark_trials):
        """Details should have expected keys for each trial."""
        _, details = validate_against_benchmark(benchmark_trials)
        assert len(details) == 5
        for d in details:
            assert "nctId" in d
            assert "expected_count" in d
            assert "detected_count" in d
            assert "true_positives" in d
            assert "false_positives" in d
            assert "false_negatives" in d
            assert "correct" in d


# ============================================================
# DASHBOARD TEST (1)
# ============================================================

class TestDashboard:
    def test_data_export_json_serializable(self, sample_trials):
        """Dashboard data should be JSON serializable."""
        from run_analysis import run_pipeline
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            data = run_pipeline(
                output_path=tmp_path
            )
            # Should not raise
            serialized = json.dumps(data)
            assert isinstance(serialized, str)
            assert len(serialized) > 100

            # Verify structure
            assert "summary" in data
            assert "by_sponsor" in data
            assert "by_condition" in data
            assert "by_era" in data
            assert "trials" in data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ============================================================
# INTEGRATION TEST (1)
# ============================================================

class TestIntegration:
    def test_full_pipeline_end_to_end(self, sample_trials):
        """Full pipeline: parse -> detect -> score -> aggregate."""
        # Parse
        trials = [parse_api_response(raw) for raw in sample_trials]
        assert len(trials) == 3

        # Detect and score
        analyzed = []
        for trial in trials:
            switches = detect_switches(trial)
            switches = score_all(switches)
            analyzed.append({"trial": trial, "switches": switches})

        # NCT00000001: clean — 0 switches
        assert len(analyzed[0]["switches"]) == 0

        # NCT00000002: addition — 1 PRIMARY_ADDED
        added = [s for s in analyzed[1]["switches"] if s["category"] == "PRIMARY_ADDED"]
        assert len(added) == 1

        # NCT00000003: promotion + demotion
        promos = [s for s in analyzed[2]["switches"] if s["category"] == "PROMOTION"]
        demos = [s for s in analyzed[2]["switches"] if s["category"] == "DEMOTION"]
        assert len(promos) == 1
        assert len(demos) == 1

        # Aggregate
        by_sponsor = aggregate_by_sponsor(analyzed)
        assert len(by_sponsor) == 3
        by_condition = aggregate_by_condition(analyzed)
        assert len(by_condition) == 3
        by_era = aggregate_by_era(analyzed)
        assert len(by_era) >= 1


# ============================================================
# SMOKE TEST (1)
# ============================================================

class TestSmoke:
    def test_all_modules_importable(self):
        """All modules should import without error."""
        import src.fetcher
        import src.endpoint_parser
        import src.switch_detector
        import src.severity_scorer
        import src.aggregator
        import src.validator
        import run_analysis
        assert True
