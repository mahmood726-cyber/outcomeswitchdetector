# OutcomeSwitchDetector — current-record outcome discordance (research prototype)

## Critical interpretation warning

This project compares the **current ClinicalTrials.gov protocol outcome fields** with the **currently posted results outcome fields**. These are not necessarily the originally registered prespecified endpoints. **A difference is a candidate registry-record discordance, not proof of outcome switching, selective reporting, misconduct, or FDAAA noncompliance.** Confirm genuine changes using dated archived protocol versions and an independent clinical reviewer. Similarity thresholds and "HIGH" priority labels are screening heuristics, not validated clinical severity scores.

### Corrections (15 October 2026)

- Fixed ClinicalTrials.gov v2 results filtering. The previous query used an unsupported `filter.resultsReported=true`. The official results filter is `aggFilters=results:with`; it is combined with `filter.overallStatus=COMPLETED`. API responses must match both constraints, with nonduplicate NCT IDs and valid pagination.
- Submission-era comparisons now use actual `statusModule.studyFirstSubmitDate`, never the NCT ID serial number. Trial registration before or after 2007 **does not establish FDAAA 801 applicability**.
- The endpoint matcher avoids reusing the same reported result for multiple registered endpoints. Each candidate finding carries `verification_status=UNVERIFIED_CURRENT_RECORD_DISCORDANCE` and a requirement for archived protocol verification.
- Studies with posted results but no reportable primary-outcome measurements are `unassessable` for primary switching comparison, not automatically counted as having removed primary outcomes. The output reports exclusions.
- GitHub CI includes both deterministic regression tests and a live 3-trial API filter contract test.

### Running

```bash
python -m pip install pytest requests numpy scipy
python -m pytest tests/test_all.py tests/test_stats.py tests/test_registry_contract.py -q
python -m pytest tests/test_live_contract.py -q  # requires access to clinicaltrials.gov
python run_analysis.py  # uses bundled sample fixtures, NOT a full-registry study
```

`fetch_completed_with_results(max_pages, page_size)` can retrieve real posted-results records for exploratory use, but this does not independently establish ground truth. Do not generalize proportions from convenience pages or synthetic fixtures.

**Published static dashboards are historical, nonvalidated prototypes.** Their numerical results have not been regenerated or scientifically adjudicated. Use the code and synthetic tests to develop a prospective protocol-history and clinical-adjudication benchmark.
