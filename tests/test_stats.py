"""33 tests for stats_engine module."""

import os
import sys
import math
import random

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stats_engine import (
    wilson_ci,
    chi_squared_test,
    fisher_exact_test,
    cochran_armitage_trend,
    risk_adjusted_rates,
    benjamini_hochberg,
    bayesian_hierarchical_model,
    latent_class_analysis,
    enhanced_similarity,
    meta_analyze_switching,
    interrupted_time_series,
    mutual_information,
    permutation_test,
    negative_binomial_regression,
    transfer_entropy,
    mdl_model_selection,
    difference_in_differences,
    wasserstein_switching,
    extreme_value_analysis,
)


# ============================================================
# Wilson CI tests (3)
# ============================================================


class TestWilsonCI:
    def test_wilson_ci_basic(self):
        """50/100 should give CI roughly (0.40, 0.60)."""
        lo, hi = wilson_ci(50, 100)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_wilson_ci_zero(self):
        """0/100 should have lower=0 and upper > 0."""
        lo, hi = wilson_ci(0, 100)
        assert lo == 0
        assert hi > 0

    def test_wilson_ci_small_n(self):
        """1/3 with small sample — wider interval expected."""
        lo, hi = wilson_ci(1, 3)
        assert 0 < lo < 0.5
        assert 0.5 < hi < 1.0


# ============================================================
# Chi-squared tests (2)
# ============================================================


class TestChiSquared:
    def test_chi_squared_independent(self):
        """No association should yield p > 0.05."""
        table = [[50, 50], [50, 50]]
        result = chi_squared_test(table)
        assert result["p_value"] > 0.05
        assert abs(result["cramers_v"]) < 0.1

    def test_chi_squared_associated(self):
        """Strong association should yield p < 0.001."""
        table = [[90, 10], [10, 90]]
        result = chi_squared_test(table)
        assert result["p_value"] < 0.001


# ============================================================
# Fisher exact test (1)
# ============================================================


class TestFisherExact:
    def test_fisher_exact(self):
        """Strong 2x2 association should have p < 0.01 and OR > 5."""
        result = fisher_exact_test([[10, 2], [1, 10]])
        assert result["p_value"] < 0.01
        assert result["odds_ratio"] > 5


# ============================================================
# Cochran-Armitage tests (2)
# ============================================================


class TestCochranArmitage:
    def test_trend_increasing(self):
        """Clear increasing trend should be significant."""
        counts = [5, 15, 30, 50]
        totals = [100, 100, 100, 100]
        result = cochran_armitage_trend(counts, totals)
        assert result["p_value"] < 0.001

    def test_trend_none(self):
        """No trend (equal proportions) should be non-significant."""
        counts = [25, 25, 25, 25]
        totals = [100, 100, 100, 100]
        result = cochran_armitage_trend(counts, totals)
        assert result["p_value"] > 0.1


# ============================================================
# Risk-adjusted rates (1)
# ============================================================


class TestRiskAdjusted:
    def test_risk_adjusted(self):
        """Structure check with synthetic data."""
        switches = [
            {"nctId": f"T{i}", "sponsor": "A" if i < 5 else "B",
             "phase": "PHASE3", "condition": "HF"}
            for i in range(10)
        ]
        trials = [
            {"nctId": f"T{i}", "sponsor": "A" if i < 5 else "B",
             "phase": ["PHASE3"], "condition": "HF"}
            for i in range(20)
        ]
        result = risk_adjusted_rates(switches, trials)
        assert len(result) > 0
        assert all("odds_ratio" in r for r in result)


# ============================================================
# Benjamini-Hochberg (1)
# ============================================================


class TestBenjaminiHochberg:
    def test_bh_correction(self):
        """Smallest p-value should remain significant after correction."""
        p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
        result = benjamini_hochberg(p_values)
        assert len(result) == 5
        # Smallest p should still be significant
        sorted_by_p = sorted(result, key=lambda r: r["original_p"])
        assert sorted_by_p[0]["significant"] is True


# ============================================================
# Bayesian hierarchical (2)
# ============================================================


class TestBayesianHierarchical:
    def test_bayesian_shrinkage(self):
        """Sponsor with 1/1 (100%) should shrink toward grand mean."""
        switches = [10, 1, 8, 1]
        totals = [20, 1, 15, 2]
        result = bayesian_hierarchical_model(
            switches, totals, n_iter=2000, seed=42
        )
        # Sponsor 1 (1/1=100%) should have posterior mean < 1.0 (shrinkage)
        assert result["posteriors"][1]["mean"] < 0.95
        assert result["grand_mean"] > 0

    def test_bayesian_credible_intervals(self):
        """Credible intervals should contain the posterior mean."""
        switches = [10, 5, 8]
        totals = [20, 20, 20]
        result = bayesian_hierarchical_model(
            switches, totals, n_iter=2000, seed=42
        )
        for p in result["posteriors"]:
            assert p["ci_lower"] < p["mean"] < p["ci_upper"]


# ============================================================
# Latent class analysis (2)
# ============================================================


class TestLatentClass:
    def test_lca_convergence(self):
        """Two distinct profiles should yield k_best=2."""
        random.seed(42)
        # 20 trials, 6 switch types, 2 latent classes
        profiles = []
        for i in range(20):
            if i < 10:
                profiles.append([1, 1, 0, 0, 0, 0])  # class A
            else:
                profiles.append([0, 0, 0, 0, 1, 1])  # class B
        result = latent_class_analysis(profiles, max_k=3, seed=42)
        assert result["k_best"] == 2
        assert len(result["classes"]) == 2

    def test_lca_bic_values(self):
        """BIC values should be computed for each K."""
        profiles = [[1, 0, 0, 0, 0, 0]] * 10 + [[0, 1, 0, 0, 0, 0]] * 10
        result = latent_class_analysis(profiles, max_k=3, seed=42)
        assert len(result["bic_values"]) >= 2


# ============================================================
# Integration (1)
# ============================================================


class TestIntegration:
    def test_full_stats_pipeline(self):
        """Wilson CI + chi-squared + BH all work together."""
        lo, hi = wilson_ci(30, 100)
        assert 0.2 < lo < 0.3 < hi < 0.4
        chi = chi_squared_test([[30, 70], [20, 80]])
        assert "p_value" in chi
        bh = benjamini_hochberg([chi["p_value"], 0.5])
        assert len(bh) == 2


# ============================================================
# Layer 3 — Meta-Analysis of Switching Rates (2)
# ============================================================


class TestMetaAnalysis:
    def test_meta_analysis_pooling(self):
        """Pooled rate should be between 0 and 1 with valid CI."""
        data = [
            {"condition": "HF", "switched": 30, "total": 100},
            {"condition": "DM", "switched": 20, "total": 80},
            {"condition": "Onc", "switched": 40, "total": 120},
        ]
        result = meta_analyze_switching(data)
        assert 0 < result["pooled_rate"] < 1
        assert result["ci_lower"] < result["pooled_rate"] < result["ci_upper"]
        assert 0 <= result["i_squared"] <= 100

    def test_meta_analysis_heterogeneity(self):
        """Highly heterogeneous data should yield I-squared > 50."""
        data = [
            {"condition": "A", "switched": 5, "total": 100},
            {"condition": "B", "switched": 80, "total": 100},
        ]
        result = meta_analyze_switching(data)
        assert result["i_squared"] > 50


# ============================================================
# Interrupted Time Series (2)
# ============================================================


class TestITS:
    def test_its_level_change(self):
        """Clear level change at intervention should be detected."""
        year_data = [
            {"year": y, "rate": 0.3 if y < 2007 else 0.5, "n": 100}
            for y in range(2000, 2015)
        ]
        result = interrupted_time_series(year_data)
        assert result["level_change"] > 0
        assert result["p_level"] < 0.05

    def test_its_no_change(self):
        """Constant rate should yield non-significant level change."""
        year_data = [
            {"year": y, "rate": 0.3, "n": 100}
            for y in range(2000, 2015)
        ]
        result = interrupted_time_series(year_data)
        assert result["p_level"] > 0.1


# ============================================================
# Mutual Information (2)
# ============================================================


class TestMutualInformation:
    def test_mutual_information_independent(self):
        """Independent random columns should have NMI close to 0."""
        import numpy as np
        np.random.seed(42)
        matrix = np.random.binomial(1, 0.5, (100, 6))
        result = mutual_information(matrix)
        assert all(
            result["normalized_mi_matrix"][i][j] < 0.2
            for i in range(6) for j in range(6) if i != j
        )

    def test_mutual_information_correlated(self):
        """Perfectly correlated columns should have high NMI."""
        col = [[1, 1, 0, 0, 0, 0]] * 50 + [[0, 0, 1, 1, 0, 0]] * 50
        result = mutual_information(col)
        assert result["normalized_mi_matrix"][0][1] > 0.5


# ============================================================
# Permutation Test (2)
# ============================================================


class TestPermutationTest:
    def test_permutation_test_significant(self):
        """Large difference in rates should be significant."""
        result = permutation_test(80, 100, 20, 100, n_perms=5000, seed=42)
        assert result["p_value"] < 0.001

    def test_permutation_test_nonsignificant(self):
        """Similar rates should be non-significant."""
        result = permutation_test(50, 100, 48, 100, n_perms=5000, seed=42)
        assert result["p_value"] > 0.1


# ============================================================
# Negative Binomial Regression (2)
# ============================================================


class TestNegativeBinomial:
    def test_nb_regression_coefficients(self):
        """Coefficient structure should include IRR for all features."""
        trials_data = [
            {"switches": s, "sponsor_industry": 1, "phase_num": 3,
             "log_enrollment": 6}
            for s in [0, 1, 2, 3, 2, 1]
        ] + [
            {"switches": s, "sponsor_industry": 0, "phase_num": 2,
             "log_enrollment": 4}
            for s in [0, 0, 1, 0, 1, 0]
        ]
        result = negative_binomial_regression(trials_data)
        assert len(result["coefficients"]) >= 3
        assert all("irr" in c for c in result["coefficients"])

    def test_nb_regression_irr_positive(self):
        """All IRR values should be positive."""
        trials_data = [
            {"switches": i % 4, "sponsor_industry": i % 2,
             "phase_num": 2 + i % 3, "log_enrollment": 5 + i * 0.1}
            for i in range(20)
        ]
        result = negative_binomial_regression(trials_data)
        assert all(c["irr"] > 0 for c in result["coefficients"])


# ============================================================
# Layer 4 — Transfer Entropy (2)
# ============================================================


class TestTransferEntropy:
    def test_te_independent(self):
        """Independent switch types should have TE near zero."""
        import numpy as np
        rng = np.random.RandomState(123)
        # 10 sequences of length 50 with 6 independent binary switch types
        seqs = [rng.randint(0, 2, size=(50, 6)) for _ in range(10)]
        result = transfer_entropy(seqs, lag=1, seed=42)
        # TE matrix should be 6x6
        assert len(result["te_matrix"]) == 6
        assert len(result["te_matrix"][0]) == 6
        # All TEs should be near zero for independent data
        for i in range(6):
            for j in range(6):
                if i != j:
                    assert abs(result["te_matrix"][i][j]) < 0.15

    def test_te_causal(self):
        """Causal link X0->X1 should show TE[0->1] > TE[1->0]."""
        import numpy as np
        rng = np.random.RandomState(99)
        seqs = []
        for _ in range(20):
            seq = np.zeros((60, 6), dtype=int)
            # X0 is random
            seq[:, 0] = rng.randint(0, 2, 60)
            # X1 copies X0 with lag 1 (causal link)
            seq[1:, 1] = seq[:-1, 0]
            # Other columns random
            for c in range(2, 6):
                seq[:, c] = rng.randint(0, 2, 60)
            seqs.append(seq)
        result = transfer_entropy(seqs, lag=1, seed=42)
        # TE(0->1) should be larger than TE(1->0)
        te_0_to_1 = result["te_matrix"][0][1]
        te_1_to_0 = result["te_matrix"][1][0]
        assert te_0_to_1 > te_1_to_0


# ============================================================
# Layer 4 — MDL Model Selection (1)
# ============================================================


class TestMDL:
    def test_mdl_bimodal(self):
        """Bimodal data should select k >= 2."""
        import numpy as np
        rng = np.random.RandomState(42)
        # Cluster 1: high on types 0-2
        c1 = np.column_stack([
            rng.binomial(1, 0.9, (40, 3)),
            rng.binomial(1, 0.1, (40, 3)),
        ])
        # Cluster 2: high on types 3-5
        c2 = np.column_stack([
            rng.binomial(1, 0.1, (40, 3)),
            rng.binomial(1, 0.9, (40, 3)),
        ])
        data = np.vstack([c1, c2]).tolist()
        result = mdl_model_selection(data, max_components=6)
        assert result["best_k"] >= 2
        assert len(result["mdl_scores"]) >= 2


# ============================================================
# Layer 4 — Difference-in-Differences (2)
# ============================================================


class TestDiD:
    def test_did_significant_effect(self):
        """Clear policy effect should be significant."""
        import numpy as np
        rng = np.random.RandomState(42)
        # Treatment: flat around 0.30 before 2007, jumps to ~0.50 after
        treatment = [
            {"year": y,
             "rate": (0.30 + rng.normal(0, 0.02) if y < 2007
                      else 0.50 + rng.normal(0, 0.02)),
             "n": 100}
            for y in range(2000, 2015)
        ]
        # Control: stays flat at ~0.30 throughout
        control = [
            {"year": y, "rate": 0.30 + rng.normal(0, 0.02), "n": 100}
            for y in range(2000, 2015)
        ]
        result = difference_in_differences(treatment, control, 2007)
        assert result["p_value"] < 0.05
        assert result["att"] > 0  # treatment effect positive

    def test_did_parallel_pretrends(self):
        """Parallel pre-trends should have pre_trend_p > 0.05."""
        # Both groups have same pre-trend
        treatment = [
            {"year": y, "rate": 0.3 + 0.01 * (y - 2000), "n": 100}
            for y in range(2000, 2007)
        ] + [
            {"year": y, "rate": 0.6 + 0.01 * (y - 2000), "n": 100}
            for y in range(2007, 2015)
        ]
        control = [
            {"year": y, "rate": 0.3 + 0.01 * (y - 2000), "n": 100}
            for y in range(2000, 2015)
        ]
        result = difference_in_differences(treatment, control, 2007)
        assert result["pre_trend_p"] > 0.05


# ============================================================
# Layer 4 — Wasserstein Distance (2)
# ============================================================


class TestWasserstein:
    def test_wasserstein_identical(self):
        """Identical profiles should have distance 0."""
        profiles = [
            {"sponsor": "A", "profile": [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]},
            {"sponsor": "B", "profile": [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]},
        ]
        result = wasserstein_switching(profiles)
        assert abs(result["distance_matrix"][0][1]) < 1e-10

    def test_wasserstein_different(self):
        """Very different profiles should have distance > 0.5."""
        profiles = [
            {"sponsor": "X", "profile": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"sponsor": "Y", "profile": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
        ]
        result = wasserstein_switching(profiles)
        assert result["distance_matrix"][0][1] > 0.5
        # Clusters and embedding should exist
        assert len(result["clusters"]) == 2
        assert len(result["mds_embedding"]) == 2


# ============================================================
# Layer 4 — Extreme Value Analysis / GPD (1)
# ============================================================


class TestGPD:
    def test_gpd_heavy_tailed(self):
        """GPD fit on heavy-tailed data should produce return levels."""
        import numpy as np
        rng = np.random.RandomState(42)
        # Pareto-like data: most small, some very large
        data = np.concatenate([
            rng.exponential(2, 200),
            rng.exponential(20, 20),  # tail events
        ])
        result = extreme_value_analysis(data, threshold_quantile=0.90)
        assert result["n_exceedances"] > 0
        assert result["threshold"] > 0
        assert len(result["return_levels"]) == 3
        # Return levels should increase with period
        levels = [rl["level"] for rl in result["return_levels"]]
        assert levels[0] <= levels[1] <= levels[2]
