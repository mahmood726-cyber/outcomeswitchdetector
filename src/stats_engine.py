"""Statistical engine for OutcomeSwitchDetector.

Layer 1 (Publication-Essential): Wilson CI, chi-squared, Fisher exact,
    Cochran-Armitage trend, risk-adjusted rates, Benjamini-Hochberg FDR.
Layer 2 (Methodologically Novel): Bayesian hierarchical model,
    latent class analysis, enhanced TF-IDF similarity.
Layer 3 (Advanced): Meta-analysis of switching rates, interrupted time
    series, mutual information, permutation test, negative binomial
    regression.
"""

import math
import random
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats


# ============================================================
# Layer 1 — Publication-Essential (6 methods)
# ============================================================


def wilson_ci(x, n, alpha=0.05):
    """Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    x : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple : (lower, upper) confidence interval bounds.
    """
    # Edge cases
    if n == 0:
        return (0.0, 0.0)
    if x == 0:
        # Lower bound is exactly 0 for x=0
        p_hat = 0.0
        z = sp_stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
        return (0.0, center + margin)
    if x == n:
        # Upper bound is exactly 1 for x=n
        p_hat = 1.0
        z = sp_stats.norm.ppf(1 - alpha / 2)
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
        return (center - margin, 1.0)

    p_hat = x / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - margin, center + margin)


def chi_squared_test(contingency_table):
    """Chi-squared test for independence with Cramer's V.

    Parameters
    ----------
    contingency_table : list of list
        2D contingency table (rows x columns).

    Returns
    -------
    dict : {chi2, p_value, df, cramers_v}
    """
    table = np.array(contingency_table)
    chi2, p_value, df, _ = sp_stats.chi2_contingency(table)
    n_total = table.sum()
    r, c = table.shape
    min_dim = min(r, c) - 1
    if min_dim == 0 or n_total == 0:
        cramers_v = 0.0
    else:
        cramers_v = math.sqrt(chi2 / (n_total * min_dim))
    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "df": int(df),
        "cramers_v": float(cramers_v),
    }


def fisher_exact_test(table_2x2):
    """Fisher exact test for a 2x2 contingency table.

    Parameters
    ----------
    table_2x2 : list of list
        2x2 contingency table.

    Returns
    -------
    dict : {odds_ratio, p_value}
    """
    odds_ratio, p_value = sp_stats.fisher_exact(table_2x2)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
    }


def cochran_armitage_trend(counts, totals, scores=None):
    """Cochran-Armitage test for trend in proportions.

    Parameters
    ----------
    counts : list of int
        Number of events in each group.
    totals : list of int
        Total observations in each group.
    scores : list of float, optional
        Dose scores for each group. Default: 0, 1, 2, ...

    Returns
    -------
    dict : {z_stat, p_value}
    """
    k = len(counts)
    if scores is None:
        scores = list(range(k))

    counts = [float(c) for c in counts]
    totals = [float(t) for t in totals]
    scores = [float(s) for s in scores]

    n_total = sum(totals)
    if n_total == 0:
        return {"z_stat": 0.0, "p_value": 1.0}

    p_bar = sum(counts) / n_total

    # Statistic T
    T = sum(scores[i] * counts[i] for i in range(k))
    E_T = p_bar * sum(scores[i] * totals[i] for i in range(k))

    # Variance of T
    sum_s2_n = sum(scores[i] ** 2 * totals[i] for i in range(k))
    sum_s_n = sum(scores[i] * totals[i] for i in range(k))
    Var_T = p_bar * (1 - p_bar) * (sum_s2_n - sum_s_n ** 2 / n_total)

    if Var_T <= 0:
        return {"z_stat": 0.0, "p_value": 1.0}

    z = (T - E_T) / math.sqrt(Var_T)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {
        "z_stat": float(z),
        "p_value": float(p_value),
    }


def risk_adjusted_rates(switches, trials):
    """Risk-adjusted switching rates using logistic regression.

    Parameters
    ----------
    switches : list of dict
        Each dict has nctId, sponsor, phase, condition.
    trials : list of dict
        Each dict has nctId, sponsor, phase (list), condition.

    Returns
    -------
    list of dict : [{sponsor, raw_rate, adjusted_rate, odds_ratio,
                     ci_lower, ci_upper, p_value}]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    if not trials:
        return []

    # Build trial-level data
    switch_ncts = set(s["nctId"] for s in switches)

    # Collect sponsors
    sponsors = sorted(set(t["sponsor"] for t in trials))
    if len(sponsors) < 2:
        # Can't do logistic regression with only one sponsor
        sponsor = sponsors[0] if sponsors else "Unknown"
        raw_rate = len(switch_ncts) / len(trials) if trials else 0.0
        return [{
            "sponsor": sponsor,
            "raw_rate": raw_rate,
            "adjusted_rate": raw_rate,
            "odds_ratio": 1.0,
            "ci_lower": 0.0,
            "ci_upper": float('inf'),
            "p_value": 1.0,
        }]

    # Phase encoding
    phase_map = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}

    # Build feature matrix
    y = []
    sponsor_labels = []
    phase_nums = []

    for t in trials:
        nct = t["nctId"]
        y.append(1 if nct in switch_ncts else 0)
        sponsor_labels.append(t["sponsor"])
        # Phase: take max if list
        phases = t.get("phase", [])
        if isinstance(phases, str):
            phases = [phases]
        max_phase = 0
        for p in phases:
            max_phase = max(max_phase, phase_map.get(p, 0))
        phase_nums.append(max_phase)

    y = np.array(y)

    # Encode sponsors as one-hot
    le = LabelEncoder()
    sponsor_encoded = le.fit_transform(sponsor_labels)
    n_sponsors = len(le.classes_)

    # Build X: one-hot sponsors (drop first for identifiability) + phase
    X = np.zeros((len(trials), n_sponsors - 1 + 1))
    for i, s_idx in enumerate(sponsor_encoded):
        if s_idx > 0:
            X[i, s_idx - 1] = 1
        X[i, -1] = phase_nums[i]

    # Check for degenerate cases
    if len(np.unique(y)) < 2:
        # All switched or none switched: return raw rates only
        results = []
        sponsor_counts = defaultdict(lambda: {"total": 0, "switches": 0})
        for i, t in enumerate(trials):
            sponsor_counts[t["sponsor"]]["total"] += 1
            if y[i] == 1:
                sponsor_counts[t["sponsor"]]["switches"] += 1
        for sp in sponsors:
            total = sponsor_counts[sp]["total"]
            sw = sponsor_counts[sp]["switches"]
            raw = sw / total if total > 0 else 0.0
            results.append({
                "sponsor": sp,
                "raw_rate": raw,
                "adjusted_rate": raw,
                "odds_ratio": 1.0,
                "ci_lower": 0.0,
                "ci_upper": float('inf'),
                "p_value": 1.0,
            })
        return results

    # Fit logistic regression
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    # Extract results per sponsor
    results = []
    ref_sponsor = le.classes_[0]  # reference category

    # Raw rates
    sponsor_counts = defaultdict(lambda: {"total": 0, "switches": 0})
    for i, t in enumerate(trials):
        sponsor_counts[t["sponsor"]]["total"] += 1
        if y[i] == 1:
            sponsor_counts[t["sponsor"]]["switches"] += 1

    # Coefficients: first (n_sponsors-1) columns are sponsor dummies
    coefs = model.coef_[0]
    intercept = model.intercept_[0]

    # Compute standard errors via Hessian (Fisher information)
    probs = model.predict_proba(X)[:, 1]
    W = probs * (1 - probs)
    W = np.clip(W, 1e-10, None)  # avoid division by zero
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    H = X_with_intercept.T @ np.diag(W) @ X_with_intercept
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(X_with_intercept.shape[1], float('inf'))

    z_crit = sp_stats.norm.ppf(0.975)

    for idx, sponsor in enumerate(le.classes_):
        total = sponsor_counts[sponsor]["total"]
        sw = sponsor_counts[sponsor]["switches"]
        raw_rate = sw / total if total > 0 else 0.0

        if idx == 0:
            # Reference sponsor: OR=1 by definition
            log_or = 0.0
            se_or = se[0] if len(se) > 0 else float('inf')
            p_val = 1.0
        else:
            coef_idx = idx  # coef index in X_with_intercept (0 = intercept)
            log_or = coefs[idx - 1]
            se_or = se[coef_idx] if coef_idx < len(se) else float('inf')
            z_val = log_or / se_or if se_or > 0 and se_or != float('inf') else 0.0
            p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_val)))

        # Guard against overflow: math.exp caps at ~709
        _MAX_EXP = 700
        try:
            odds_ratio = math.exp(max(-_MAX_EXP, min(_MAX_EXP, log_or)))
        except OverflowError:
            odds_ratio = float('inf') if log_or > 0 else 0.0

        if se_or == float('inf') or (log_or - z_crit * se_or) < -_MAX_EXP:
            ci_lower = 0.0
        else:
            ci_lower = math.exp(log_or - z_crit * se_or)

        if se_or == float('inf') or (log_or + z_crit * se_or) > _MAX_EXP:
            ci_upper = float('inf')
        else:
            ci_upper = math.exp(log_or + z_crit * se_or)

        # Adjusted rate: marginal prediction with sponsor dummy set
        adjusted_rate = raw_rate  # fallback
        if total > 0:
            # Average predicted probability for this sponsor's trials
            mask = np.array([t["sponsor"] == sponsor for t in trials])
            if mask.any():
                adjusted_rate = float(probs[mask].mean())

        results.append({
            "sponsor": sponsor,
            "raw_rate": float(raw_rate),
            "adjusted_rate": float(adjusted_rate),
            "odds_ratio": float(odds_ratio),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "p_value": float(p_val),
        })

    return results


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values.
    alpha : float
        FDR level (default 0.05).

    Returns
    -------
    list of dict : [{original_p, adjusted_p, significant}]
        Ordered to match the input p_values.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Create indexed list and sort by p-value
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    # Compute adjusted p-values (step-up)
    adjusted = [0.0] * m
    for rank, (p, orig_idx) in enumerate(indexed, start=1):
        adjusted_p = p * m / rank
        adjusted[orig_idx] = adjusted_p

    # Enforce monotonicity (from largest rank to smallest)
    # Re-sort by original p-value rank to apply cumulative min from bottom
    sorted_by_p = sorted(range(m), key=lambda i: p_values[i])
    # Apply cumulative minimum from right (largest rank first)
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx = sorted_by_p[rank_idx]
        adjusted[orig_idx] = min(adjusted[orig_idx], prev)
        adjusted[orig_idx] = min(adjusted[orig_idx], 1.0)  # cap at 1
        prev = adjusted[orig_idx]

    # Build results in original order
    results = []
    for i in range(m):
        results.append({
            "original_p": p_values[i],
            "adjusted_p": adjusted[i],
            "significant": adjusted[i] <= alpha,
        })

    return results


# ============================================================
# Layer 2 — Methodologically Novel (3 methods)
# ============================================================


def _log_beta_pdf(x, a, b):
    """Log density of Beta(a, b) at x. Avoids overflow with lgamma."""
    if x <= 0 or x >= 1:
        return -float('inf')
    if a <= 0 or b <= 0:
        return -float('inf')
    return (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
    )


def _log_gamma_pdf(x, shape, rate):
    """Log density of Gamma(shape, rate) at x."""
    if x <= 0 or shape <= 0 or rate <= 0:
        return -float('inf')
    return (
        shape * math.log(rate) - math.lgamma(shape)
        + (shape - 1) * math.log(x) - rate * x
    )


def bayesian_hierarchical_model(sponsor_switches, sponsor_totals,
                                 n_iter=5000, seed=42):
    """Bayesian hierarchical Beta-Binomial model via Gibbs/MH sampler.

    Parameters
    ----------
    sponsor_switches : list of int
        Number of switches per sponsor.
    sponsor_totals : list of int
        Number of trials per sponsor.
    n_iter : int
        Total MCMC iterations (default 5000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict : {posteriors: [{sponsor, mean, ci_lower, ci_upper, shrinkage}],
            alpha, beta, grand_mean}
    """
    random.seed(seed)
    np.random.seed(seed)

    k = len(sponsor_switches)
    x = [int(s) for s in sponsor_switches]
    n = [int(t) for t in sponsor_totals]

    # Initialize theta_j
    theta = []
    for j in range(k):
        if n[j] == 0:
            theta.append(0.5)
        else:
            # Avoid exact 0 or 1 to prevent log(0) issues
            raw = x[j] / n[j]
            theta.append(max(0.01, min(0.99, raw)))

    # Initialize hyperparameters
    alpha_hyper = 2.0
    beta_hyper = 2.0

    burn_in = min(1000, n_iter // 2)

    # Storage for posterior samples
    theta_samples = [[] for _ in range(k)]
    alpha_samples = []
    beta_samples = []

    # MH proposal scale
    proposal_scale = 0.5

    for iteration in range(n_iter):
        # Step a: Sample theta_j | alpha, beta, x, n
        for j in range(k):
            a_post = alpha_hyper + x[j]
            b_post = beta_hyper + n[j] - x[j]
            if a_post > 0 and b_post > 0:
                theta[j] = np.random.beta(a_post, b_post)
                # Clamp to avoid degenerate values
                theta[j] = max(1e-6, min(1 - 1e-6, theta[j]))

        # Step b: Update alpha, beta via MH step
        # Propose new alpha and beta from log-normal random walk
        log_alpha_prop = math.log(alpha_hyper) + random.gauss(0, proposal_scale)
        log_beta_prop = math.log(beta_hyper) + random.gauss(0, proposal_scale)
        alpha_prop = math.exp(log_alpha_prop)
        beta_prop = math.exp(log_beta_prop)

        # Log-likelihood under current vs proposed
        log_lik_current = sum(
            _log_beta_pdf(theta[j], alpha_hyper, beta_hyper) for j in range(k)
        )
        log_lik_proposed = sum(
            _log_beta_pdf(theta[j], alpha_prop, beta_prop) for j in range(k)
        )

        # Gamma(1,1) prior on alpha and beta
        log_prior_current = (
            _log_gamma_pdf(alpha_hyper, 1.0, 1.0)
            + _log_gamma_pdf(beta_hyper, 1.0, 1.0)
        )
        log_prior_proposed = (
            _log_gamma_pdf(alpha_prop, 1.0, 1.0)
            + _log_gamma_pdf(beta_prop, 1.0, 1.0)
        )

        # Jacobian for log-normal proposal (symmetric in log space)
        log_jacobian = (log_alpha_prop - math.log(alpha_hyper)
                        + log_beta_prop - math.log(beta_hyper))

        log_accept = (
            log_lik_proposed + log_prior_proposed
            - log_lik_current - log_prior_current
            + log_jacobian
        )

        if math.log(random.random() + 1e-300) < log_accept:
            alpha_hyper = alpha_prop
            beta_hyper = beta_prop

        # Store post-burn-in samples
        if iteration >= burn_in:
            for j in range(k):
                theta_samples[j].append(theta[j])
            alpha_samples.append(alpha_hyper)
            beta_samples.append(beta_hyper)

    # Compute posterior summaries
    posteriors = []
    for j in range(k):
        samples = theta_samples[j]
        if not samples:
            posteriors.append({
                "sponsor": j,
                "mean": 0.5,
                "ci_lower": 0.0,
                "ci_upper": 1.0,
                "shrinkage": 0.0,
            })
            continue

        post_mean = float(np.mean(samples))
        ci_lower = float(np.percentile(samples, 2.5))
        ci_upper = float(np.percentile(samples, 97.5))
        raw_rate = x[j] / n[j] if n[j] > 0 else 0.5
        shrinkage = (raw_rate - post_mean) / raw_rate if raw_rate != 0 else 0.0

        posteriors.append({
            "sponsor": j,
            "mean": post_mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "shrinkage": float(shrinkage),
        })

    grand_mean = float(np.mean(alpha_samples)) / (
        float(np.mean(alpha_samples)) + float(np.mean(beta_samples))
    ) if alpha_samples and beta_samples else 0.5

    return {
        "posteriors": posteriors,
        "alpha": float(np.mean(alpha_samples)) if alpha_samples else 2.0,
        "beta": float(np.mean(beta_samples)) if beta_samples else 2.0,
        "grand_mean": float(grand_mean),
    }


def latent_class_analysis(switch_profiles, max_k=4, seed=42):
    """Latent class analysis via EM for mixture of Bernoulli.

    Parameters
    ----------
    switch_profiles : list of list
        Binary matrix (trials x 6 switch types).
    max_k : int
        Maximum number of classes to evaluate (default 4).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict : {k_best, classes, bic_values, assignments}
    """
    random.seed(seed)
    np.random.seed(seed)

    X = np.array(switch_profiles, dtype=float)
    n_obs, n_features = X.shape

    if n_obs == 0:
        return {
            "k_best": 1,
            "classes": [],
            "bic_values": [],
            "assignments": [],
        }

    best_bic = float('inf')
    best_result = None
    bic_values = []

    for K in range(1, max_k + 1):
        # Initialize
        pi = np.ones(K) / K  # mixing proportions
        # Random initialization of class-conditional probabilities
        p = np.random.dirichlet(np.ones(K), size=n_features).T  # K x n_features
        # Clamp to avoid log(0)
        p = np.clip(p, 0.01, 0.99)

        # EM iterations
        max_em_iter = 200
        prev_ll = -float('inf')
        tol = 1e-6

        gamma = np.zeros((n_obs, K))

        for em_iter in range(max_em_iter):
            # E-step: compute responsibilities
            for k_idx in range(K):
                log_lik = np.zeros(n_obs)
                for j in range(n_features):
                    log_lik += (
                        X[:, j] * np.log(p[k_idx, j] + 1e-300)
                        + (1 - X[:, j]) * np.log(1 - p[k_idx, j] + 1e-300)
                    )
                gamma[:, k_idx] = np.log(pi[k_idx] + 1e-300) + log_lik

            # Log-sum-exp for normalization
            max_gamma = np.max(gamma, axis=1, keepdims=True)
            gamma_shifted = gamma - max_gamma
            log_sum = max_gamma.flatten() + np.log(np.sum(np.exp(gamma_shifted), axis=1))
            gamma = np.exp(gamma - log_sum[:, np.newaxis])

            # Compute log-likelihood
            ll = np.sum(log_sum)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # M-step
            N_k = gamma.sum(axis=0)  # effective count per class
            for k_idx in range(K):
                if N_k[k_idx] < 1e-10:
                    continue
                pi[k_idx] = N_k[k_idx] / n_obs
                for j in range(n_features):
                    p[k_idx, j] = (gamma[:, k_idx] * X[:, j]).sum() / N_k[k_idx]
                    # Clamp
                    p[k_idx, j] = max(0.01, min(0.99, p[k_idx, j]))

        # Compute BIC
        n_params = K * n_features + (K - 1)  # class probabilities + mixing weights
        bic = -2 * prev_ll + n_params * math.log(n_obs)
        bic_values.append({"k": K, "bic": float(bic)})

        if bic < best_bic:
            best_bic = bic
            assignments = np.argmax(gamma, axis=1).tolist()
            classes = []
            for k_idx in range(K):
                profile = {}
                switch_types = [
                    "PRIMARY_ADDED", "PRIMARY_REMOVED", "PROMOTION",
                    "DEMOTION", "TIMEFRAME_CHANGE", "MEASURE_MODIFIED"
                ]
                for j in range(min(n_features, len(switch_types))):
                    profile[switch_types[j]] = float(p[k_idx, j])
                # Handle extra features beyond 6
                for j in range(len(switch_types), n_features):
                    profile[f"feature_{j}"] = float(p[k_idx, j])
                classes.append({
                    "class_id": k_idx,
                    "proportion": float(pi[k_idx]),
                    "profile": profile,
                })
            best_result = {
                "k_best": K,
                "classes": classes,
                "assignments": assignments,
            }

    best_result["bic_values"] = bic_values
    return best_result


def enhanced_similarity(text_a, text_b):
    """TF-IDF n-gram cosine similarity between two texts.

    Uses sklearn TfidfVectorizer with ngram_range=(1,3).
    Falls back to bag-of-words cosine similarity if sklearn unavailable.

    Parameters
    ----------
    text_a, text_b : str
        Texts to compare.

    Returns
    -------
    float : Cosine similarity in [0, 1].
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
        return float(sim)
    except ImportError:
        # Fallback: bag-of-words cosine similarity
        import re
        from collections import Counter

        tokens_a = re.findall(r'[a-z0-9]+', text_a.lower())
        tokens_b = re.findall(r'[a-z0-9]+', text_b.lower())

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        ca = Counter(tokens_a)
        cb = Counter(tokens_b)
        all_tokens = set(ca.keys()) | set(cb.keys())

        dot = sum(ca.get(t, 0) * cb.get(t, 0) for t in all_tokens)
        mag_a = math.sqrt(sum(v ** 2 for v in ca.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in cb.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ============================================================
# Layer 3 — Advanced Statistical Methods (5 methods)
# ============================================================


def meta_analyze_switching(condition_data, method='DL'):
    """Meta-analysis of switching rates using DerSimonian-Laird random effects.

    Uses Freeman-Tukey double arcsine transformation for proportions.

    Parameters
    ----------
    condition_data : list of dict
        Each dict has {condition, switched, total}.
    method : str
        Pooling method (default 'DL' for DerSimonian-Laird).

    Returns
    -------
    dict : {pooled_rate, ci_lower, ci_upper, tau2, i_squared,
            q_stat, q_pvalue, per_condition}
    """
    k = len(condition_data)
    if k == 0:
        return {
            "pooled_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "tau2": 0.0, "i_squared": 0.0, "q_stat": 0.0,
            "q_pvalue": 1.0, "per_condition": [],
        }

    # Freeman-Tukey double arcsine transformation
    t_vals = []
    var_vals = []
    for d in condition_data:
        x_i = d["switched"]
        n_i = d["total"]
        t_i = math.asin(math.sqrt(x_i / (n_i + 1))) + math.asin(math.sqrt((x_i + 1) / (n_i + 1)))
        v_i = 1.0 / (n_i + 0.5)
        t_vals.append(t_i)
        var_vals.append(v_i)

    t_vals = np.array(t_vals)
    var_vals = np.array(var_vals)

    # Fixed-effect weights
    w_fe = 1.0 / var_vals
    t_hat_fe = np.sum(w_fe * t_vals) / np.sum(w_fe)

    # Cochran's Q statistic
    q_stat = float(np.sum(w_fe * (t_vals - t_hat_fe) ** 2))

    # DerSimonian-Laird tau2
    sum_w = np.sum(w_fe)
    sum_w2 = np.sum(w_fe ** 2)
    c = sum_w - sum_w2 / sum_w
    tau2 = max(0.0, (q_stat - (k - 1)) / c) if c > 0 else 0.0

    # Random-effects weights
    w_re = 1.0 / (var_vals + tau2)
    t_hat_re = np.sum(w_re * t_vals) / np.sum(w_re)

    # SE of pooled estimate
    se_re = math.sqrt(1.0 / np.sum(w_re))

    # Back-transform: pooled_rate = (sin(t_hat_re / 2))^2
    pooled_rate = (math.sin(t_hat_re / 2)) ** 2
    pooled_rate = max(0.0, min(1.0, pooled_rate))

    # CI on transformed scale, then back-transform
    z_crit = sp_stats.norm.ppf(0.975)
    t_lower = t_hat_re - z_crit * se_re
    t_upper = t_hat_re + z_crit * se_re
    ci_lower = max(0.0, (math.sin(t_lower / 2)) ** 2)
    ci_upper = min(1.0, (math.sin(t_upper / 2)) ** 2)

    # I-squared
    if q_stat > 0 and k > 1:
        i_squared = max(0.0, (q_stat - (k - 1)) / q_stat * 100)
    else:
        i_squared = 0.0

    # Q p-value
    q_pvalue = float(1.0 - sp_stats.chi2.cdf(q_stat, k - 1)) if k > 1 else 1.0

    # Per-condition results
    per_condition = []
    for i, d in enumerate(condition_data):
        x_i = d["switched"]
        n_i = d["total"]
        rate_i = x_i / n_i if n_i > 0 else 0.0
        w_i = float(w_re[i])
        # Wilson CI for individual condition
        ci_lo, ci_hi = wilson_ci(x_i, n_i)
        per_condition.append({
            "condition": d["condition"],
            "rate": float(rate_i),
            "weight": w_i,
            "ci": (float(ci_lo), float(ci_hi)),
        })

    return {
        "pooled_rate": float(pooled_rate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "tau2": float(tau2),
        "i_squared": float(i_squared),
        "q_stat": float(q_stat),
        "q_pvalue": float(q_pvalue),
        "per_condition": per_condition,
    }


def interrupted_time_series(year_data, intervention_year=2007):
    """Interrupted time series (segmented regression) analysis.

    Fits: rate = beta0 + beta1*time + beta2*post + beta3*time_after + epsilon

    Parameters
    ----------
    year_data : list of dict
        Each dict has {year, rate, n}.
    intervention_year : int
        Year of intervention (default 2007).

    Returns
    -------
    dict : {pre_slope, post_slope, level_change, slope_change,
            p_level, p_slope, r_squared}
    """
    if len(year_data) < 4:
        return {
            "pre_slope": 0.0, "post_slope": 0.0,
            "level_change": 0.0, "slope_change": 0.0,
            "p_level": 1.0, "p_slope": 1.0, "r_squared": 0.0,
        }

    # Sort by year
    year_data = sorted(year_data, key=lambda d: d["year"])
    min_year = year_data[0]["year"]

    n_obs = len(year_data)
    y = np.array([d["rate"] for d in year_data])

    # Build design matrix: intercept, time, post, time_after
    X = np.zeros((n_obs, 4))
    for i, d in enumerate(year_data):
        yr = d["year"]
        time_val = yr - min_year
        post = 1.0 if yr >= intervention_year else 0.0
        time_after = max(0.0, float(yr - intervention_year))
        X[i, 0] = 1.0       # intercept
        X[i, 1] = time_val  # time
        X[i, 2] = post      # post indicator
        X[i, 3] = time_after # time after intervention

    # OLS via numpy least squares
    result_lstsq = np.linalg.lstsq(X, y, rcond=None)
    beta = result_lstsq[0]

    # Predictions and residuals
    y_hat = X @ beta
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors and p-values
    df_residual = n_obs - 4
    if df_residual > 0:
        mse = ss_res / df_residual
        try:
            cov_matrix = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.abs(np.diag(cov_matrix)))
        except np.linalg.LinAlgError:
            se = np.full(4, float('inf'))
    else:
        se = np.full(4, float('inf'))

    # t-statistics and p-values
    def _t_pval(coef, se_val, df):
        if se_val <= 0 or se_val == float('inf') or df <= 0:
            return 1.0
        t_stat = coef / se_val
        return float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df)))

    p_level = _t_pval(beta[2], se[2], df_residual)
    p_slope = _t_pval(beta[3], se[3], df_residual)

    pre_slope = float(beta[1])
    post_slope = float(beta[1] + beta[3])
    level_change = float(beta[2])
    slope_change = float(beta[3])

    return {
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "level_change": level_change,
        "slope_change": slope_change,
        "p_level": p_level,
        "p_slope": p_slope,
        "r_squared": r_squared,
    }


def mutual_information(switch_matrix):
    """Mutual information between pairs of switch types.

    Parameters
    ----------
    switch_matrix : array-like
        Binary matrix (trials x switch_types).

    Returns
    -------
    dict : {mi_matrix, normalized_mi_matrix, top_associations}
    """
    X = np.array(switch_matrix, dtype=float)
    n_obs, n_types = X.shape

    mi_matrix = np.zeros((n_types, n_types))
    nmi_matrix = np.zeros((n_types, n_types))

    def _entropy(col):
        """Shannon entropy of a binary column."""
        p1 = np.mean(col)
        p0 = 1.0 - p1
        h = 0.0
        if p0 > 0:
            h -= p0 * math.log(p0)
        if p1 > 0:
            h -= p1 * math.log(p1)
        return h

    def _mi(col_a, col_b):
        """Mutual information between two binary columns."""
        mi = 0.0
        for x_val in [0, 1]:
            for y_val in [0, 1]:
                p_xy = np.mean((col_a == x_val) & (col_b == y_val))
                p_x = np.mean(col_a == x_val)
                p_y = np.mean(col_b == y_val)
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * math.log(p_xy / (p_x * p_y))
        return mi

    # Compute entropies
    entropies = [_entropy(X[:, j]) for j in range(n_types)]

    # Compute MI and NMI for all pairs
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                mi_matrix[i][j] = entropies[i]
                nmi_matrix[i][j] = 1.0 if entropies[i] > 0 else 0.0
            else:
                mi_val = _mi(X[:, i], X[:, j])
                mi_matrix[i][j] = mi_val
                denom = math.sqrt(entropies[i] * entropies[j])
                nmi_matrix[i][j] = mi_val / denom if denom > 0 else 0.0

    # Top associations (off-diagonal, upper triangle)
    top_associations = []
    for i in range(n_types):
        for j in range(i + 1, n_types):
            top_associations.append({
                "type1": i,
                "type2": j,
                "nmi": float(nmi_matrix[i][j]),
            })
    top_associations.sort(key=lambda a: a["nmi"], reverse=True)

    return {
        "mi_matrix": mi_matrix.tolist(),
        "normalized_mi_matrix": nmi_matrix.tolist(),
        "top_associations": top_associations,
    }


def permutation_test(group1_switches, group1_total, group2_switches,
                     group2_total, n_perms=10000, seed=42):
    """Exact permutation test for difference in switching rates.

    Parameters
    ----------
    group1_switches, group1_total : int
        Switches and total for group 1.
    group2_switches, group2_total : int
        Switches and total for group 2.
    n_perms : int
        Number of permutations (default 10000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict : {observed_diff, p_value, ci_lower, ci_upper}
    """
    rng = np.random.RandomState(seed)

    n1 = int(group1_total)
    n2 = int(group2_total)
    s1 = int(group1_switches)
    s2 = int(group2_switches)

    rate1 = s1 / n1 if n1 > 0 else 0.0
    rate2 = s2 / n2 if n2 > 0 else 0.0
    observed_diff = rate1 - rate2

    # Create pooled indicator array
    pooled = np.zeros(n1 + n2)
    pooled[:s1 + s2] = 1.0

    perm_diffs = np.empty(n_perms)
    for i in range(n_perms):
        rng.shuffle(pooled)
        perm_rate1 = pooled[:n1].mean()
        perm_rate2 = pooled[n1:].mean()
        perm_diffs[i] = perm_rate1 - perm_rate2

    # Two-sided p-value
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

    # 95% CI from permutation distribution
    ci_lower = float(np.percentile(perm_diffs, 2.5))
    ci_upper = float(np.percentile(perm_diffs, 97.5))

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def negative_binomial_regression(trials_data):
    """Negative binomial regression for switch counts per trial.

    Fits: log(E[switches]) = beta0 + beta1*sponsor_industry
          + beta2*phase_num + beta3*log_enrollment

    Uses scipy.optimize.minimize on NB log-likelihood.

    Parameters
    ----------
    trials_data : list of dict
        Each dict has {switches, sponsor_industry, phase_num, log_enrollment}.

    Returns
    -------
    dict : {coefficients, deviance, aic}
    """
    from scipy.optimize import minimize

    n = len(trials_data)
    if n == 0:
        return {"coefficients": [], "deviance": 0.0, "aic": 0.0}

    # Build arrays
    y = np.array([d["switches"] for d in trials_data], dtype=float)
    X = np.column_stack([
        np.ones(n),
        np.array([d["sponsor_industry"] for d in trials_data], dtype=float),
        np.array([d["phase_num"] for d in trials_data], dtype=float),
        np.array([d["log_enrollment"] for d in trials_data], dtype=float),
    ])

    feature_names = ["intercept", "sponsor_industry", "phase_num", "log_enrollment"]
    n_features = X.shape[1]

    def _nb_log_likelihood(params):
        """Negative of NB log-likelihood (for minimization)."""
        beta = params[:n_features]
        log_alpha = params[n_features]  # log of overdispersion
        alpha = math.exp(max(-20, min(20, log_alpha)))

        mu = np.exp(np.clip(X @ beta, -20, 20))
        r = 1.0 / alpha if alpha > 1e-10 else 1e10  # r = 1/alpha

        # NB log-likelihood: sum of log(NB(y_i; r, p_i))
        # where p_i = r / (r + mu_i)
        ll = 0.0
        for i in range(n):
            yi = y[i]
            mu_i = mu[i]
            ri = r
            # Use lgamma for numerical stability
            ll += (
                math.lgamma(yi + ri) - math.lgamma(yi + 1) - math.lgamma(ri)
                + ri * math.log(ri / (ri + mu_i))
                + yi * math.log(mu_i / (ri + mu_i))
            )

        return -ll  # minimize negative LL

    # Initial values: Poisson MLE approximation
    init_beta = np.zeros(n_features)
    mean_y = np.mean(y)
    if mean_y > 0:
        init_beta[0] = math.log(mean_y)
    init_params = np.concatenate([init_beta, [0.0]])  # log_alpha = 0 -> alpha = 1

    try:
        opt_result = minimize(
            _nb_log_likelihood, init_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8},
        )
        params_hat = opt_result.x
        neg_ll = opt_result.fun
    except Exception:
        # Fallback: return Poisson-like results
        params_hat = init_params
        neg_ll = _nb_log_likelihood(init_params)

    beta_hat = params_hat[:n_features]
    log_alpha_hat = params_hat[n_features]

    # Standard errors via finite-difference Hessian
    eps = 1e-5
    n_params = len(params_hat)
    hessian = np.zeros((n_params, n_params))
    f0 = _nb_log_likelihood(params_hat)
    for i_p in range(n_params):
        for j_p in range(i_p, n_params):
            params_pp = params_hat.copy()
            params_pm = params_hat.copy()
            params_mp = params_hat.copy()
            params_mm = params_hat.copy()
            params_pp[i_p] += eps
            params_pp[j_p] += eps
            params_pm[i_p] += eps
            params_pm[j_p] -= eps
            params_mp[i_p] -= eps
            params_mp[j_p] += eps
            params_mm[i_p] -= eps
            params_mm[j_p] -= eps
            h = (
                _nb_log_likelihood(params_pp)
                - _nb_log_likelihood(params_pm)
                - _nb_log_likelihood(params_mp)
                + _nb_log_likelihood(params_mm)
            ) / (4 * eps * eps)
            hessian[i_p, j_p] = h
            hessian[j_p, i_p] = h

    try:
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(n_params, float('inf'))

    # Build coefficient table (skip intercept in output, but include it)
    z_crit = sp_stats.norm.ppf(0.975)
    coefficients = []
    for i_c in range(n_features):
        coef = float(beta_hat[i_c])
        se_i = float(se[i_c]) if i_c < len(se) else float('inf')
        z_val = coef / se_i if se_i > 0 and se_i != float('inf') else 0.0
        p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z_val))))
        irr = math.exp(max(-700, min(700, coef)))
        if se_i != float('inf'):
            irr_lo = math.exp(max(-700, min(700, coef - z_crit * se_i)))
            irr_hi = math.exp(max(-700, min(700, coef + z_crit * se_i)))
        else:
            irr_lo = 0.0
            irr_hi = float('inf')
        coefficients.append({
            "feature": feature_names[i_c],
            "coef": coef,
            "se": se_i,
            "z": z_val,
            "p_value": p_val,
            "irr": irr,
            "irr_ci": (irr_lo, irr_hi),
        })

    # Deviance and AIC
    # Saturated model log-lik approximation
    mu_hat = np.exp(np.clip(X @ beta_hat, -20, 20))
    deviance = float(2 * neg_ll)  # simplified deviance
    aic = float(2 * neg_ll + 2 * (n_features + 1))

    return {
        "coefficients": coefficients,
        "deviance": deviance,
        "aic": aic,
    }
