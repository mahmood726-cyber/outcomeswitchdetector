"""Statistical engine for OutcomeSwitchDetector.

Layer 1 (Publication-Essential): Wilson CI, chi-squared, Fisher exact,
    Cochran-Armitage trend, risk-adjusted rates, Benjamini-Hochberg FDR.
Layer 2 (Methodologically Novel): Bayesian hierarchical model,
    latent class analysis, enhanced TF-IDF similarity.
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
