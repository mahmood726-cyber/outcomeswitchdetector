"""NLP endpoint normalization + synonym dictionary for clinical trial outcomes."""

import re
import math
from collections import Counter


# ~40 abbreviation expansions for clinical trial endpoints
ABBREVIATIONS = {
    "lvef": "left ventricular ejection fraction",
    "ef": "ejection fraction",
    "os": "overall survival",
    "pfs": "progression-free survival",
    "dfs": "disease-free survival",
    "rfs": "relapse-free survival",
    "efs": "event-free survival",
    "orr": "objective response rate",
    "crr": "complete response rate",
    "dcr": "disease control rate",
    "ttp": "time to progression",
    "dor": "duration of response",
    "ttf": "time to treatment failure",
    "hba1c": "glycated hemoglobin",
    "fpg": "fasting plasma glucose",
    "bmi": "body mass index",
    "bp": "blood pressure",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "hr": "heart rate",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "pet": "positron emission tomography",
    "qol": "quality of life",
    "hrqol": "health-related quality of life",
    "sf-36": "short form 36",
    "eq-5d": "euroqol 5 dimensions",
    "nyha": "new york heart association",
    "nt-probnp": "n-terminal pro-b-type natriuretic peptide",
    "bnp": "b-type natriuretic peptide",
    "crp": "c-reactive protein",
    "ldl": "low-density lipoprotein",
    "hdl": "high-density lipoprotein",
    "gfr": "glomerular filtration rate",
    "egfr": "estimated glomerular filtration rate",
    "ace": "angiotensin-converting enzyme",
    "arb": "angiotensin receptor blocker",
    "mace": "major adverse cardiovascular events",
    "cv": "cardiovascular",
    "mi": "myocardial infarction",
    "acs": "acute coronary syndrome",
    "6mwd": "6-minute walk distance",
    "6mwt": "6-minute walk test",
    "ae": "adverse event",
    "sae": "serious adverse event",
    "teae": "treatment-emergent adverse event",
}

# Timeframe normalization patterns
TIMEFRAME_CONVERSIONS = {
    # weeks to months (approximate)
    r"(\d+)\s*weeks?": lambda m: _weeks_to_months(int(m.group(1))),
    r"(\d+)\s*days?": lambda m: _days_to_months(int(m.group(1))),
}


def _weeks_to_months(weeks):
    """Convert weeks to months string if evenly divisible by ~4.33."""
    months = weeks / 4.33
    rounded = round(months)
    if abs(months - rounded) < 0.3:
        return f"{rounded} months"
    return f"{weeks} weeks"


def _days_to_months(days):
    """Convert days to months string if evenly divisible by ~30.44."""
    months = days / 30.44
    rounded = round(months)
    if abs(months - rounded) < 0.5:
        return f"{rounded} months"
    return f"{days} days"


def normalize_endpoint(text):
    """Normalize a clinical trial endpoint string.

    Steps:
    1. Lowercase
    2. Expand abbreviations (word-boundary aware)
    3. Normalize timeframes
    4. Strip excess whitespace and punctuation

    Parameters
    ----------
    text : str
        Raw endpoint text.

    Returns
    -------
    str : Normalized endpoint text.
    """
    if not text:
        return ""

    result = text.lower().strip()

    # Expand abbreviations (word-boundary aware, case-insensitive already)
    for abbr, expansion in ABBREVIATIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

    # Normalize timeframes
    for pattern, converter in TIMEFRAME_CONVERSIONS.items():
        result = re.sub(pattern, converter, result, flags=re.IGNORECASE)

    # Remove extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def _tokenize(text):
    """Split text into tokens for similarity computation."""
    return re.findall(r'[a-z0-9]+', text.lower())


def compute_similarity(a, b):
    """Compute cosine similarity between two endpoint strings after normalization.

    Parameters
    ----------
    a, b : str
        Endpoint strings to compare.

    Returns
    -------
    float : Cosine similarity in [0, 1].
        >= 0.85 → same endpoint
        0.60-0.85 → modified endpoint
        < 0.60 → different endpoint
    """
    norm_a = normalize_endpoint(a)
    norm_b = normalize_endpoint(b)

    tokens_a = _tokenize(norm_a)
    tokens_b = _tokenize(norm_b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)

    # All unique tokens
    all_tokens = set(counter_a.keys()) | set(counter_b.keys())

    # Dot product
    dot = sum(counter_a.get(t, 0) * counter_b.get(t, 0) for t in all_tokens)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in counter_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)
