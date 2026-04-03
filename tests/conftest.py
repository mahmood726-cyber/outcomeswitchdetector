"""Shared fixtures for OutcomeSwitchDetector tests."""

import json
import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")


@pytest.fixture
def sample_trials():
    """Load sample_trials.json fixture."""
    path = os.path.join(FIXTURES_DIR, "sample_trials.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def benchmark_trials():
    """Load compare_benchmark.json fixture."""
    path = os.path.join(FIXTURES_DIR, "compare_benchmark.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def clean_trial():
    """NCT00000001 — no switching."""
    path = os.path.join(FIXTURES_DIR, "sample_trials.json")
    with open(path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    return trials[0]


@pytest.fixture
def addition_trial():
    """NCT00000002 — outcome addition."""
    path = os.path.join(FIXTURES_DIR, "sample_trials.json")
    with open(path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    return trials[1]


@pytest.fixture
def promotion_trial():
    """NCT00000003 — promotion/demotion."""
    path = os.path.join(FIXTURES_DIR, "sample_trials.json")
    with open(path, "r", encoding="utf-8") as f:
        trials = json.load(f)
    return trials[2]
