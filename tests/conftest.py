"""Pytest configuration and shared fixtures."""

import sys
import os
import pytest

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from load_candles import load_candles_from_db


@pytest.fixture(scope="session")
def real_candles_small():
    """Load 1000 real candles for quick tests."""
    return load_candles_from_db(limit=1000)


@pytest.fixture(scope="session")
def real_candles_medium():
    """Load 10,000 real candles for medium tests."""
    return load_candles_from_db(limit=10000)


@pytest.fixture(scope="session")
def real_candles_large():
    """Load 100,000 real candles for comprehensive tests."""
    return load_candles_from_db(limit=100000)


@pytest.fixture(scope="session")
def real_candles_prefill():
    """Load enough candles for proper prefill (72,000+ candles for 4H/300 periods)."""
    return load_candles_from_db(limit=80000)
