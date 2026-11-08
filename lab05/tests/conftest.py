"""Shared pytest fixtures for late deliveries tests."""

import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def sample_data():
    """Return a small sample orders DataFrame for testing."""
    return make_orders(n=200, seed=42)
