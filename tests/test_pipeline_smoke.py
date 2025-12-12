"""Smoke tests for pipeline."""
from pathlib import Path


def test_pipeline_smoke():
    """Test that example quick_test script exists."""
    assert Path("examples/quick_test.py").exists()
