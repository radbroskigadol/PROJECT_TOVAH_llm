"""
TOVAH v14 conformance — Fixtures, regression seeds, benchmark runner.

Validates runtime core first, then report layer.
May import from core/ and invariants/, but NOT from kernel/.
"""
from tovah_v14.conformance.fixtures import BASELINE_FIXTURES
from tovah_v14.conformance.regression import run_regression_suite
