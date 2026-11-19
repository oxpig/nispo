"""Shared validation for opt-out-capable naming deadlines."""
import math
from numbers import Real

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_REQUESTS_PER_WORKER = 100000


def normalize_timeout(value):
    if value is None:
        return None
    try:
        valid = (not isinstance(value, bool) and isinstance(value, Real)
                 and math.isfinite(value) and value > 0)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError('timeout_seconds must be a positive finite number or None')
    return float(value)


def validate_fallback(value):
    if not isinstance(value, bool):
        raise ValueError('fallback_to_fast must be a boolean')
    return value


def normalize_worker_request_limit(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError('max_requests_per_worker must be a positive integer or None')
    return value
