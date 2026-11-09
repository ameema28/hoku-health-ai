"""
Structured-log fields for Cloud Run, correlation-id validation and the
access-log skip list. No network.
"""

import json
import logging

import pytest

from app.core.logging import (
    CORRELATION_HEADER,
    JSONFormatter,
    RequestLoggingMiddleware,
    set_correlation_id,
)


def _record(level: int) -> logging.LogRecord:
    return logging.LogRecord("app.test", level, __file__, 1, "hello", (), None)


@pytest.mark.parametrize(
    "level, name",
    [
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.CRITICAL, "CRITICAL"),
    ],
)
def test_json_log_has_cloud_logging_severity(level, name):
    payload = json.loads(JSONFormatter().format(_record(level)))
    assert payload["severity"] == name
    assert payload["level"] == name


def test_valid_inbound_correlation_id_is_kept():
    assert set_correlation_id("abc-123-DEF") == "abc-123-DEF"


@pytest.mark.parametrize(
    "value",
    ["a" * 65, "bad id", "x\nInjected: line", "<script>", "id;drop", ""],
)
def test_invalid_inbound_correlation_id_is_replaced(value):
    resolved = set_correlation_id(value)
    assert resolved != value
    assert len(resolved) == 32 and resolved.isalnum()


def test_health_probes_are_not_access_logged():
    assert "/health" in RequestLoggingMiddleware.SKIP_PATHS
    assert "/api/ai/health" in RequestLoggingMiddleware.SKIP_PATHS


def test_response_echoes_a_sanitised_correlation_id(client):
    resp = client.get("/health", headers={CORRELATION_HEADER: "evil value\twith tab"})
    echoed = resp.headers[CORRELATION_HEADER]
    assert echoed != "evil value\twith tab"
    assert echoed.isalnum()
