"""app/scripts/deploy_check.py: every check, the runner, the report and the CLI exit codes.

DB checks run against a temporary SQLite engine; the Groq checks never reach
the network (client construction is offline, the live call is mocked).
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.scripts import deploy_check as dc
from app.scripts.deploy_check import CheckResult, Status

DUMMY_KEY = "gsk_" + "x" * 32


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'deploy.db'}")
    monkeypatch.setattr("app.core.database.engine", engine)
    yield engine
    engine.dispose()


@pytest.fixture
def env(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setattr(settings, key, value)

    _set(GROQ_API_KEY=DUMMY_KEY, ENVIRONMENT="test", DEBUG=False)
    return _set


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------
class TestEnvironment:
    def test_pass(self, env):
        assert dc.check_environment().status is Status.PASS

    def test_missing_key_fails(self, env):
        env(GROQ_API_KEY="")
        result = dc.check_environment()
        assert result.status is Status.FAIL and "GROQ_API_KEY" in result.detail

    def test_unexpected_key_prefix_warns_outside_production(self, env):
        env(GROQ_API_KEY="not-a-groq-key")
        assert dc.check_environment().status is Status.WARN

    def test_production_problems_fail(self, env):
        env(
            ENVIRONMENT="production",
            SECRET_KEY="short",
            DEBUG=True,
            DATABASE_URL="sqlite:///x.db",
        )
        result = dc.check_environment()
        assert result.status is Status.FAIL
        for fragment in ("SECRET_KEY", "DEBUG", "SQLite"):
            assert fragment in result.detail
        assert "short" not in result.detail.replace("shorter", "")


# ---------------------------------------------------------------------------
# Database, pgvector, vector store
# ---------------------------------------------------------------------------
class TestDatabaseChecks:
    def test_database_pass(self, sqlite_engine):
        result = dc.check_database()
        assert result.status is Status.PASS and "sqlite" in result.detail

    def test_database_fail(self, tmp_path, monkeypatch):
        broken = create_engine(f"sqlite:///{tmp_path / 'missing' / 'dir' / 'x.db'}")
        monkeypatch.setattr("app.core.database.engine", broken)
        assert dc.check_database().status is Status.FAIL

    def test_pgvector_warns_on_sqlite(self, sqlite_engine):
        assert dc.check_pgvector().status is Status.WARN

    @pytest.mark.parametrize(
        "installed,expected", [("0.7.0", Status.PASS), (None, Status.FAIL)]
    )
    def test_pgvector_on_postgres(self, monkeypatch, installed, expected):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        conn = engine.connect.return_value.__enter__.return_value
        conn.execute.return_value.scalar.return_value = installed
        monkeypatch.setattr("app.core.database.engine", engine)
        assert dc.check_pgvector().status is expected

    def test_pgvector_error_fails(self, monkeypatch):
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        engine.connect.side_effect = RuntimeError("down")
        monkeypatch.setattr("app.core.database.engine", engine)
        assert dc.check_pgvector().status is Status.FAIL

    def test_vector_store_missing_table_warns(self, sqlite_engine):
        result = dc.check_vector_store()
        assert result.status is Status.WARN and "not found" in result.detail

    def test_vector_store_empty_warns(self, sqlite_engine):
        with sqlite_engine.begin() as conn:
            conn.execute(text("CREATE TABLE vector_store (id INTEGER PRIMARY KEY)"))
        result = dc.check_vector_store()
        assert result.status is Status.WARN and "empty" in result.detail

    def test_vector_store_with_rows_passes(self, sqlite_engine):
        with sqlite_engine.begin() as conn:
            conn.execute(text("CREATE TABLE vector_store (id INTEGER PRIMARY KEY)"))
            conn.execute(text("INSERT INTO vector_store (id) VALUES (1), (2)"))
        result = dc.check_vector_store()
        assert result.status is Status.PASS and result.detail.startswith("2 documents")

    def test_vector_store_error_fails(self, monkeypatch):
        monkeypatch.setattr("app.core.database.engine", object())
        assert dc.check_vector_store().status is Status.FAIL


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
class TestGroq:
    def test_empty_key_fails(self, env):
        env(GROQ_API_KEY="")
        assert dc.check_groq().status is Status.FAIL

    def test_client_init_passes_offline(self, env):
        result = dc.check_groq(live=False)
        assert result.status is Status.PASS and "--live-groq" in result.detail

    def test_client_init_error_fails(self, env):
        with patch("langchain_groq.ChatGroq", side_effect=ValueError("bad")):
            assert dc.check_groq().status is Status.FAIL

    def test_live_call_success(self, env):
        with patch("langchain_groq.ChatGroq") as chat:
            chat.return_value.invoke.return_value = "pong"
            result = dc.check_groq(live=True)
        assert result.status is Status.PASS and result.elapsed_ms is not None

    def test_live_call_rejected(self, env):
        with patch("langchain_groq.ChatGroq") as chat:
            chat.return_value.invoke.side_effect = PermissionError("401")
            assert dc.check_groq(live=True).status is Status.FAIL


# ---------------------------------------------------------------------------
# Clinical safety
# ---------------------------------------------------------------------------
class TestClinicalSafety:
    def test_invariants_hold(self):
        assert dc.check_clinical_safety().status is Status.PASS

    def test_cacheable_clinical_intent_fails(self):
        with patch("app.ai.caching.ResponseCache.should_cache", return_value=True):
            result = dc.check_clinical_safety()
        assert result.status is Status.FAIL and "SYMPTOM" in result.detail

    def test_disclaimer_drift_fails(self):
        with patch("app.utils.constants.SAFETY_DISCLAIMER", "changed"):
            assert dc.check_clinical_safety().status is Status.FAIL


# ---------------------------------------------------------------------------
# Runner, report, CLI
# ---------------------------------------------------------------------------
def _results(*statuses):
    return [
        CheckResult(f"check {i}", s, "detail", elapsed_ms=1.0)
        for i, s in enumerate(statuses)
    ]


def test_run_all_checks_survives_a_crashing_check(sqlite_engine, env):
    with patch.object(dc, "check_pgvector", side_effect=RuntimeError("boom")):
        results = dc.run_all_checks()
    assert len(results) == 6
    crashed = [r for r in results if r.name == "pgvector"]
    assert crashed and crashed[0].status is Status.FAIL
    assert all(r.elapsed_ms is not None for r in results)


def test_render_report_counts():
    report = dc.render_report(_results(Status.PASS, Status.WARN, Status.FAIL))
    assert "1 passed | 1 warnings | 1 failed" in report
    assert "[PASS]" in report and "[WARN]" in report and "[FAIL]" in report


@pytest.mark.parametrize(
    "statuses,argv,code",
    [
        ((Status.PASS, Status.PASS), [], 0),
        ((Status.PASS, Status.WARN), [], 0),
        ((Status.PASS, Status.WARN), ["--strict"], 1),
        ((Status.PASS, Status.FAIL), [], 1),
    ],
)
def test_main_exit_codes(statuses, argv, code, capsys):
    with patch.object(dc, "run_all_checks", return_value=_results(*statuses)):
        assert dc.main(argv) == code
    assert "DEPLOYMENT PRE-FLIGHT CHECK" in capsys.readouterr().out


def test_main_crash_returns_2():
    with patch.object(dc, "run_all_checks", side_effect=RuntimeError("x")):
        assert dc.main([]) == 2
