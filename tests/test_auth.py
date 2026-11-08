"""Authentication: login, JWT validation, startup guard, demo login, admin CLI."""

import base64
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jose import jwt
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.ai.config import AISettings
from app.api.v1.endpoints.auth import reset_demo_rate_limiter
from app.core.config import Settings, settings
from app.core.security import (
    JWT_ALGORITHM,
    decode_access_token,
    validate_security_settings,
    verify_password,
)
from app.crud.crud_user import create_user, purge_demo_users
from app.models.models_chat import ChatHistory
from app.models.user import User
from tests.auth_utils import add_user, bearer

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "correct horse battery staple"
EMAIL = "Patient.One@Example.com"
PROTECTED = "/api/ai/chat/history"


@pytest.fixture
def patient(db):
    return create_user(db, EMAIL, PASSWORD, role="patient")


def _claims(**overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "1",
        "role": "patient",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
    }
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


def _sign(claims, key=None, algorithm=JWT_ALGORITHM):
    return jwt.encode(claims, key or settings.SECRET_KEY, algorithm=algorithm)


def _b64(data: dict) -> str:
    return (
        base64.urlsafe_b64encode(json.dumps(data, default=str).encode())
        .rstrip(b"=")
        .decode()
    )


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
class TestLogin:
    def test_success_returns_token_for_user(self, client, patient):
        resp = client.post(
            "/api/auth/login", json={"email": EMAIL, "password": PASSWORD}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        claims = decode_access_token(body["access_token"])
        assert claims["sub"] == str(patient.id)
        assert claims["role"] == "patient"
        assert (
            client.get(
                PROTECTED, headers={"Authorization": f"Bearer {body['access_token']}"}
            ).status_code
            == 200
        )

    def test_email_is_case_insensitive(self, client, patient):
        resp = client.post(
            "/api/auth/login", json={"email": EMAIL.upper(), "password": PASSWORD}
        )
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "email,password",
        [(EMAIL, "wrong password here"), ("nobody@example.com", PASSWORD)],
        ids=["wrong-password", "unknown-email"],
    )
    def test_failures_are_identical_401(self, client, patient, email, password):
        resp = client.post(
            "/api/auth/login", json={"email": email, "password": password}
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Incorrect email or password"

    def test_inactive_user_rejected(self, client, db, patient):
        patient.is_active = False
        db.commit()
        resp = client.post(
            "/api/auth/login", json={"email": EMAIL, "password": PASSWORD}
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Incorrect email or password"

    @pytest.mark.parametrize("body", [{}, {"email": EMAIL}, {"password": PASSWORD}])
    def test_missing_fields_422(self, client, body):
        assert client.post("/api/auth/login", json=body).status_code == 422

    def test_password_over_72_bytes_422_without_echo(self, client, patient):
        long_pw = "é" * 37  # 74 UTF-8 bytes
        resp = client.post(
            "/api/auth/login", json={"email": EMAIL, "password": long_pw}
        )
        assert resp.status_code == 422
        assert long_pw not in resp.text

    def test_no_credentials_in_response_or_logs(self, client, patient, caplog):
        caplog.set_level(logging.DEBUG)
        for pw in (PASSWORD, "wrong password here"):
            resp = client.post("/api/auth/login", json={"email": EMAIL, "password": pw})
            assert EMAIL.lower() not in resp.text and pw not in resp.text
        client.post(
            "/api/auth/login", json={"email": "ghost@example.com", "password": PASSWORD}
        )
        for record in caplog.records:
            text = record.getMessage() + str(record.args)
            assert EMAIL.lower() not in text.lower()
            assert "ghost@example.com" not in text
            assert PASSWORD not in text


# ---------------------------------------------------------------------------
# Token validation (get_current_user)
# ---------------------------------------------------------------------------
class TestTokenValidation:
    @pytest.fixture(autouse=True)
    def _user(self, db):
        add_user(db, user_id=1, role="patient")

    def _get(self, client, token):
        return client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})

    def test_valid_token_accepted(self, client):
        assert self._get(client, _sign(_claims())).status_code == 200

    def test_missing_header(self, client):
        assert client.get(PROTECTED).status_code == 401

    @pytest.mark.parametrize(
        "token_factory",
        [
            pytest.param(lambda: "not-a-jwt", id="malformed"),
            pytest.param(lambda: _sign(_claims(), key="x" * 40), id="bad-signature"),
            pytest.param(
                lambda: _sign(
                    _claims(exp=datetime.now(timezone.utc) - timedelta(seconds=5))
                ),
                id="expired",
            ),
            pytest.param(lambda: _sign(_claims(iss="someone-else")), id="wrong-iss"),
            pytest.param(lambda: _sign(_claims(aud="other-api")), id="wrong-aud"),
            pytest.param(lambda: _sign(_claims(iss=None)), id="missing-iss"),
            pytest.param(lambda: _sign(_claims(aud=None)), id="missing-aud"),
            pytest.param(lambda: _sign(_claims(exp=None)), id="missing-exp"),
            pytest.param(lambda: _sign(_claims(sub=None)), id="missing-sub"),
            pytest.param(lambda: _sign(_claims(sub="abc")), id="non-integer-sub"),
            pytest.param(lambda: _sign(_claims(role=None)), id="missing-role"),
            pytest.param(lambda: _sign(_claims(role="superuser")), id="unknown-role"),
            pytest.param(lambda: _sign(_claims(), algorithm="HS512"), id="alg-HS512"),
            pytest.param(
                lambda: f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(_claims())}.",
                id="alg-none",
            ),
        ],
    )
    def test_invalid_tokens_401(self, client, token_factory):
        assert self._get(client, token_factory()).status_code == 401

    def test_deleted_user_401(self, client, db):
        token = _sign(_claims())
        db.delete(db.get(User, 1))
        db.commit()
        assert self._get(client, token).status_code == 401

    def test_inactive_user_401(self, client, db):
        token = _sign(_claims())
        db.get(User, 1).is_active = False
        db.commit()
        assert self._get(client, token).status_code == 401

    def test_role_mismatch_with_db_401(self, client):
        assert self._get(client, _sign(_claims(role="admin"))).status_code == 401

    def test_failure_log_has_no_token(self, client, caplog):
        caplog.set_level(logging.DEBUG)
        token = _sign(_claims(), key="y" * 40)
        self._get(client, token)
        assert token not in caplog.text


# ---------------------------------------------------------------------------
# Startup guard and .env isolation
# ---------------------------------------------------------------------------
class TestStartupGuard:
    SHORT_KEY = "short-but-recognisable-key"

    def _settings(self, env, key):
        return Settings(_env_file=None, ENVIRONMENT=env, SECRET_KEY=key)

    @pytest.mark.parametrize("env", ["production", "development", "staging"])
    def test_short_key_rejected_outside_test(self, env, caplog):
        caplog.set_level(logging.DEBUG)
        with pytest.raises(RuntimeError) as excinfo:
            validate_security_settings(self._settings(env, self.SHORT_KEY))
        assert self.SHORT_KEY not in str(excinfo.value)
        assert self.SHORT_KEY not in repr(excinfo.value)
        assert self.SHORT_KEY not in caplog.text

    def test_empty_key_rejected_outside_test(self):
        with pytest.raises(RuntimeError):
            validate_security_settings(self._settings("production", ""))

    def test_long_key_accepted(self):
        validate_security_settings(self._settings("production", "k" * 32))

    def test_short_key_allowed_in_test(self):
        validate_security_settings(self._settings("test", ""))

    def test_app_import_refuses_short_key(self, tmp_path):
        """End to end: importing app.main exits non-zero and never prints the key."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SECRET_KEY", "ENVIRONMENT", "GROQ_API_KEY")
        }
        env.update(
            ENVIRONMENT="production",
            SECRET_KEY=self.SHORT_KEY,
            PYTHONPATH=str(ROOT),
            DATABASE_URL=f"sqlite:///{tmp_path / 'guard.db'}",
        )
        # cwd=tmp_path so no .env file can be picked up.
        proc = subprocess.run(
            [sys.executable, "-c", "import app.main"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode != 0
        assert "SECRET_KEY must be at least 32 characters" in proc.stderr
        assert self.SHORT_KEY not in proc.stdout + proc.stderr

    def test_env_file_not_read_in_test_environment(self):
        assert Settings.model_config["env_file"] is None
        assert AISettings.model_config["env_file"] is None


# ---------------------------------------------------------------------------
# Demo login
# ---------------------------------------------------------------------------
class TestDemoLogin:
    @pytest.fixture(autouse=True)
    def _limiter(self):
        reset_demo_rate_limiter()
        yield
        reset_demo_rate_limiter()

    @pytest.fixture
    def enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "DEMO_LOGIN_ENABLED", True)

    def test_disabled_by_default_404(self, client):
        assert settings.DEMO_LOGIN_ENABLED is False
        assert client.post("/api/auth/demo").status_code == 404

    def test_enabled_issues_short_patient_token(self, client, db, enabled):
        resp = client.post("/api/auth/demo")
        assert resp.status_code == 200
        body = resp.json()
        assert body["expires_in"] == 15 * 60
        claims = decode_access_token(body["access_token"])
        assert claims["role"] == "patient"
        assert claims["exp"] - claims["iat"] <= 15 * 60
        user = db.get(User, int(claims["sub"]))
        assert user.is_demo is True
        assert user.role == "patient"
        assert not verify_password("", user.hashed_password)

    def test_each_demo_user_has_separate_history(self, client, db, enabled):
        tokens = [
            client.post("/api/auth/demo").json()["access_token"] for _ in range(2)
        ]
        ids = [int(decode_access_token(t)["sub"]) for t in tokens]
        assert ids[0] != ids[1]
        db.add(
            ChatHistory(
                user_id=ids[0], message="only for A", ai_response="ok", intent="general"
            )
        )
        db.commit()
        hist_a = client.get(
            PROTECTED, headers={"Authorization": f"Bearer {tokens[0]}"}
        ).json()
        hist_b = client.get(
            PROTECTED, headers={"Authorization": f"Bearer {tokens[1]}"}
        ).json()
        assert [m["content"] for m in hist_a["messages"]] == ["only for A", "ok"]
        assert hist_b["messages"] == []

    def test_rate_limited_per_ip(self, client, enabled, monkeypatch):
        monkeypatch.setattr(settings, "DEMO_RATE_LIMIT_PER_HOUR", 2)
        reset_demo_rate_limiter()
        codes = [client.post("/api/auth/demo").status_code for _ in range(3)]
        assert codes == [200, 200, 429]

    def test_demo_call_purges_stale_demo_users(self, client, db, enabled):
        old = add_user(db, user_id=None, role="patient", is_demo=True)
        old.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        db.commit()
        old_email = old.email  # ids can be reused by SQLite, emails are random
        assert client.post("/api/auth/demo").status_code == 200
        db.expire_all()
        assert db.scalar(select(User).where(User.email == old_email)) is None
        assert db.scalar(select(func.count()).select_from(User)) == 1


class TestPurge:
    def test_purge_removes_only_stale_demo_users_and_their_history(self, db):
        now = datetime.now(timezone.utc)
        stale = add_user(db, user_id=None, is_demo=True)
        fresh = add_user(db, user_id=None, is_demo=True)
        real_old = add_user(db, user_id=None, is_demo=False)
        stale.created_at = now - timedelta(hours=25)
        real_old.created_at = now - timedelta(days=30)
        db.add(
            ChatHistory(
                user_id=stale.id, message="m", ai_response="r", intent="general"
            )
        )
        db.add(
            ChatHistory(
                user_id=fresh.id, message="m", ai_response="r", intent="general"
            )
        )
        db.commit()
        stale_id, fresh_id, real_id = stale.id, fresh.id, real_old.id

        assert purge_demo_users(db, now=now) == 1
        db.expire_all()
        assert db.get(User, stale_id) is None
        assert db.get(User, fresh_id) is not None
        assert db.get(User, real_id) is not None
        remaining = db.scalars(select(ChatHistory.user_id)).all()
        assert remaining == [fresh_id]

    def test_purge_cli(self, db, capsys):
        from app.scripts import purge_demo_users as cli

        stale = add_user(db, user_id=None, is_demo=True)
        stale.created_at = datetime.now(timezone.utc) - timedelta(hours=48)
        db.commit()
        assert cli.main(sessionmaker(bind=db.get_bind())) == 1
        assert "Removed 1 demo users" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# create_user CLI
# ---------------------------------------------------------------------------
class TestCreateUserCli:
    NEW_PW = "a brand new long password"

    @pytest.fixture
    def factory(self, db):
        return sessionmaker(bind=db.get_bind())

    def _run(self, factory, *args):
        from app.scripts import create_user as cli

        return cli.main(list(args), session_factory=factory)

    def test_create_admin_from_env(self, db, factory, monkeypatch, capsys):
        monkeypatch.setenv("HOKU_NEW_USER_PASSWORD", self.NEW_PW)
        assert self._run(factory, "--email", "Ops@Example.com", "--role", "admin") == 0
        user = db.scalar(select(User).where(User.email == "ops@example.com"))
        assert user.role == "admin" and not user.is_demo
        assert verify_password(self.NEW_PW, user.hashed_password)
        out = capsys.readouterr()
        assert self.NEW_PW not in out.out + out.err

    def test_create_from_prompt(self, db, factory, monkeypatch):
        monkeypatch.delenv("HOKU_NEW_USER_PASSWORD", raising=False)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": self.NEW_PW)
        assert self._run(factory, "--email", "p@example.com") == 0
        user = db.scalar(select(User).where(User.email == "p@example.com"))
        assert user.role == "patient"

    def test_prompt_mismatch_creates_nothing(self, db, factory, monkeypatch):
        monkeypatch.delenv("HOKU_NEW_USER_PASSWORD", raising=False)
        answers = iter([self.NEW_PW, self.NEW_PW + "x"])
        monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
        assert self._run(factory, "--email", "p@example.com") == 2
        assert db.scalar(select(func.count()).select_from(User)) == 0

    def test_short_password_rejected(self, db, factory, monkeypatch):
        monkeypatch.setenv("HOKU_NEW_USER_PASSWORD", "short")
        assert self._run(factory, "--email", "p@example.com") == 2
        assert db.scalar(select(func.count()).select_from(User)) == 0

    def test_existing_user_untouched_without_reset(
        self, db, factory, monkeypatch, patient
    ):
        before = patient.hashed_password
        monkeypatch.setenv("HOKU_NEW_USER_PASSWORD", self.NEW_PW)
        assert self._run(factory, "--email", EMAIL, "--role", "admin") == 1
        db.expire_all()
        user = db.get(User, patient.id)
        assert user.hashed_password == before and user.role == "patient"

    def test_reset_password(self, db, factory, monkeypatch, patient):
        monkeypatch.setenv("HOKU_NEW_USER_PASSWORD", self.NEW_PW)
        assert self._run(factory, "--email", EMAIL, "--reset-password") == 0
        db.expire_all()
        assert verify_password(self.NEW_PW, db.get(User, patient.id).hashed_password)

    def test_reset_unknown_user(self, factory, monkeypatch):
        monkeypatch.setenv("HOKU_NEW_USER_PASSWORD", self.NEW_PW)
        assert (
            self._run(factory, "--email", "nobody@example.com", "--reset-password") == 1
        )


def test_bearer_helper_matches_login_claims(db):
    add_user(db, user_id=7, role="admin")
    claims = decode_access_token(bearer(7, "admin").split()[1])
    assert claims["sub"] == "7" and claims["role"] == "admin"
