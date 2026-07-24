# Contributing to Hoku Health Care — AI Chatbot Backend

Thanks for contributing. This is a **clinical** product: the same code
that answers "what are your visiting hours?" also answers "I have chest
pain." Read the [Clinical Safety Rule](#clinical-safety-rule-read-first)
before writing a line.

---

## Clinical Safety Rule (read first)

These invariants are **non-negotiable**. A PR that weakens any of them
will be rejected regardless of test status:

1. **No definitive diagnosis.** The chatbot informs; it never diagnoses.
2. **Mandatory disclaimer.** Every response path ends with
   *"Please consult a doctor for proper diagnosis."*
   (`SAFETY_DISCLAIMER` in `app/utils/constants.py`).
3. **Emergencies bypass the LLM.** Emergency detection runs first and
   short-circuits the LLM, RAG, symptom extraction, and doctor lookup.
4. **Never cache clinical intents.** `EMERGENCY` and `SYMPTOM` intents
   must never be served from cache. Keep them in `CACHE_EXCLUDE_INTENTS`
   and out of `ResponseCache.should_cache`.
5. **Never log PHI.** Patient messages, AI replies, symptoms, and FAQ
   context must never reach a log sink. `RedactionFilter` enforces this;
   don't add `extra={"user_message": ...}` to loosen it.

`python -m app.scripts.deploy_check` asserts 1, 2, and 4 automatically.

---

## Branch Naming

`<type>/<short-kebab-description>` — for example:

```
feat/prometheus-cache-gauge
fix/emergency-header-severity
chore/pin-locust-version
docs/api-curl-examples
test/rate-limit-429
refactor/rag-context-builder
```

Allowed `<type>` prefixes: `feat`, `fix`, `chore`, `docs`, `test`,
`refactor`, `perf`, `ci`. Branch off `develop`; release branches merge
`develop → main`.

---

## Conventional Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

Examples:

```
feat(monitoring): add cache-hit-ratio Prometheus gauge
fix(safety): append disclaimer on the 3-strike fallback path
test(load): assert P95 < 4s for 50 concurrent users
docs(api): document 429 Retry-After header
```

Rules:
- `type` is one of the branch-name types above.
- Use `feat!:` or a `BREAKING CHANGE:` footer for breaking changes.
- Scope is optional but encouraged (`safety`, `rag`, `monitoring`, `api`).
- Keep the subject ≤ 72 characters, imperative mood ("add", not "added").

---

## Pull Request Requirements

Every PR must:

1. **Target `develop`** (or `main` only for hotfixes).
2. **Pass CI** — the `ci.yml` workflow runs:
   - `ruff check` and `ruff format --check`
   - `mypy app`
   - `pytest` with **coverage ≥ 80%**
   - a Docker build + health-check smoke test
3. **Add or update tests** for any behaviour change. New endpoints need
   an integration test; new pipeline logic needs a unit test.
4. **Mock Groq** in all tests — never call the real API.
5. **Preserve the 215-test baseline** — zero regressions across Days 0–9.
6. **Keep type hints + docstrings** on all new/changed public functions.
7. **Update docs** when endpoints, env vars, or metrics change
   (`README.md`, `app/docs/chatbot_api.md`, `.env.example`).
8. **Fill in the PR description**: what changed, why, and how the
   clinical safety invariants are preserved.

PRs require at least one approving review from the AI Lead or Backend
Lead before merge. Squash-merge; the squash commit message must itself be
a valid Conventional Commit.

---

## Local Development Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # add your GROQ_API_KEY
pre-commit install           # optional, if configured
```

Before pushing:

```bash
ruff check app tests
ruff format app tests
mypy app --ignore-missing-imports
pytest tests/ --cov=app --cov-fail-under=80 --ignore=tests/load
```

---

## Code Style

- **Formatter/linter:** `ruff` (line length 100). Run `ruff format`.
- **Types:** `mypy`-clean on `app/`. Annotate every public signature.
- **Docstrings:** Google-style, on every module, class, and public
  function. Explain *why* for non-obvious safety/performance choices.
- **Imports:** stdlib → third-party → first-party, sorted by ruff.
- **Logging:** module-level `logger = logging.getLogger(__name__)`; never
  `print`; never log PHI.

---

## Questions

Open a draft PR or tag the AI Lead. When in doubt about a clinical-safety
implication, ask first — a withheld change is cheaper than an unsafe
response reaching a patient.