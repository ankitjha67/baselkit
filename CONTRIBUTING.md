# Contributing to CreditRiskEngine

Thank you for your interest in contributing! This document explains how to get
started.

## Development Setup

```bash
git clone https://github.com/ankitjha67/baselkit.git
cd baselkit
pip install -e ".[dev]"
```

Requires **Python 3.11+**.

## Running Checks

All three must pass before submitting a PR:

```bash
pytest                       # Tests (100% line coverage required)
ruff check src/ tests/       # Linting
mypy src/creditriskengine/   # Type checking (strict mode)
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`.
2. Make your changes. Add or update tests to maintain 100% coverage.
3. Ensure `pytest`, `ruff check`, and `mypy` all pass locally.
4. Write a clear commit message describing **why** the change is needed.
5. Open a PR against `main`. The CI pipeline will re-run all checks.

## Code Style

- Follow existing patterns in the codebase.
- All public functions must have docstrings with parameter documentation.
- Regulatory functions must cite the relevant BCBS/CRR/IFRS paragraph
  (e.g., "BCBS CRE31.5").
- Type annotations are mandatory (`mypy --strict`).

## Regulatory Contributions

If your change affects a regulatory calculation:

- Reference the specific regulatory paragraph in the docstring.
- Document any interpretation choices in the PR description.
- Add backtesting or reference-value tests where possible.
- Review [docs/regulatory_disclaimers.md](docs/regulatory_disclaimers.md) to
  ensure disclaimers remain accurate.

## Reporting Issues

Open an issue on [GitHub Issues](https://github.com/ankitjha67/baselkit/issues)
with:

- A clear description of the bug or feature request.
- For bugs: steps to reproduce, expected vs. actual behavior.
- For regulatory issues: cite the paragraph and your interpretation.

## License

By contributing you agree that your contributions will be licensed under the
Apache 2.0 license (see [LICENSE](LICENSE)).
