# Contributing to CreditRiskEngine

Thank you for your interest in contributing to CreditRiskEngine!

## FINOS Community

CreditRiskEngine is preparing for contribution to the
[Fintech Open Source Foundation (FINOS)](https://www.finos.org/).

Participants are bound by the
[FINOS Community Code of Conduct](.github/CODE_OF_CONDUCT.md) and the
[LF Antitrust Policy](https://www.linuxfoundation.org/antitrust-policy/).

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin (DCO)](DCO). This certifies that you have the
right to submit the contribution under the project's open-source license.

Sign off your commits by adding `-s` to the commit command:

```bash
git commit -s -m "Add downturn LGD floor for residential mortgages"
```

This appends a `Signed-off-by: Your Name <your.email@example.com>` line to the
commit message, using your Git `user.name` and `user.email` configuration.

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

## Contribution Process

1. Fork the repo and create a feature branch from `main`.
2. Make your changes. Add or update tests to maintain 100% coverage.
3. Ensure `pytest`, `ruff check`, and `mypy` all pass locally.
4. Sign off every commit (`git commit -s`).
5. Write a clear commit message describing **why** the change is needed.
6. Open a PR against `main`. The CI pipeline will re-run all checks.

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
using the appropriate template:

- **Bug Report** -- steps to reproduce, expected vs. actual behavior.
- **Feature Request** -- describe the problem and proposed solution.
- **Regulatory Issues** -- cite the paragraph and your interpretation.

## Governance

### Roles

| Role | Description |
|------|-------------|
| **Contributor** | Anyone who submits a pull request or files an issue |
| **Maintainer** | Trusted contributors with merge rights (see [MAINTAINERS.md](MAINTAINERS.md)) |
| **Lead Maintainer** | Overall project direction and FINOS liaison |

### Becoming a Maintainer

Maintainers are nominated by existing maintainers based on:

- Sustained, high-quality contributions over time
- Deep understanding of credit risk regulatory frameworks
- Commitment to the project's goals and code quality standards
- Adherence to the FINOS Code of Conduct

## License

By contributing you agree that your contributions will be licensed under the
Apache 2.0 license (see [LICENSE](LICENSE)).
