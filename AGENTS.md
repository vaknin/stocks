# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code by domain
  - `data/`, `features/`, `models/`, `trading/`, `analysis/`, `backtesting/`, `monitoring/`, `config/`, `utils/`
- `tests/`: Pytest suite in `tests/test_*.py` (unit + integration).
- `trading_advisor.py`: CLI for daily recommendations and portfolio ops.
- `portfolio.toml`: Example portfolio state file.
- `colab_setup/`: Colab-first install scripts and pinned requirements.
- `README.md`: Overview and usage.

## Operational Constraints & Goals
- Colab-only execution for production runs; do not run locally.
- Do not execute bash or Python scripts outside Colab.
- Sole target: generate monthly revenue on stocks.
- Use state-of-the-art methods with direct revenue impact.
- Always include tests and follow best practices.
- Before adding packages, ensure they’re in `colab_setup/colab_requirements.txt` or justify additions.
- No mock/placeholder code.
