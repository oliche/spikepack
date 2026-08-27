# Contributing to spikepack

## Development setup

```bash
git clone https://github.com/oliche/spikepack
cd spikepack
uv sync --group dev
git config core.hooksPath .githooks
```

## Code style

All code is formatted and linted with [ruff](https://docs.astral.sh/ruff/).
Configuration lives in `pyproject.toml` (`line-length = 120`, rules `E F W I`).

```bash
uv run ruff format src/ tests/
uv run ruff check  src/ tests/
```

## Docstrings

All public functions use [NumPy/SciPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

## Documentation

The docs site lives in `docs/` and follows the [Diátaxis](https://diataxis.fr) framework:

| Directory | Type | Purpose |
|---|---|---|
| `docs/tutorials/` | Tutorial | Guided learning paths |
| `docs/how-to/` | How-To | Step-by-step task guides |
| `docs/reference/` | Reference | API docs + on-disk format spec |
| `docs/explanation/` | Explanation | Codec design rationale, benchmarks |

## Tests

```bash
uv run pytest                              # unit tests only (fast, synthetic data)
uv run pytest -m network                   # + real-data benchmark tests (requires ONE/Alyx access)
uv run pytest --cov --cov-report=term-missing
```

`tests/test_benchmark_pids.py` is marked `network`: it downloads real spike trains
from the standard IBL benchmark insertions and is skipped by default (`pytest`
deselects it via the `-m 'not network'` default in CI).

## Versioning

[Semantic Versioning](https://semver.org). Changes recorded in `CHANGELOG.md`
following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
