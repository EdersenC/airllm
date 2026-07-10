# Repository Guidelines

## Project Structure & Module Organization

The installable package lives in `air_llm/`. Core streaming and model-family code is in
`air_llm/airllm/`; persistence back ends are in `air_llm/airllm/persist/`. Keep
architecture-specific behavior in the appropriate `airllm_<family>.py` module and shared
streaming behavior in `airllm_base.py` or a focused helper module. Unit tests live in
`air_llm/tests/`; notebooks and examples belong in `air_llm/examples/`. Use `scripts/` for
repeatable local runners and `benchmarks/` for performance comparisons. Do not commit model
weights, Hugging Face caches, or generated benchmark artifacts.

## Build, Test, and Development Commands

Create an isolated Python 3.12 environment and install the checkout:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e './air_llm[awq]'
```

Run focused CPU-safe tests while developing:

```bash
.venv/bin/python -m unittest air_llm.tests.test_group_streaming -v
.venv/bin/python -m unittest air_llm/tests/test_cpu_layer_cache.py -v
```

`air_llm/tests/test_streaming_gpu.py` is a manual GPU/model harness; invoke it with an
explicit model, for example `python air_llm/tests/test_streaming_gpu.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --compare`.
Use `scripts/run_qwen3_awq.py` for the documented WSL/Linux smoke test.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and
variables, `PascalCase` for classes, and short docstrings for public or non-obvious behavior.
Prefer typed, focused helpers over broad changes to model dispatch. There is no repository
formatter or linter configuration; keep imports grouped, avoid unrelated reformatting, and
validate changed files with `python -m py_compile <files>`.

## Testing Guidelines

Use `unittest`, name files `test_*.py`, and name cases `test_<behavior>`. Add dependency-light
CPU tests for logic such as grouping, caching, and error handling. Gate CUDA-only tests and keep
large model downloads out of normal test runs. Test both success and cleanup/error paths for
streaming changes.

## Commit & Pull Request Guidelines

Recent history favors concise, imperative subjects with prefixes such as `perf:`, `chore:`, and
`fix:`. Keep commits scoped. PRs should explain the user-visible or performance impact, list the
commands run, link relevant issues, and include measured VRAM/throughput output or screenshots
when changing benchmarks or docs. Never include tokens, model weights, or cache contents in
patches; documented example paths are fine.
