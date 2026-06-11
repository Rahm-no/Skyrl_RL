# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SkyRL is a full-stack reinforcement learning library for training LLMs, designed for modularity and extensibility.

## Critical Rules

- **Always use `uv run --isolated`** to run commands. Never use bare `python`, `pip`, or `pip install`.
- **Log output to files**: `<cmd> > /tmp/results_1.log 2>&1` for persistence.
- Backend extras (`fsdp`, `megatron`, `jax`) conflict with each other — never combine them.
- Always read the relevant documentation files in `.claude/docs` before troubleshooting or working on any changes. Follow the routing rules below.
- **Sign off every commit**: `git commit -s`
- **Run pre-commit hooks** before every commit.

## Test Commands

```bash
# CPU tests
uv run --extra dev --extra jax pytest tests/tx/ tests/tinker/ tests/utils/
uv run --extra dev pytest tests/train/ tests/backends/skyrl_train/ --ignore=tests/backends/skyrl_train/gpu/

# GPU tests (requires Ray cluster with GPUs)
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py

# Run a single test
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py -k "test_name" -v

# Weight sync tests
uv run --extra dev pytest tests/backends/skyrl_train/weight_sync/ -v

# Lint / format
bash format.sh
```

## Training Quick Start

```bash
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron trainer.policy.model.path=<model> environment.env_class=gsm8k ...

# Use --env-file for secrets
uv run --isolated --extra megatron --env-file .env.test -m skyrl.train.entrypoints.main_base ...
```

## Architecture

- **Ray orchestration**: Training workers and inference engines run as Ray actors.
- **Config hierarchy**: `SkyRLTrainConfig` → `TrainerConfig`, `GeneratorConfig`, `DataConfig`, `EnvironmentConfig`. Accessed as `cfg.trainer.*`, `cfg.generator.*`, etc. OmegaConf for CLI parsing, loaded into dataclasses for typing. Pass overrides as `key=value` args — no `+` prefix for new keys.
- **Backend selection**: `trainer.strategy` chooses `fsdp` (default), `megatron`, or `jax`.
- **Environments**: `skyrl-gym/skyrl_gym/envs/` — each env extends `BaseTextEnv` with `step()`.
- **Inference**: New path (`_SKYRL_USE_NEW_INFERENCE=1`, default) uses `RemoteInferenceClient` → `VLLMRouter` → `VLLMServerActor`. Legacy path (`=0`) uses `InferenceEngineClient` → `RayWrappedInferenceEngine`/`RemoteInferenceEngine`.
- **Weight sync**: After each training step, NCCL broadcast (non-colocated) or CUDA IPC (colocated with `colocate_all=true`) pushes training weights into vLLM. New path uses 3-phase chunked lifecycle (`start_weight_update` / `update_weights_chunk` / `finish_weight_update`).

## Code Style

- **Google Python Style Guide** throughout.
- Comments describe **why**, not what. Never reference the current task, fix, or callers in comments — those belong in the PR description and rot over time. See `.claude/docs/contributing.md` for do/don't examples.
- For tokenizer init, use `skyrl.utils.tok.get_tokenizer` — not manual init.
- Import `vllm` lazily (inside methods, not at module top) — it's a Linux-only optional dep.

## Anti-patterns

- Ray tasks/actors with `fork` start method — use `spawn`.
- Passing the full `SkyRLTrainConfig` when only a sub-config (e.g. `InferenceEngineConfig`) is needed.
- Manual `ray.init`, `ray.shutdown`, `ray.kill` in tests — use the provided Ray fixtures.
- Using `InferenceEngineState` manually in tests — use the helper instead.

## Contribution Checklist

- Update relevant example scripts in `examples/train/<task>/` for any changes.
- Update `.claude/` files and `CLAUDE.md` for any path/naming changes.
- For documentation changes: `cd docs/; npm install; npm run build`.
- When bumping `megatron-bridge`, refresh `.claude/skills/parallelism-strategies/SKILL.md`.

## Routing Rules

When working on these areas, read the corresponding doc first:

| Area | Read first |
|------|-----------|
| Package management, uv, formatting | `.claude/docs/development.md` |
| Overall guide for modifying or working with SkyRL | `.claude/docs/contributing.md` |
| Tests, fixtures, CI quirks | `.claude/docs/testing.md` |
| Project layout, Ray actors, config | `.claude/docs/architecture.md` |
| Training entrypoints, configs | `.claude/docs/training.md` |
| Inference engines, vLLM, PD disagg | `.claude/docs/inference.md` |
| GitHub Actions, Anyscale CI | `.claude/docs/ci.md` |
| Tinker API server | `.claude/docs/tinker.md` |
| Megatron backend | `.claude/docs/backends/megatron.md` |
| FSDP backend | `.claude/docs/backends/fsdp.md` |
| JAX/TPU backend | `.claude/docs/backends/jax.md` |
| Weight sync | `.claude/docs/weight_sync.md` |

## Troubleshooting

1. Check known errors: `docs/content/docs/troubleshooting/troubleshooting.mdx`
2. See contributing guide: `.claude/docs/contributing.md`
