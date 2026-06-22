#!/bin/bash
exec doppler run -p sn66 -c prd -- bash -lc '
set -euo pipefail
umask 002
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in Doppler}"
: "${SOLVER_UPSTREAM_API_KEY:?Set SOLVER_UPSTREAM_API_KEY (self-hosted Qwen endpoint key) in Doppler}"
# Self-hosted Qwen3-32B endpoint (its own upstream + key). The solver proxy reads
# SOLVER_UPSTREAM_BASE_URL; complete_text routes the SELF_HOSTED_MODEL here too.
export SOLVER_UPSTREAM_BASE_URL=http://107.170.115.150:8000/v1
export SELF_HOSTED_MODEL=Qwen/Qwen3-32B
# Non-judge models all run on the self-hosted Qwen3-32B (solver via the proxy;
# generator/eval routed to SOLVER_UPSTREAM_BASE_URL by SELF_HOSTED_MODEL).
export GENERATOR_MODEL=Qwen/Qwen3-32B
export EVAL_MODEL=Qwen/Qwen3-32B
# Both judges -> glm-5.2 via OpenRouter (uses OPENROUTER_API_KEY + OPENROUTER_UPSTREAM_BASE_URL).
export OPENROUTER_UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export TAU_DIFF_JUDGE_MODEL=z-ai/glm-5.2
export PRIVATE_SUBMISSION_JUDGE_MODEL=z-ai/glm-5.2
export SOLVER_SHELL_TOOLS=true
export SOLVER_TEMPERATURE=0
export SOLVER_EMPTY_RESPONSE_RETRIES=5
export SOLVER_RATE_LIMIT_RETRIES=6
exec /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model Qwen/Qwen3-32B \
  --max-concurrency 1 \
  --round-concurrency 50 \
  --docker-solver-start-concurrency 50 \
  --candidate-timeout-streak-limit 10 \
  --poll-interval-seconds 600 \
  --task-pool-target 50 \
  --task-pool-static \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --duel-rounds 50 \
  --win-margin 6 \
  --min-commitment-block 7951985 \
  --hotkey-spent-since-block 8104340 \
  --watch-private-submissions \
  --private-submission-only \
  --publish-repo unarbos/ninja \
  --publish-base main
'
