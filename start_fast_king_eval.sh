#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec .venv/bin/python -m cli fast-king-eval \
  --validate-root workspace/validate/netuid-66 \
  --terminal-manifest data/terminal_bench_core_fast_50_seed66.json \
  --swebench-manifest data/swebench_verified_sample_50_seed66.json \
  --baseline mini-swe-agent \
  --model minimax/minimax-m2.7 \
  --provider-only minimax/fp8 \
  --workers 50 \
  --terminal-workers 10 \
  --swebench-workers 50 \
  --agent-timeout-seconds 600 \
  --run-timeout-seconds 600 \
  --no-rebuild
