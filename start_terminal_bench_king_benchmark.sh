#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec .venv/bin/python -m cli terminal-bench-king-benchmark \
  --validate-root workspace/validate/netuid-66 \
  --manifest data/terminal_bench_sample_10_seed66.json \
  --baseline terminus \
  --model minimax/minimax-m2.7 \
  --workers 10
