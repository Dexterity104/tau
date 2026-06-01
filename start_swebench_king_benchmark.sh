#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
exec /home/const/subnet66/tau/.venv/bin/python -m cli swebench-king-benchmark \
  --validate-root /home/const/subnet66/tau/workspace/validate/netuid-66 \
  --manifest /home/const/subnet66/tau/data/swebench_verified_sample_50_seed66.json \
  --baseline mini-swe-agent \
  --mini-swe-agent-repo https://github.com/SWE-agent/mini-swe-agent \
  --mini-swe-agent-ref main \
  --model minimax/minimax-m2.7 \
  --provider-only minimax/fp8 \
  --workers 50 \
  --poll-interval-seconds 60
'
