#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_piqa_signed_gate_signal.sh
bash scripts/run_piqa_signed_gate_interventions.sh
