#!/usr/bin/env bash
# Deploy to Vercel (needs the Vercel CLI and the trained artifacts in artifacts/).
set -euo pipefail
cd "$(dirname "$0")"
for f in artifacts/credit_model.txt artifacts/credit_meta.json artifacts/dqn.npz; do
  [[ -f "$f" ]] || { echo "missing $f - run: python scripts/train_all.py"; exit 1; }
done
vercel deploy --prod "$@"
