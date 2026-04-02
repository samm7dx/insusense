#!/usr/bin/env bash
set -euo pipefail

# Owner: Surya (API + DevOps + Cloud)
# Render entrypoint.

# In case Render uses an ephemeral filesystem or the build cache is skipped,
# ensure artifacts exist at runtime as well.
if [[ ! -f "model/model.pkl" || ! -f "model/scaler.pkl" ]]; then
  python train_model.py
fi

PORT="${PORT:-8501}"
exec streamlit run app.py --server.port="$PORT" --server.address="0.0.0.0"
