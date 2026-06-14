#!/usr/bin/env bash
# One-shot launcher for the Grant AI pipeline web server.
# Creates a venv, installs deps, downloads spaCy model, and starts the server.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8000}"

# 1. venv ----------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[start.sh] creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# 2. deps ----------------------------------------------------------------------
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
  echo "[start.sh] installing requirements"
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm || true
  python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]" || true
  touch "$VENV_DIR/.deps_installed"
fi

# 3. data dirs -----------------------------------------------------------------
mkdir -p data/uploads/json_data data/results

# 4. run -----------------------------------------------------------------------
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"
export PORT
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-128000}"
export STAGE1_MAX_TOKENS="${STAGE1_MAX_TOKENS:-50000}"
export STAGE2_MAX_TOKENS="${STAGE2_MAX_TOKENS:-50000}"
echo "[start.sh] launching server on http://0.0.0.0:${PORT}"
exec python web/server.py
