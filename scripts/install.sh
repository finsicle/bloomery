#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Aswin Alexander Sam
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Install bloomery on Linux or macOS.
#
#   ./scripts/install.sh                  core only, then report hardware
#   ./scripts/install.sh --with-training  also install PyTorch and the trainers
#
# Core is installed first on purpose. It is a few megabytes, so `bloomery doctor`
# runs within seconds and tells you what this machine can do *before* you commit
# to a multi-gigabyte PyTorch download — and before a wrong backend wastes it.

set -euo pipefail

PYTHON_VERSION="3.12"
WITH_TRAINING=0

for arg in "$@"; do
    case "$arg" in
        --with-training) WITH_TRAINING=1 ;;
        -h | --help)
            sed -n '5,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$arg" >&2
            exit 2
            ;;
    esac
done

info() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$1" >&2; }
die() {
    printf '\033[1;31m==>\033[0m %s\n' "$1" >&2
    exit 1
}

# Run from the repository root regardless of where this was invoked.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[ -f pyproject.toml ] || die "pyproject.toml not found in $REPO_ROOT"

# --------------------------------------------------------------------------- #
# uv
# --------------------------------------------------------------------------- #
if ! command -v uv > /dev/null 2>&1; then
    info "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv here but only edits shell rc files, which do not
    # affect the already-running shell.
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$candidate/uv" ] && PATH="$candidate:$PATH"
    done
    export PATH
    command -v uv > /dev/null 2>&1 || die "uv installed but not on PATH; open a new shell and re-run"
fi
info "uv $(uv --version | awk '{print $2}')"

# --------------------------------------------------------------------------- #
# virtualenv
# --------------------------------------------------------------------------- #
if [ ! -d .venv ]; then
    info "creating .venv on python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION"
else
    info "reusing existing .venv"
fi

# --------------------------------------------------------------------------- #
# core install, then immediately report
# --------------------------------------------------------------------------- #
info "installing bloomery core"
uv pip install --quiet -e .

info "probing hardware"
echo
set +e
.venv/bin/bloomery doctor
DOCTOR_STATUS=$?
set -e
echo

if [ "$DOCTOR_STATUS" -ne 0 ]; then
    warn "doctor reported a blocking problem — see the notes above"
fi

# --------------------------------------------------------------------------- #
# training stack
# --------------------------------------------------------------------------- #
if [ "$WITH_TRAINING" -eq 1 ]; then
    info "installing PyTorch and trainers (several GB, this takes a while)"
    # --torch-backend=auto inspects the CUDA driver, AMD GPU version and Intel
    # GPU presence and resolves the matching wheel index. It is only available
    # on `uv pip`, which is why this is not a `uv sync`.
    uv pip install --torch-backend=auto -e ".[train,serve]"
    info "verifying torch can see the hardware"
    echo
    .venv/bin/bloomery doctor || true
else
    cat <<'EOF'
Core is installed. To add PyTorch and the training stack:

  uv pip install --torch-backend=auto -e ".[train,serve]"

or re-run this script with --with-training.
EOF
fi

echo
info "activate with: source .venv/bin/activate"
