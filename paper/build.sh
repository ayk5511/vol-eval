#!/usr/bin/env bash
# Build Paper 3 and refresh the SSRN-upload PDF.
# Convention (see papers/CONVENTIONS.md): the versioned PDF lives in
# paper/submission-ssrn/ so it's separated from working build artefacts.
set -euo pipefail

VERSION="v2"
OUT_NAME="Khan_2026_vol_eval_${VERSION}.pdf"

cd "$(dirname "$0")"
tectonic main.tex
mkdir -p submission-ssrn
cp main.pdf "submission-ssrn/${OUT_NAME}"
echo "Built:           paper/main.pdf"
echo "Versioned copy:  paper/submission-ssrn/${OUT_NAME}"
