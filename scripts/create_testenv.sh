#!/usr/bin/env bash

set -ex # fail on first error, print commands

PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6
PYTORCH=${PYTORCH:-"pytorch>=1.0.0"}

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

conda install --yes python=${PYTHON_VERSION}
pip install --upgrade pip
conda install --yes numpy mkl-service
conda install --yes ${PYTORCH} -c pytorch
pip install -r requirements-dev.txt

