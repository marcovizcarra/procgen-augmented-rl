#!/usr/bin/env bash
# Creates the conda environment in x86_64 (Rosetta) mode so that the
# pre-built osx-64 procgen wheel is used instead of building from source.
set -e

ENV_NAME="procgen-augmented-rl"

echo "Creating conda environment in osx-64 (Rosetta) mode..."
CONDA_SUBDIR=osx-64 conda env create -f environment.yml

echo "Pinning subdir to osx-64 so future 'conda install' commands stay on x86_64..."
conda run -n "$ENV_NAME" conda config --env --set subdir osx-64

echo "Installing legacy gym build tooling (needed for gym==0.21.0)..."
conda run -n "$ENV_NAME" python -m pip install "pip<24.1" "setuptools<66" "wheel<0.39" "packaging<24"

echo "Installing gym==0.21.0..."
conda run -n "$ENV_NAME" python -m pip install "gym==0.21.0" --no-build-isolation

echo "Done! Activate with: conda activate $ENV_NAME"
