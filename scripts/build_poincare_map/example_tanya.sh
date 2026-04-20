#!/bin/bash

# Compute the repo root relative to this script's location
# (this script lives at scripts/build_poincare_map/example_tanya.sh)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Example using the bundled thioredoxins dataset
python ./main.py --plot true\
    --input_path "${REPO_ROOT}/examples/thioredoxins/fasta0.9/"\
    --output_path "${REPO_ROOT}/results/thioredoxins/TEST/"

# Example using the bundled globins dataset
python ./main.py --plot true\
    --input_path "${REPO_ROOT}/examples/globins/fasta0.9/"\
    --output_path "${REPO_ROOT}/results/globins/TEST/"
