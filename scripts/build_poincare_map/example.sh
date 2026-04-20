#!/bin/bash

# Compute the repo root relative to this script's location
# (this script lives at scripts/build_poincare_map/example.sh)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Example using the bundled thioredoxin dataset
# python ./main.py --plot true\
#     --input_path "${REPO_ROOT}/examples/globins/fasta0.9/"\
#     --output_path "${REPO_ROOT}/results/globins/fasta0.9/"\
#     --epochs 1500

python ./main.py --plot true\
    --input_path "${REPO_ROOT}/examples/thioredoxins/fasta0.9/"\
    --output_path "${REPO_ROOT}/results/thioredoxins/fasta0.9/"\
    --epochs 1500