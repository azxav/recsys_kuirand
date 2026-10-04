#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw
cd data/raw
if [[ ! -f KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv ]]; then
  wget -c https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
  tar -xzf KuaiRand-Pure.tar.gz
fi
