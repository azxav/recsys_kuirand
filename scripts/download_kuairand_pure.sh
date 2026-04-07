#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw
cd data/raw
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz
