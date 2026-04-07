import argparse
import json

import numpy as np
from qdrant_client.http.models import PointStruct

from libs.storage.vector_db import ensure_collection, upsert_item_vectors


def run(item_factors_path: str, item_index_path: str, batch_size: int = 512) -> None:
    factors = np.load(item_factors_path)
    with open(item_index_path, encoding="utf-8") as f:
        idx_map = json.load(f)

    ensure_collection()
    batch: list[PointStruct] = []
    for idx_str, item_id in idx_map.items():
        vec = factors[int(idx_str)].astype(float).tolist()
        batch.append(PointStruct(id=int(item_id), vector=vec))
        if len(batch) >= batch_size:
            upsert_item_vectors(batch)
            batch = []
    if batch:
        upsert_item_vectors(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-factors", required=True)
    parser.add_argument("--item-index", required=True)
    args = parser.parse_args()
    run(args.item_factors, args.item_index)
