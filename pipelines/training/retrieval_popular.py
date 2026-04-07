import argparse
import json

import pandas as pd


def run(input_path: str, output_path: str, top_k: int = 200) -> None:
    df = pd.read_csv(input_path)
    score = df.groupby("item_id")["watch_time_ms"].sum().sort_values(ascending=False).head(top_k)
    out = [{"item_id": int(item_id), "score": float(value)} for item_id, value in score.items()]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()
    run(args.input, args.output, args.top_k)
