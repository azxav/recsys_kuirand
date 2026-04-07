import argparse
from pathlib import Path

import pandas as pd


def run(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> None:
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be >0 and sum to <1")

    df = pd.read_csv(input_path)
    if "time_ms" not in df.columns and "ts_ms" not in df.columns:
        raise ValueError("input must contain time_ms or ts_ms column")
    ts_col = "time_ms" if "time_ms" in df.columns else "ts_ms"

    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.iloc[:train_end].to_csv(outdir / "train.csv", index=False)
    df.iloc[train_end:valid_end].to_csv(outdir / "valid.csv", index=False)
    df.iloc[valid_end:].to_csv(outdir / "test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
