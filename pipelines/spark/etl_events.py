import argparse

import pandas as pd


def sessionize(df: pd.DataFrame, gap_ms: int = 30 * 60 * 1000) -> pd.DataFrame:
    df = df.sort_values(["user_id", "time_ms"]).copy()
    df["prev_ts"] = df.groupby("user_id")["time_ms"].shift(1)
    df["new_sess"] = (
        (df["prev_ts"].isna()) | ((df["time_ms"] - df["prev_ts"]) > gap_ms)
    ).astype(int)
    df["session_num"] = df.groupby("user_id")["new_sess"].cumsum()
    df["session_id"] = df["user_id"].astype(str) + "-" + df["session_num"].astype(str)
    return df


def run(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df = df.rename(
        columns={
            "video_id": "item_id",
            "play_time_ms": "watch_time_ms",
            "duration_ms": "item_duration_ms",
        }
    )
    df = sessionize(df)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)
