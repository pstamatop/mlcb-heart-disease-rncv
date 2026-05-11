import pandas as pd

def describe_by_group(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        frame.groupby("group")[col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_index()
        .round(3)
    )
