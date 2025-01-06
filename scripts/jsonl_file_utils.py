import pandas as pd
import gzip, json
from typing import Dict, Generator, Any


def open_dataframe(df_dir: str, printable: bool = False) -> pd.DataFrame:
    metadata_df: pd.DataFrame = pd.read_json(df_dir, lines=True)
    if printable is True:
        print(metadata_df.head())
    return metadata_df


def dump_dataframe(df_dir: str, df_data: pd.DataFrame, printable: bool = False) -> None:
    if printable is True:
        print(df_data.head())
    json_str: str = df_data.to_json(orient="records", lines=True)
    with gzip.open(df_dir, "wt", encoding="utf-8") as f:
        f.write(json_str)
    print(f"successfully dumped into file {df_dir}")



