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


def open_jsonl_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    with gzip.open(file_path, mode="rt") as f:
        for line in f:
            yield line


def write_jsonl_file(dump_file_path: str, item_data: Dict[str, str]) -> None:
    with gzip.open(dump_file_path, "at", encoding="utf-8") as file:
        json_line = json.dumps(item_data)
        file.write(json_line + "\n")
