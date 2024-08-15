import numpy as np
import pandas as pd
import gzip

from typing import Dict, List, Tuple
from ..utils.jsonl_file_utils import open_dataframe, dump_dataframe


def get_htid_content_length(htid_name: str, full_data: pd.DataFrame) -> int:
    filtered_data = full_data[full_data["htid"] == htid_name]

    return -1 if len(filtered_data) == 0 else filtered_data.iloc[0]["content_length"]


if __name__ == "__main__":
    input_dir = "/home/zxia15/data_zxia15/russian-semantics/work/hathi_year_metadata_russian_documents.jsonl.gz"
    content_length_dir = "/home/zxia15/data_zxia15/russian-semantics/scripts/07_30_further_metadata_filtering/output.jsonl.gz"
    dump_dir = "/home/zxia15/data_zxia15/russian-semantics/work/full_metadata_russian_documents.jsonl.gz"

    metadata_df = open_dataframe(input_dir)
    length_correspondence_df = open_dataframe(content_length_dir)
    # print(length_correspondence_df.head())

    metadata_df["content_length"] = metadata_df["htid"].map(
        lambda x: get_htid_content_length(x, length_correspondence_df)
    )

    print(metadata_df.head())

    dump_dataframe(dump_dir, metadata_df)
