from typing import List, Dict, Set, Tuple, Optional, Generator, Any

from jsonl_file_utils import (
    open_dataframe,
    open_jsonl_file,
    write_jsonl_file,
)

import pandas as pd
import json

input_metadata_dir: str = (
    "/home/zxia15/data_zxia15/russian-semantics/work/final_filtered_metadata_russian_documents.jsonl.gz"
)
input_document_dir: str = (
    "/home/zxia15/data_zxia15/russian-semantics/work/new_re_filtered_russian_documents.jsonl.gz"
)

dump_document_dir: str = (
    "/home/zxia15/data_zxia15/russian-semantics/work/final_filtered_russian_documents.jsonl.gz"
)


if __name__ == "__main__":

    metadata_df: pd.DataFrame = open_dataframe(input_metadata_dir)

    all_htids_list: List[str] = metadata_df["htid"].unique()
    all_written_htids_set: Set[str] = set()

    # each entry of the metadata should have unique htids
    assert len(all_htids_list) == len(metadata_df)

    data_generator: Generator[Dict[str, Any], None, None] = open_jsonl_file(
        input_document_dir
    )

    idx: int = 0
    # 从这里开始，别改了！！
    prior_idx: int = 0
    total_written_idx: int = 0

    # get rid of all the data beforehand
    while idx < prior_idx:
        next(data_generator)
        if idx % 50 == 0:
            print(f"skipping to line #{idx}")
        idx += 1

    print(f"complete skipping the first {prior_idx} instances")

    try:
        while True:
            record: Dict[str, any] = json.loads(next(data_generator))
            # print(record.keys())
            if (
                record["htid"] in all_htids_list
                and record["htid"] not in all_written_htids_set
            ):
                # record["content_length"] = len(record["content"])
                write_jsonl_file(dump_document_dir, record)
                total_written_idx += 1
                all_written_htids_set.add(record["htid"])

            idx += 1

            if idx % 2000 == 0:
                print(f"navigating to {idx} instance")

    except StopIteration:
        print(f"end of input at {idx} line")
        pass
    except KeyboardInterrupt:
        print(
            f"KeyboardInterrupt caught. Currently at {idx} line with {total_written_idx} lines written."
        )
        raise KeyboardInterrupt
    except Exception as e:
        raise Exception(f"encounters exception at {idx} line : {str(e)}")

    assert total_written_idx == len(all_written_htids_set)
    assert len(all_written_htids_set) == len(all_htids_list)
