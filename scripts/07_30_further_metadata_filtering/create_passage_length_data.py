import json, gzip, os, time
import regex as re
from typing import Tuple, List, Dict, Generator, Any


DATA_DIR = "/home/zxia15/data_zxia15/russian-semantics/work/new_re_filtered_russian_documents.jsonl.gz"
DUMP_METADARA_DIR = "output.jsonl.gz"


def OUT_OF_IDX():
    raise Exception("the current data is out of index")


def open_jsonl_file(file_path: str) -> Generator[Dict[str, Any], None, None]:
    with gzip.open(file_path, mode="rt") as f:
        for line in f:
            yield line


def write_jsonl_file(dump_file_path: str, item_data: Dict[str, str]) -> None:
    with gzip.open(dump_file_path, "at", encoding="utf-8") as file:
        json_line = json.dumps(item_data)
        file.write(json_line + "\n")


if __name__ == "__main__":
    start_time = time.time()
    data_generator = open_jsonl_file(DATA_DIR)
    idx: int = 0

    try:
        while True:
            record: Dict[str, any] = json.loads(next(data_generator))
            metadata_instance: Dict[str, int] = {}
            metadata_instance["htid"] = record["htid"]
            metadata_instance["content_length"] = (
                len(record["content"]) if type(record["content"]) == str else 0
            )
            # print(metadata_instance)

            write_jsonl_file(DUMP_METADARA_DIR, metadata_instance)

            if idx % 1000 == 0:
                print(f"querying idx # {idx + 1}")

            idx += 1

    except StopIteration:
        print(f"end of input at {idx} line")
        pass
    except Exception as e:
        raise Exception(f"encounters exception at {idx + 1} line : {str(e)}")
