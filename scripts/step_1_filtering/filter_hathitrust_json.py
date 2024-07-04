import json, gzip, os, time
import regex as re
from typing import Tuple, List, Dict, Generator, Any

DEBUG_MODE : bool = True
CREATE_METADATA : bool = False
RE_FILTER_MODE : bool = DEBUG_MODE is not True

EPSILON : float = 1e-7

DATA_DIR = '/home/zxia15/data_zxia15/russian-semantics/work/full_russian_documents.jsonl.gz'
DUMP_RE_DATA_DIR = '/home/zxia15/data_zxia15/russian-semantics/work/new_re_filtered_russian_documents.jsonl.gz'
DUMP_METADARA_DIR = '/home/zxia15/data_zxia15/russian-semantics/work/metadata_russian_documents.jsonl.gz'

CYRILLIC_UNICODE : Dict[str, Tuple[str, str, int]] = {
    'Cyrillic': ('\u0400', '\u04FF', 256),
    'Cyrillic_Supplement' : ('\u0500', '\u052F', 48),
    'Cyrillic_Extended_A' : ('\u2DE0', '\u2DFF', 32),
    'Cyrillic_Extended_B' : ('\uA640', '\uA69F', 96),
    'Cyrillic_Extended_C' : ('\u1C80', '\u1C8F', 9),
    # 'Cyrillic_Extended_D' : ('\u1E030', '\u1E08F', 63),
    'Phonetic_Extensions' : ('\u1D2B', '\u1D78', 2),
    'Combining Half Marks' : ('\uFE2E', '\uFE2F', 2),
}


def OUT_OF_IDX():
    raise Exception('the current data is out of index')


def open_jsonl_file(file_path : str) -> Generator[Dict[str, Any], None, None]:
    with gzip.open(file_path, mode="rt") as f:
        for line in f:
            yield line

def write_jsonl_file(dump_file_path : str, item_data :Dict[str, str]) -> None:
    with gzip.open(dump_file_path, 'at', encoding='utf-8') as file:
        json_line = json.dumps(item_data)
        file.write(json_line + "\n")

def check_cyrillic_dicts() -> None:
    for (_, value) in CYRILLIC_UNICODE.items():
        (lower_b, upper_b, range_num) = value
        print(f"the characters lower bound is {lower_b}, upper bound is {upper_b}")
        print(f"the range encompasses {ord(upper_b)- ord(lower_b) + 1} code points but is only assigned {range_num} code points")

def re_filtering(data: str) -> str:
    # character class include:
    pattern = re.compile(r'[\u0049\u0056\u0058\u004C\u0043\u0044\u004D\u0000-\u0040\u005B-\u0060\u007B-\u007F\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u1D2B-\u1D78\uFE2E-\uFE2F]*')
    matches = pattern.findall(data)
    return matches
    

if __name__ == '__main__':
    start_time = time.time()
    data_generator = open_jsonl_file(DUMP_RE_DATA_DIR) if DEBUG_MODE is True else open_jsonl_file(DATA_DIR)
    idx : int = 0
    # 从这里开始，别改了！！
    prior_idx : int = 89353

    # get rid of all the data beforehand
    while idx < prior_idx:
        next(data_generator)
        if idx % 50 == 0:
            print(f"skipping to line #{idx}")
        idx += 1
    
    print(f"complete skipping the first {prior_idx} instances")

    try:
        if DEBUG_MODE is True:
            while True:
                record : Dict[str, any] = json.loads(next(data_generator))
                if CREATE_METADATA is True:
                    metadata_instance : Dict[str, any] = {}
                    for key, value in record.items():
                        if key != 'content':
                            metadata_instance[key] = value
                    write_jsonl_file(DUMP_METADARA_DIR, metadata_instance)
                if idx % 50 == 0:
                    print(f"record #{idx}: has russian percentage {record['russian text ratio']}%")
                idx += 1
        
        if RE_FILTER_MODE is True:
            while True:
                record : Dict[str, str] = json.loads(next(data_generator))
                record_copy : Dict[str, str]  = record.copy()
                record_text : str = record['content']
                record_matches : List[str] = re_filtering(record_text)
                record_copy['content'] = ''.join(record_matches)
                record_copy['russian text ratio'] = float(len(record_copy['content'])) / (float(len(record['content'])) +  + EPSILON) * 100
                write_jsonl_file(DUMP_RE_DATA_DIR, record_copy)

                if CREATE_METADATA is True:
                    metadata_instance : Dict[str, any] = {}
                    for key, value in record_copy.items():
                        if key != 'content':
                            metadata_instance[key] = value
                    write_jsonl_file(DUMP_METADARA_DIR, metadata_instance)
            
                if idx % 50 == 0:
                    print(f"querying idx # {idx + 1}")
                idx += 1

    except StopIteration:
        print(f'end of input at {idx} line')
        pass
    except Exception as e:
        raise Exception(f'encounters exception at {idx + 1} line : {str(e)}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    

