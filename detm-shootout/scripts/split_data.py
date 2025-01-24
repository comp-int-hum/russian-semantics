import logging
import gzip
import math
import json
import random
import argparse
import re


logger = logging.getLogger("split_data")


def split_into_subdocs(text, max_subdoc_length, lowercase):
    tokens = re.split(r"\s+", text.lower() if lowercase else text)
    num_subdocs = math.ceil(len(tokens) / max_subdoc_length)
    retval = []
    for i in range(num_subdocs):
        retval.append(tokens[i * max_subdoc_length : (i + 1) * max_subdoc_length])
    return retval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Input JSONL file")
    parser.add_argument("--first_output", dest="first_output", help="")
    parser.add_argument("--second_output", dest="second_output", help="")
    parser.add_argument("--second_proportion", dest="second_proportion", type=float)
    parser.add_argument("--split_field", dest="split_field")
    parser.add_argument("--random_seed", dest="random_seed", default=None)
    parser.add_argument("--down_sample", dest="down_sample", default=0.0, type=float)
    parser.add_argument("--content_field", dest="content_field")
    parser.add_argument("--max_subdoc_length", dest="max_subdoc_length", type=int, required=True)
    parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true")
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    data = {}
    count = 0
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            key = j[args.split_field] if args.split_field else i
            subdocs = split_into_subdocs(j[args.content_field], args.max_subdoc_length, args.lowercase)
            if args.down_sample > 0.0:
                random.shuffle(subdocs)
                subdocs = subdocs[:int((1.0 - args.down_sample) * len(subdocs))]
            j[args.content_field] = subdocs
            data[i] = data.get(i, [])
            data[i].append(j)
            count += 1

    data = list(data.values())
    random.shuffle(data)

    target = int(count * args.second_proportion)
    written = 0
    with gzip.open(args.first_output, "wt") as ofdA, gzip.open(args.second_output, "wt") as ofdB:
        for data_chunk in data:
            if written < target:
                for datum in data_chunk:
                    ofdB.write(json.dumps(datum) + "\n")
                    written += 1
            else:
                for datum in data_chunk:
                    ofdA.write(json.dumps(datum) + "\n")
                    written += 1        
