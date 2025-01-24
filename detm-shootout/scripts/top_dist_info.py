import logging
import gzip
import json
import argparse
import numpy
import torch
import pickle
from detm import get_top_topic_info

logger = logging.getLogger("get_dist_info")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Data file")
    parser.add_argument("--output", dest="output", help="File to save stats to", required=True)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with gzip.open(args.input, "rb") as ifd:
        precomp = pickle.loads(ifd.read())

    data = get_top_topic_info(precomp)

    with open(args.output, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)