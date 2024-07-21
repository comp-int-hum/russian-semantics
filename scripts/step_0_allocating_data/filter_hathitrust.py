from demarc import Record
import random
import logging
import gzip
import json
import re
import argparse


logger = logging.getLogger("filter_hathitrust")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hathitrust_marc", dest="hathitrust_marc", help="HathiTrust MARC file")
    parser.add_argument("--language", dest="language", required=True)
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.hathitrust_marc, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for line in ifd:
            j = json.loads(line)
            rec = Record(j)
            if rec.language == args.language:
                ofd.write(line)
