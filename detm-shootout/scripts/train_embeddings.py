import logging
import gzip
import math
import json
import argparse
import re
import random
from detm import Corpus, train_embeddings, save_embeddings

logger = logging.getLogger("train_embeddings")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Input, already split into sentences")
    parser.add_argument("--content_field", dest="content_field", help="", required=True)
    parser.add_argument("--output", dest="output", help="Model output")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="How long to train")
    parser.add_argument("--window_size", dest="window_size", type=int, default=5, help="Skip-gram window size")
    parser.add_argument("--embedding_size", dest="embedding_size", type=int, default=300, help="Embedding size")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.random_seed:
        random.seed(args.random_seed)

    corpus = Corpus()
    
    with gzip.open(args.input, "rt") as ifd:
        for line in ifd:
            corpus.append(json.loads(line))

    embs = train_embeddings(corpus, args.content_field, random_seed=args.random_seed, epochs=args.epochs)
    save_embeddings(embs, args.output)
    
