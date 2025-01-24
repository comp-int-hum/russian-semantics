import logging
import gzip
import json
import argparse
import numpy
import torch
from detm import Corpus, apply_model


logger = logging.getLogger("apply_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Data file")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--time_field", dest="time_field", default="time")
    parser.add_argument("--content_field", dest="content_field", default="content")
    parser.add_argument('--device')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    args = parser.parse_args()

    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    if not args.device:
       args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    elif args.device == "cuda" and not torch.cuda.is_available():
       logger.warning("Setting device to CPU because CUDA isn't available")
       args.device = "cpu"

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(args.device), weights_only=False)
       
    corpus = Corpus()
    with gzip.open(args.input, "rt") as ifd:
        for i, line in enumerate(ifd):
            corpus.append(json.loads(line))

    subdocs, times, _ = corpus.filter_for_model(model, args.content_field, args.time_field)
            
    model = model.to(args.device)
    ppl = apply_model(
        model,
        subdocs,
        times,
        args.batch_size
    )

    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps({"test_perplexity" : ppl}) + "\n")
