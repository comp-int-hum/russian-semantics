import logging, gzip, json, argparse, torch, pickle, re, math, numpy
from tqdm import tqdm
from detm import Corpus, get_matrice

logger = logging.getLogger("generate_matrice")

def split_into_subdocs(text, max_subdoc_length, lowercase):
    tokens = re.split(r"\s+", text.lower() if lowercase else text)
    num_subdocs = math.ceil(len(tokens) / max_subdoc_length)
    retval = []
    for i in range(num_subdocs):
        retval.append(tokens[i * max_subdoc_length : (i + 1) * max_subdoc_length])
    return retval


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input_t", type=str, default=None)
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--time_field", dest="time_field", default="time")
    parser.add_argument("--content_field", dest="content_field", default="content")
    parser.add_argument("--use_raw_data", action="store_true", default=False)
    parser.add_argument("--get_prob", action="store_true", default=False)
    parser.add_argument('--device')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--max_subdoc_length', type=int, default=100)
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
    if args.use_raw_data:
        with gzip.open(args.input, "rt") as ifd:
            for line in tqdm(ifd):
                j = json.loads(line)
                j['text'] = split_into_subdocs(j['text'], args.max_subdoc_length, lowercase=True)
                corpus.append(j)
    else:
        # this should include both train, val and test
        with gzip.open(args.input, "rt") as ifd:
            for line in tqdm(ifd):
                corpus.append(json.loads(line))
            
        if args.input_t:
            with gzip.open(args.input_t, "rt") as ifd:
                for line in tqdm(ifd):
                    corpus.append(json.loads(line))

    subdocs, times, auxiliaries = corpus.filter_for_model(model, args.content_field, args.time_field)

    model = model.to(args.device)
    matrice = get_matrice(model, subdocs, times, auxiliaries,
                          batch_size=args.batch_size, output_dir=args.output,
                          workid_field="htid", time_field="written_year", 
                          author_field="author_info", workname_field="title",
                          logger=logger, get_prob=args.get_prob) 

    with gzip.open(args.output, "wb") as ofd:
        ofd.write(pickle.dumps(matrice))