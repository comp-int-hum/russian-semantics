import logging, argparse, gzip, torch, json
from detm import Corpus
from detm import annotate_data

logger = logging.getLogger('generate_auth_data')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--train_val_data_dir", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--author_field", type=str, default="author")
    parser.add_argument("--content_field", type=str, default="text")
    parser.add_argument("--title_field", type=str, default="title")
    parser.add_argument("--time_field", type=str, default="time")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not args.device:
       args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    elif args.device == "cuda" and not torch.cuda.is_available():
       logger.warning("Setting device to CPU because CUDA isn't available")
       args.device = "cpu"

    with gzip.open(args.model_dir, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(args.device), weights_only=False)
       
    corpus = Corpus()

    with gzip.open(args.train_val_data_dir, "rt") as ifd:
        for i, line in enumerate(ifd):
            corpus.append(json.loads(line))
        
    if args.test_data_dir:
        with gzip.open(args.train_val_data_dir, "rt") as ifd:
            for i, line in enumerate(ifd):
                corpus.append(json.loads(line))
    
    subdocs, times, auxiliaries = corpus.filter_for_model(model, args.content_field, args.time_field)
    auth_matrix = annotate_data(model, subdocs, times, auxiliaries,
                                args.author_field, args.title_field, 
                                batch_size=args.batch_size, device=args.device)
 
    with gzip.open(args.output_dir,  "wt", encoding="utf-8") as ofd:
        for auth_data in auth_matrix:
            json.dump(auth_data, ofd)
            ofd.write("\n")