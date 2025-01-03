import logging, gzip, math, json, argparse, torch, os
# from gensim.models import Word2Vec
# import numpy as np
from detm import DETM
from detm_dataloader import CustomDataloader

logger = logging.getLogger("apply_detm")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_subdoc_length", type=int, default=200)
    parser.add_argument("--min_word_occurrence", type=int, default=0)
    parser.add_argument("--max_word_proportion", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--device")  # , choices=["cpu", "cuda"], help='')
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--min_time", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument("--batch_preprocess", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=args.log,
    )

    if not args.device:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("Setting device to CPU because CUDA isn't available")
        args.device = "cpu"
    
    if not args.eval_batch_size:
        args.eval_batch_size = args.batch_size

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(args.device))

    token2id = {v: k for k, v in model.id2token.items()}
    id2token = model.id2token
    
    model.eval()

    apply_dataloader = CustomDataloader(args, logger)
    _, rnn_input, num_times, num_train, num_eval = apply_dataloader.preprocess_data(args.input, None, 0, 
                                                                                    id2token=id2token, token2id=token2id)
    
    logger.info(f"current have {num_times} time windows, {num_train} train instances and {num_eval} evaluation instances")                                                                        
    appl_batch_generator = apply_dataloader.batch_generator(len(token2id), is_train=False)

    try:
        while True:
            appl_data_batch, appl_normalized_data_batch, appl_times_batch, inds = next(appl_batch_generator)
            out = model(
                appl_data_batch, 
                appl_normalized_data_batch, 
                appl_times_batch, 
                rnn_input['eval'], 
                num_docs=num_eval, 
                is_train=False, get_lik=True
                )

            apply_dataloader.update_subdoc_counts(out, inds, token2id)
            
    except StopIteration:
        pass
                
    except Exception as e:
        logger.info(f"encountering exception on apply execution: {str(e)}")
        raise Exception(e)


    with gzip.open(args.output, "wt") as ofd:
        for sd in apply_dataloader.subdoc_counts['eval']:
            ofd.write(json.dumps(sd) + "\n")
