import logging, gzip, json, argparse, torch
# from detm import DETM
from detm_dataloader import CustomDataloader

logger = logging.getLogger("apply_detm")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_01", required=True)
    parser.add_argument("--model_02", required=True)
    parser.add_argument("--log", required=True)
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

    with gzip.open(args.model_01, "rb") as ifd:
        model_01 = torch.load(ifd, map_location=torch.device(args.device))
    with gzip.open(args.model_02, "rb") as ifd:
        model_02 = torch.load(ifd, map_location=torch.device(args.device))

    logger.info(f"model 01 vocab size: {len(model_01.id2token)}; model 02 vocab size: {len(model_02.id2token)}")
    assert model_01.id2token == model_02.id2token
    id2token = model_01.id2token
    token2id = {v: k for k, v in model_01.id2token.items()}
    
    
    model_01.eval()
    model_02.eval()

    comp_dataloader = CustomDataloader(args, logger)
    _, rnn_input, num_times, num_train, num_eval = comp_dataloader.preprocess_data(args.input, None, 0, 
                                                                                   id2token=id2token, token2id=token2id)
    
    logger.info(f"current have {num_times} time windows,"  +
                f"{num_train} train instances and {num_eval} evaluation instances")                                                                        
    comp_batch_generator = comp_dataloader.batch_generator(len(token2id), is_train=False)

    try:
        while True:
            comp_data_batch, comp_normalized_data_batch, comp_times_batch, _ = next(comp_batch_generator)

            model_01(
                comp_data_batch, 
                comp_normalized_data_batch, 
                comp_times_batch, 
                rnn_input['eval'], 
                num_docs=num_eval, 
                is_train=False)
            model_02(
                comp_data_batch, 
                comp_normalized_data_batch, 
                comp_times_batch, 
                rnn_input['eval'], 
                num_docs=num_eval, 
                is_train=False)
            
    except StopIteration:
        pass
                
    except Exception as e:
        logger.info(f"encountering exception on apply execution: {str(e)}")
        raise Exception(e)

    model_01.log_stats(1, 0, logger, only_eval=True)
    model_02.log_stats(2, 0, logger, only_eval=True)
