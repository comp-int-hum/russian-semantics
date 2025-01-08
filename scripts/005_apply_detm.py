import logging, gzip, math, json, argparse, torch
from detm import Dataset, DataLoader, Trainer

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
    parser.add_argument("--min_time", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument("--content_field", type=str, required=True)
    parser.add_argument("--time_field", type=str, required=True)
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

    all_window_ranges = [f"{args.min_time + idx * args.window_size}-" + 
                         f"{args.min_time + (idx + 1) * args.window_size if args.min_time + (idx + 1) * args.window_size <= args.max_time else args.max_time}" 
                                  for idx in range(math.ceil((args.max_time - args.min_time) / args.window_size))]

    trainer = Trainer(logger)
    token2id = trainer.load_model(args.model, args.device)
    
    dataset = Dataset(args.input, None, -1)
    dataset.preprocess_data(args.min_time, args.max_time, args.window_size,
                            args.content_field, args.time_field,
                            max_subdoc_length=args.max_subdoc_length, 
                            min_word_occurrance=args.min_word_occurrence, 
                            max_word_proportion=args.max_word_proportion,
                            logger=logger, word2id=token2id)

    a_subdocs, a_times, a_auxiliaries = dataset.get_data(is_train=False)
    num_appl = len(a_times)
    appl_dataloader = DataLoader(a_subdocs, a_times, a_auxiliaries, 
                                 args.batch_size, args.device,
                                 all_window_ranges)
    
    logger.info(f"current have {len(all_window_ranges)} time windows and {num_appl} application instances")                                                                       
    del dataset, a_subdocs, a_times, a_auxiliaries

    subdoc_topics = trainer.apply_model(appl_dataloader, len(all_window_ranges), num_appl, args.random_seed)

    with gzip.open(args.output, "wt") as ofd:
        for sd in subdoc_topics:
            ofd.write(json.dumps(sd) + "\n")
