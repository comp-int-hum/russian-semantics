import copy, gzip, math, argparse, torch, logging, json
from gensim.models import Word2Vec, KeyedVectors
from tqdm import trange
import numpy as np
from detm import Dataset, DataLoader, DETM, Trainer
from detm import load_embeddings, filter_embeddings

logger = logging.getLogger("train_detm")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval")
    parser.add_argument("--embeddings", help="Embeddings file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--max_subdoc_length", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--min_word_occurrence", type=int, default=0)
    parser.add_argument("--max_word_proportion", type=float, default=1.0)
    parser.add_argument("--top_words", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--num_words", type=int, default=20)
    parser.add_argument("--log_intereval", type=int, default=10)
    parser.add_argument("--visualize_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--load_from", type=str, default="")
    parser.add_argument("--tc", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--lr_factor", type=float, default=2.0)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--device", type=str, default=None)  # choices=["cpu", "cuda"], help='')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--enc_drop", type=float, default=0.0) # dropout rate on encoder
    parser.add_argument("--eta_dropout", type=float, default=0.0) # dropout rate on rnn for eta
    parser.add_argument("--clip", type=float, default=2.0) # gradient clipping
    parser.add_argument("--nonmono", type=int, default=10) # number of bad hits allowed
    parser.add_argument("--wdecay", type=float, default=1.2e-6) # some l2 regularization
    parser.add_argument("--anneal_lr", type=int,default=0) # whether to anneal the learning rate or not
    parser.add_argument("--bow_norm", type=int, default=1) # normalize the bows or not
    parser.add_argument("--limit_docs", type=int)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_topics", type=int, default=50) # number of topics
    parser.add_argument("--rho_size", type=int, default=300) # dimension of rho
    parser.add_argument("--emb_size", type=int, default=300) # dimension of embeddings
    parser.add_argument("--t_hidden_size", type=int, default=800) # dimension of hidden space of q(theta)
    parser.add_argument("--theta_act", type=str, default="relu") # tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu
    parser.add_argument("--train_embeddings", default=False, action="store_true") # whether to fix rho or train it
    parser.add_argument("--eta_nlayers", type=int, default=3) # number of layers for eta
    parser.add_argument("--eta_hidden_size", type=int, default=200) # number of hidden units for rnn
    parser.add_argument("--delta", type=float, default=0.005, help="prior variance")
    parser.add_argument("--train_proportion", type=float, default=0.7, help="")
    parser.add_argument("--min_time", type=int, default=None)
    parser.add_argument("--max_time", type=int, default=None)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--reduce_rate", type=int, default=5)
    parser.add_argument("--batch_preprocess", action='store_true')
    parser.add_argument("--content_field", type=str, required=True)
    parser.add_argument("--time_field", type=str, required=True)
    args = parser.parse_args()

    args.device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    args.eval_batch_size = (
        args.batch_size if args.eval_batch_size is None else args.eval_batch_size
    )

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=args.log,
    )

    all_window_ranges = [f"{args.min_time + idx * args.window_size}-" + 
                         f"{args.min_time + (idx + 1) * args.window_size if args.min_time + (idx + 1) * args.window_size <= args.max_time else args.max_time}" 
                                  for idx in range(math.ceil((args.max_time - args.min_time) / args.window_size))]

    dataset = Dataset(args.train, args.eval, args.train_proportion)
    word_list = dataset.preprocess_data(args.min_time, args.max_time, args.window_size,
                                       args.content_field, args.time_field,
                                        max_subdoc_length=args.max_subdoc_length, 
                                        min_word_occurrance=args.min_word_occurrence, 
                                        max_word_proportion=args.max_word_proportion,
                                        logger=logger)
    
    t_subdocs, t_times, t_auxiliaries, t_time_counter = dataset.get_data(is_train=True)
    num_windows, num_train = len(t_time_counter), len(t_times)
    train_dataloader = DataLoader(t_subdocs, t_times, t_auxiliaries, 
                                  args.batch_size, args.device,
                                  all_window_ranges,
                                  time_counter=t_time_counter)
    
    e_subdocs, e_times, e_auxiliaries = dataset.get_data(is_train=False)
    num_eval = len(e_times)
    eval_dataloader = DataLoader(e_subdocs, e_times, e_auxiliaries, 
                                 args.eval_batch_size, args.device,
                                 all_window_ranges)
    
    del dataset, t_subdocs, t_times, t_auxiliaries, t_time_counter, e_subdocs, e_times, e_auxiliaries

    if args.embeddings:
        embeddings = load_embeddings(args.embeddings)
        embeddings = filter_embeddings(embeddings, word_list)
        
    logger.info("----- loaded embeddings ----- ")

    trainer = Trainer(logger)
    logger.info("----- initialized trainer ----- ")

    trainer.init_model(embeddings=embeddings, word_list=word_list,
                       num_windows=num_windows, num_topics=args.num_topics, 
                       min_time=args.min_time, max_time=args.max_time,
                       t_hidden_size=args.t_hidden_size,
                       eta_hidden_size=args.eta_hidden_size,
                       enc_drop=args.enc_drop, eta_dropout=args.eta_dropout,
                       eta_nlayers=args.eta_nlayers, delta=args.delta,
                       window_size=args.window_size, train_embeddings=args.train_embeddings,
                       theta_act=args.theta_act, 
                       batch_size=args.batch_size, device=args.device)
    
    trainer.init_training_params(num_train=num_train, num_eval=num_eval,
                                learning_rate=args.learning_rate, wdecay=args.wdecay, 
                                clip=args.clip, reduce_rate=args.reduce_rate, 
                                lr_factor=args.lr_factor, early_stop=args.early_stop)
    
    logger.info("----- initialized model ----- ")

    train_rnn = train_dataloader.get_rnn(num_windows, len(word_list))
    eval_rnn = eval_dataloader.get_rnn(num_windows, len(word_list))

    logger.info("----- initialized rnn_input ----- ")

    logger.info("----- starts training ----- ")

    for epoch in trange(args.epochs):
        
        trainer.start_epoch()
        train_batch_generator = train_dataloader.batch_generator(len(word_list), epoch, logger=logger)
        trainer.train_model(train_batch_generator, train_rnn)

        logger.info("Computing perplexity...")
        eval_batch_generator = eval_dataloader.batch_generator(len(word_list), epoch, logger=logger)
        trainer.eval_model(eval_batch_generator, eval_rnn)
        continue_flag = trainer.end_epoch()

        if continue_flag is False:
            break

    with gzip.open(args.output, "wb") as ofd:
        torch.save(trainer.get_best_model(), ofd)