import random
import logging
import gzip
import json
import argparse
import numpy
import torch
import math
from detm import Corpus, train_model, load_embeddings
import detm


logger = logging.getLogger("train_model")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", dest="train", help="Data file")
    parser.add_argument("--val", dest="val", help="Data file")
    parser.add_argument("--embeddings", dest="embeddings", help="Embeddings file")
    parser.add_argument("--time_field", dest="time_field", help="")
    parser.add_argument("--content_field", dest="content_field", help="")
    parser.add_argument("--min_time", type=int, default=None)
    parser.add_argument("--max_time", type=int, default=None)
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--window_size", dest="window_size", type=int, default=20, help="")
    parser.add_argument("--num_window", dest="num_window", type=int, default=-1)
    parser.add_argument("--min_word_count", dest="min_word_count", type=int, default=0, help="Words occuring less than this number of times throughout the entire dataset will be ignored")
    parser.add_argument("--max_word_proportion", dest="max_word_proportion", type=float, default=1.0, help="Words occuring in more than this proportion of documents will be ignored (probably conjunctions, etc)")    
    
    parser.add_argument("--top_words", dest="top_words", type=int, default=10, help="Number of words to show for each topic in the summary file")
    parser.add_argument("--max_epochs", dest="max_epochs", type=int, default=100, help="How long to train")
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")

    parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
    parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
    parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
    parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
    parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
    parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')
    
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=2.0, help='divide learning rate by this')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')
    parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
    parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

    parser.add_argument('--limit_docs', type=int, help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
    parser.add_argument('--train_embeddings', default=False, action="store_true", help='whether to fix rho or train it')
    parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
    parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
    parser.add_argument('--delta', type=float, default=0.005, help='prior variance')
    parser.add_argument('--train_proportion', type=float, default=0.7, help='')
    parser.add_argument("--model_type", dest="model_type", required=True)
    args = parser.parse_args()

    # either specify the number of window wanted or window required
    assert args.num_window == -1 or args.window_size == -1
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model_class = getattr(detm, args.model_type)
    
    if args.random_seed:
        random.seed(args.random_seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    elif args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("Setting device to CPU because CUDA isn't available")
        args.device = "cpu"

    torch.set_default_device(args.device)
    corpus = Corpus()

    with gzip.open(args.train, "rt") as ifd:
        for i, line in enumerate(ifd):
            corpus.append(json.loads(line))

    subdocs, times, word_list, time_reg = corpus.get_filtered_subdocs(
        content_field=args.content_field,
        time_field=args.time_field,
        time_reg= (args.min_time, args.max_time),
        min_word_count=args.min_word_count,
        max_word_proportion=args.max_word_proportion,
    )

    logger.info(f"specified time duration: {time_reg}")
    

    if args.window_size == -1:
        # try to get the window size
        args.window_size = math.ceil((time_reg[1] - time_reg[0]) / args.num_window)
    time_range = [(time_reg[0] + idx * args.window_size, 
                   time_reg[0] + (idx + 1) * args.window_size if time_reg[0] + (idx + 1) * args.window_size < time_reg[1] else time_reg[1])
                   for idx in range(math.ceil((time_reg[1] - time_reg[0]) / args.window_size))]
    logger.info(f"setting windows to be {time_range}")

    embeddings = load_embeddings(args.embeddings)
    
    model = model_class(
        word_list=word_list,
        num_topics= args.num_topics,
        window_size= args.window_size,
        min_time= time_reg[0],
        max_time= time_reg[1],
        embeddings=embeddings,
    )
    
    model.to(args.device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.wdecay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate, 
            weight_decay=args.wdecay)
    else:
        raise Exception('Unimplemented')

    best_state = train_model(
        model=model,        
        subdocs=subdocs,
        times=times,
        optimizer=optimizer,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=args.device,
        detect_anomalies=False
    )
    
    # model.load_state_dict(best_state)
    
    # with gzip.open(args.output, "wb") as ofd:
    #     torch.save(model, ofd)
