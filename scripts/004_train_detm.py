import copy, gzip, math, argparse, torch, logging, json
from gensim.models import Word2Vec, KeyedVectors
from tqdm import trange
import numpy as np
from detm import DETM
from detm_dataloader import CustomDataloader

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

    dataloader = CustomDataloader(args, logger)
    id2token, rnn_input, num_times, num_train, num_eval = dataloader.preprocess_data(args.train, args.eval, args.train_proportion)
    
    if args.embeddings:
        if args.embeddings.endswith("txt"):
            wv = {}
            with open(args.embeddings, "rt") as ifd:
                for line in ifd:
                    toks = line.split()
                    wv[toks[0]] = list(map(float, toks[1:]))
            embeddings = torch.tensor([wv[id2token[i]] for i in range(len(id2token))])

        else:
            # error of failure to allocate at this step
            w2v = Word2Vec.load(args.embeddings)  
            # test_model = KeyedVectors.load(args.embeddings)
            # words = list(test_model.wv.key_to_index.keys())
            embeddings = torch.tensor(
                np.array([w2v.wv[id2token[i]] for i in range(len(id2token))])
            )
        
    logger.info("----- loaded embeddings ----- ")

    args.embeddings_dim = embeddings.size()
    args.num_times = num_times
    args.vocab_size = len(id2token)
    args.train_embeddings = 0

    detm_model = DETM(
        args,
        id2token,
        args.min_time,
        embeddings=embeddings,
    )

    logger.info("----- initialized model ----- ")

    detm_model.to(args.device)
    optimizer = torch.optim.Adam(
        detm_model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay
    )

    logger.info("----- starts training ----- ")

    best_state = None
    best_eval_ppl = None
    since_annealing = 0
    since_improvement = 0
    for epoch in trange(1, args.epochs + 1):
        logger.info("Starting epoch %d", epoch)

        detm_model.start_epoch()
        detm_model.train()

        train_batch_generator = dataloader.batch_generator(args.vocab_size, by_category=args.batch_preprocess)

        try:
            while True:  
                train_data_batch, train_normalized_data_batch, train_times_batch, _ = next(train_batch_generator)
                loss = detm_model(train_data_batch, train_normalized_data_batch, 
                                  train_times_batch, rnn_input["train"], 
                                  num_train)
                
                if not torch.any(torch.isnan(loss)):
                    loss.backward()
                    if args.clip > 0:
                        torch.nn.utils.clip_grad_norm_(detm_model.parameters(), args.clip)
                    optimizer.step()

        except StopIteration:
            pass

        logger.info("Computing perplexity...")

        detm_model.eval()
        with torch.no_grad():
            eval_acc_loss = 0.0
            eval_cnt = 0

            eval_batch_generator = dataloader.batch_generator(args.vocab_size,
                                                           is_train=False, by_category=args.batch_preprocess)
            
            try:
                while True:
                    eval_data_batch, eval_normalized_data_batch, eval_times_batch, _ = next(eval_batch_generator)
                    detm_model(eval_data_batch, eval_normalized_data_batch, 
                               eval_times_batch, rnn_input["eval"], 
                               is_train=False)
            except StopIteration:
                pass
        
        eval_ppl = detm_model.log_stats(epoch, optimizer.param_groups[0]["lr"], logger)

        if best_eval_ppl == None or eval_ppl < best_eval_ppl:
            logger.info("Copying new best model...")
            best_eval_ppl = eval_ppl
            best_state = copy.deepcopy(detm_model.state_dict())
            since_improvement = 0
            logger.info("Copied.")
        else:
            since_improvement += 1
        since_annealing += 1
        if (
            since_improvement > args.reduce_rate and since_annealing > args.reduce_rate
        ):
            optimizer.param_groups[0]["lr"] /= args.lr_factor
            detm_model.load_state_dict(best_state)
            since_annealing = 0
        elif since_improvement >= args.early_stop:
            break    

    detm_model.load_state_dict(best_state)
    with gzip.open(args.output, "wb") as ofd:
        torch.save(detm_model, ofd)
