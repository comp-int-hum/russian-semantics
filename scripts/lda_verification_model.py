import random, argparse, logging, gzip, math, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Dict, Set

# Dirichlet, hyperparams, tokenizer


def split_doc(tokens: List[str], max_length: int) -> List[List[str]]:
    num_subdocs = math.ceil(len(tokens) / max_length)
    retval = [
        tokens[idx * max_length : (idx + 1) * max_length] for idx in range(num_subdocs)
    ]
    return retval


# def generate_frequencies(data: List[str], max_docs: int = 10000) -> Counter[str]:
#     freqs: Counter[str] = Counter()
#     all_stopwords = sp.Defaults.stop_words
#     all_stopwords.add("enron")
#     nr_tokens = 0

#     for doc in tqdm(data[:max_docs]):
#         tokens = sp.tokenizer(doc)
#         for token in tokens:
#             token_text = token.text.lower()
#             if token_text not in all_stopwords and token.is_alpha:
#                 nr_tokens += 1
#                 freqs[token_text] += 1

#     return freqs


# def get_vocab(
#     freqs: Counter[str], freq_threshold: int = 3
# ) -> Tuple[Dict[str, int], Dict[int, str]]:
#     vocab: Dict[str, int] = {}
#     vocab_idx_str: Dict[int, str] = {}
#     vocab_idx = 0

#     for word in tqdm(freqs):
#         if freqs[word] >= freq_threshold:
#             vocab[word] = vocab_idx
#             vocab_idx_str[vocab_idx] = word
#             vocab_idx += 1

#     return vocab, vocab_idx_str


# def tokenize_dataset(
#     data: List[str], vocab: Dict[str, int], max_docs: int = 10000
# ) -> Tuple[List[List[str]], List[np.ndarray]]:
#     nr_tokens = 0
#     nr_docs = 0
#     docs = []

#     for doc in tqdm(data[:max_docs]):
#         tokens = sp.tokenizer(doc)

#         if len(tokens) > 1:
#             doc = []

#             for token in tokens:
#                 token_text = token.text.lower()
#                 if token_text in vocab:
#                     doc.append(token_text)
#                     nr_tokens += 1

#             nr_docs += 1
#             docs.append(doc)
#     print(f"Number of emails {nr_docs}")
#     print(f"Number of tokens: {nr_tokens}")

#     # Numericalize
#     corpus = []
#     for doc in tqdm(docs):
#         corpus_d = []

#         for token in doc:
#             corpus_d.append(vocab[token])

#         corpus.append(np.asarray(corpus_d))

#     return docs, corpus


# data = df.iloc[0:20000]["message"].sample(frac=0.5, random_state=42).values
# freqs = generate_frequencies(data, max_docs=10000)
# vocab, vocab_idx_str = get_vocab(freqs)
# docs, corpus = tokenize_dataset(data, vocab)

# vocab_size = len(vocab)
# print(f"vocab size: {vocab_size}")


# def LDA_Collapsed_Gibbs(corpus, num_iter=200):

#     # initialize counts and z
#     Z = []
#     num_docs = len(corpus)

#     for _, doc in enumerate(corpus):
#         # the initialized topics involved for each token
#         Zd = np.random.randint(low=0, high=NUM_TOPICS, size=(len(doc)))
#         Z.append(Zd)

#     # counter of topics per doc
#     ndk = np.zeros((num_docs, NUM_TOPICS))

#     for d in range(num_docs):
#         for k in range(NUM_TOPICS):
#             # counter for how many of doc d has topic k
#             ndk[d, k] = np.sum(Z[d] == k)

#     # word distribution for particular topic
#     nkw = np.zeros((NUM_TOPICS, vocab_size))
#     for doc_idx, doc in enumerate(corpus):
#         for i, word in enumerate(doc):
#             topic = Z[doc_idx][i]
#             nkw[topic, word] += 1

#     # how many words in each topic
#     nk = np.sum(nkw, axis=1)
#     topic_list = [i for i in range(NUM_TOPICS)]

#     # loop
#     for _ in tqdm(range(num_iter)):
#         for doc_idx, doc in enumerate(corpus):
#             for i in range(len(doc)):
#                 word = doc[i]
#                 topic = Z[doc_idx][i]

#                 # remove z_i because conditioned on z_(-i)
#                 ndk[doc_idx, topic] -= 1
#                 nkw[topic, word] -= 1
#                 nk[topic] -= 1

#                 p_z = (
#                     (ndk[doc_idx, :] + ALPHA)
#                     * (nkw[:, word] + BETA)
#                     / (nk[:] + BETA * vocab_size)
#                 )
#                 topic = random.choices(topic_list, weights=p_z, k=1)[0]

#                 # update n parameters
#                 Z[doc_idx][i] = topic
#                 ndk[doc_idx, topic] += 1
#                 nkw[topic, word] += 1
#                 nk[topic] += 1

#     return Z, ndk, nkw, nk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # all the i/o files
    parser.add_argument("--train", dest="train", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str)
    parser.add_argument("--log", dest="log", type=str, required=True)

    # filtering step:
    #   1. min and max time duration specification
    parser.add_argument("--min_time", type=int, default=None)
    parser.add_argument("--max_time", type=int, default=None)
    #   2. vocab filtering
    parser.add_argument("--min_word_occur", dest="min_word_occur", type=int, default=0)
    parser.add_argument(
        "--max_word_ratio", dest="max_word_ratio", type=float, default=1
    )
    #   2. subdocument divide
    parser.add_argument("--max_doclen", dest="max_doclen", type=int, default=200)

    # training
    parser.add_argument("--epochs", dest="epochs", type=int, default=400)
    parser.add_argument("--random_seed", dest="random_seed", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.0001)
    parser.add_argument("--lr_factor", type=float, default=2.0)
    parser.add_argument("--num_topics", type=int, default=50)

    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--reduce_rate", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)  # symmetric
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.log)
    logger = logging.getLogger("lda_verification_model")

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    vocab_counter: Counter = Counter()

    # step 1: compile all the documents and split into subdocs
    all_subdocs = []
    with gzip.open(args.train, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            if type(j["year"]) == int:
                time = int(j["year"])
                # try to eliminated unneeded instances right here
                if (args.min_time is None or time > args.min_time) and (
                    args.max_time is None or time < args.max_time
                ):

                    doc = []
                    full_text_words = j["content"].split()
                    vocab_counter.update(full_text_words)

                    subdocs = split_doc(doc, args.max_doclen)
                    all_subdocs.extend(subdocs)

        # step 2: filter based on min word occurence
        above_threshold_vocabs: Set[str] = set(
            [
                item
                for item, count in vocab_counter.items()
                if count >= args.min_word_occur
            ]
        )
        logger.info(
            f"there are a total of {len(vocab_counter)}, among which {len(above_threshold_vocabs)} are above the threshold"
        )
        filtered_all_subdocs = [
            [
                subdoc_token
                for subdoc_token in subdoc
                if subdoc_token in above_threshold_vocabs
            ]
            for subdoc in all_subdocs
        ]

        random.shuffle(filtered_all_subdocs)
        logger.info(f"the total number of subdocs are {len(filtered_all_subdocs)}")

    # Z, ndk, nkw, nk = LDA_Collapsed_Gibbs(corpus)
    # phi = nkw / nk.reshape(args.num_topics, 1)  # to get the probability distribution

    # for k in range(args.num_topics):
    #     most_common_words = np.argsort(phi[k])[::-1][:num_words]
    #     print(f"Topic {k} most common words")

    #     for word in most_common_words:
    #         print(vocab_idx_str[word])
    #     print("\n")
