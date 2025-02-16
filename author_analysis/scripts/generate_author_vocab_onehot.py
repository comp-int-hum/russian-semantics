import argparse, logging, json, gzip, torch, numpy, csv
from tqdm import tqdm
from collections import defaultdict, Counter

logger = logging.getLogger("generate author vocab onehot")


def sorted_auth_iterator(auth_mat):
    # Sort by (win_id, topic_id, auth_id, vocab_id)
    for key, value in sorted(
        auth_mat.items(), key=lambda x: (x[0][3], x[0][1], x[0][0], x[0][2])
    ):
        yield key + (value,)


def sorted_topic_iterator(topic_mat):
    # Sort by (win_id, topic_id, vocab_id)
    for key, value in sorted(
        topic_mat.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])
    ):
        yield key + (value,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--auth_output_dir", type=str, required=True)
    parser.add_argument("--topic_output_dir", type=str, required=True)
    parser.add_argument("--embed_field", type=str, default="embedding")
    parser.add_argument("--word2id_field", type=str, default="vocab2idx")
    parser.add_argument("--work2id_field", type=str, default="work2idx")
    parser.add_argument("--divisor", type=str, default=" [-] ")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    args.device = (
        args.device
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    with gzip.open(args.model_dir, "rb") as ifd:
        model = torch.load(
            ifd, map_location=torch.device(args.device), weights_only=False
        )

    word_list, num_topics, num_windows = (
        model.word_list,
        model.num_topics,
        model.num_windows,
    )

    id2word = {k: v for k, v in enumerate(word_list)}
    word2id = {v: k for k, v in id2word.items()}

    auth_mat = defaultdict(float)
    topic_mat = defaultdict(float)

    win_counter = Counter()

    num_auth, num_words = 0, len(word_list)

    with gzip.open(args.input_dir, "rt") as ifd:
        for auth_id, line in tqdm(enumerate(ifd)):
            num_auth += 1
            data = json.loads(line)
            embed_data = numpy.array(data[args.embed_field])
            localword2localid, work2id = (
                data[args.word2id_field],
                data[args.work2id_field],
            )
            num_vocab, num_work, num_topic = embed_data.shape
            id2work = {v: k for k, v in work2id.items()}
            localid2id = {
                localword2localid[w]: word2id[w] for w, _ in localword2localid.items()
            }
            win2works = [[] for _ in range(num_windows)]

            for work_id in range(num_work):
                work_time = int(id2work[work_id].split(args.divisor)[1])
                win_id = model.represent_time(work_time)
                win_counter[win_id] += 1

                for localvocab_id, vocab_id in localid2id.items():
                    for topic_id in range(num_topic):
                        per_vocab_top_counter = embed_data[
                            localvocab_id, work_id, topic_id
                        ]
                        if per_vocab_top_counter > 0:
                            auth_mat[
                                (auth_id, topic_id, vocab_id, win_id)
                            ] += per_vocab_top_counter
                            topic_mat[
                                (topic_id, vocab_id, win_id)
                            ] += per_vocab_top_counter
            # this is here to ensure that it has a proxy for knowing where to end
            for win_id in range(num_windows):
                for topic_id in range(num_topics):
                    auth_mat[((auth_id, topic_id, num_words, win_id))] += 1
    logger.info(f"{win_counter}")
    for win_id in range(num_windows):
        for topic_id in range(num_topics):
            topic_mat[(topic_id, num_words, win_id)] += 1

    with open(args.auth_output_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["author_id", "topic_id", "vocab_id", "window_id", "counter"])
        writer.writerow([num_auth, num_topics, num_words, num_windows, -1])
        for d1, d2, d3, d4, value in sorted_auth_iterator(auth_mat):
            writer.writerow([d1, d2, d3, d4, value])

    with open(args.topic_output_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic_id", "vocab_id", "window_id", "counter"])
        writer.writerow([num_topics, num_words, num_windows, -1])
        for d1, d2, d3, value in sorted_topic_iterator(topic_mat):
            writer.writerow([d1, d2, d3, value])
