import argparse, numpy, csv, logging, gzip, itertools
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger("get auth2centroid matrix")


def calculate_js(top_data, auth_data):
    vocab_ids = set()
    for vocab_id in range(num_vocab):
        if top_data[vocab_id] > 0 or auth_data[vocab_id] > 0:
            vocab_ids.add(vocab_id)

    if len(vocab_ids) > 0:

        top_centroid = numpy.array(
            [top_data[vocab_id] for vocab_id in vocab_ids], dtype=float
        )
        top_auth = numpy.array(
            [auth_data[vocab_id] for vocab_id in vocab_ids], dtype=float
        )

        top_centroid /= top_centroid.sum()
        top_auth /= top_auth.sum()

        return jensenshannon(top_centroid, top_auth)

    return 2.0


def data_generator(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield [int(float(ins)) for ins in row]


def get_data_batch(generator, vocab_num, is_auth=False):
    data = defaultdict(float)
    cat = None
    try:
        while True:
            instance = numpy.array(next(generator))
            if cat is None:
                cat = instance[[0, 1, 3]] if is_auth else instance[[0, 2]]
            if (instance[2] if is_auth else instance[1]) == vocab_num:
                break
            data[instance[-3]] = instance[-1]
    except StopIteration:
        return cat, data

    return cat, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_auth_dir", type=str, required=True)
    parser.add_argument("--input_centroid_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, default="js")
    args = parser.parse_args()
    logging.basicConfig(
        # format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        # filename="example.log",
    )

    centroid_generator = data_generator(args.input_centroid_dir)
    auth_generator = data_generator(args.input_auth_dir)
    _, auth_metadata = next(centroid_generator), next(auth_generator)
    num_auth, num_topic, num_vocab, num_window = (
        auth_metadata[0],
        auth_metadata[1],
        auth_metadata[2],
        auth_metadata[3],
    )

    logger.info(
        f"data is of size (num_auth, num_topic, num_vocab, num_win) which is ({num_auth}, {num_topic}, {num_vocab}, {num_window})"
    )

    # set all value to 2.0 unless specified
    js_matrix = numpy.full((num_auth, num_topic, num_window), 2)
    past_tw = -1
    while True:
        top_cat, top_counter = get_data_batch(centroid_generator, num_vocab)
        if top_cat is None:
            break
        if past_tw < top_cat[1]:
            past_tw = top_cat[1]
            logger.info(f"---- iterating time window {top_cat[1]} ---- ")

        for auth_id in range(num_auth):
            auth_cat, auth_counter = get_data_batch(
                auth_generator, num_vocab, is_auth=True
            )
            if len(auth_counter) > 0:
                js_div = calculate_js(top_counter, auth_counter)
                if js_div < 2:
                    js_matrix[auth_cat[0]][auth_cat[1]][auth_cat[2]] = js_div

    numpy.save(args.output_dir, js_matrix)
    logger.info(f"saved the matrix to {args.output_dir}")
