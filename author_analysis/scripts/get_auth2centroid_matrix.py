import argparse, numpy, csv, torch, logging, gzip, itertools
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger("get auth2centroid matrix")

# centroid = numpy.array([centroid_data[(topic_id, vocab_id, win_id)] for vocab_id in vocab_ids])
# auth = numpy.array([auth_data[(auth_id, topic_id, vocab_id, win_id)] for vocab_id in vocab_ids])

# centroid /= centroid.sum()
# auth /= auth.sum()

# return jensenshannon(centroid, auth)

def data_generator(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in tqdm(reader):
            yield [int(float(ins)) for ins in row]

def get_vocab_batch(generator, cat=None):
    data = defaultdict(float)

    try:
        while True:
            instance = numpy.array(next(generator))
            if cat is None:
                cat = instance[[0, 2]]
            elif not numpy.array_equal(cat, instance[[0, 2]]):
                # put it back
                generator = itertools.chain([instance], generator)
                break
            data[instance[-3]] = instance[-1]
    except StopIteration:
        return cat, data, None
    
    return cat, data, generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_auth_dir', type=str, required=True)
    parser.add_argument('--input_centroid_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, default="js")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                        # filename='example.log')

    centroid_generator = data_generator(args.input_centroid_dir)
    auth_generator = data_generator(args.input_auth_dir)
    _, auth_metadata = next(centroid_generator), next(auth_generator)
    num_auth, num_topic, num_vocab, num_window =  auth_metadata[0], auth_metadata[1], auth_metadata[2], auth_metadata[3]
    logger.info(f"data is of size  num_auth, num_topic, num_vocab, num_win which is {num_auth}, {num_topic}, {num_vocab}, {num_window}")

    # set all value to 2.0 unless specified
    js_info = defaultdict(lambda : 2.0)

    while auth_generator and centroid_generator:
        cat, centroid_topic_counter, centroid_generator = get_vocab_batch(centroid_generator)
        logger.info(f"{cat}")
        logger.info(f"vocab number: {len(centroid_topic_counter)}, vocab total counter: {sum(centroid_topic_counter.values())}")
        exit(0)
    # for win_id in range(num_windows):
    #     for topic_id in range(num_topics):
    #         for auth_id in tqdm(range(num_auth)):
    #             # no instance inside, not applicable
    #                 vocab_ids = set()
    #                 for vocab_id in range(num_vocab):
    #                     if centroid_data[(topic_id, vocab_id, win_id)] > 0 or auth_data[(auth_id, topic_id, vocab_id, win_id)] > 0:
    #                         vocab_ids.add(vocab_id)
    
    #                 if len(vocab_ids) > 0:
    #                      logger.info(f"({auth_id}, {topic_id}, {win_id}): using a subset of {len(vocab_ids)} vocabs for js calculation")
                
                # if js_div != 2.0:
                #     logger.info(f"js divergence for auth {auth_id} topic {topic_id} win {win_id}: {js_div}")
                # js_matrix[auth_id][win_id][topic_id] = js_div