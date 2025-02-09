import logging, argparse, gzip, torch, json, numpy
from detm import Corpus
from detm import annotate_data
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger('generate_vocab_data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    # input data fields
    parser.add_argument("-vf", "--vocab_field", type=str, default="vocab")
    parser.add_argument("-if", "--vocab2id_field", type=str, default="vocab2id")

    # auxiliary info
    parser.add_argument("-e", "--embed_size", type=int, default=None)

    args = parser.parse_args()
    assert args.output_dir.endswith('.csv')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with gzip.open(args.model_dir, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device("cpu"), weights_only=False)
    
    vocab_list = model.word_list
    del model
    id2vocab = {k: v for k, v in enumerate(vocab_list)}
    vocab2id = {v: k for k, v in id2vocab.items()}
    vocab_topic_matrix = numpy.zeros((len(id2vocab), args.embed_size))

    # read input
    if args.input_dir.endswith('.gz'):
        ungziped_input_dir : str = args.input_dir[:-3]
        if ungziped_input_dir.endswith('.pkl'):
            with gzip.open(args.input_dir, "rb") as ifd:
                data = pickle.load(ifd)
        elif ungziped_input_dir.endswith('.jsonl'):
            data = []
            with gzip.open(args.input_dir, "rt") as ifd:
                for i, line in tqdm(enumerate(ifd)):
                    data.append(json.loads(line))
                
        else:
            raise Exception("unidentifiable source of input")
    
    elif args.input_dir.endswith('.csv'):
        data = numpy.loadtxt(args.input_dir, delimiter=',')
    else:
        raise Exception("unidentifiable source of input")

    logger.info(f"there are {len(data)} unique author instances")

    # try to parse input
    if isinstance(data, list):

        for data_instance in tqdm(data):
            auth_vocab2id = data_instance[args.vocab2id_field]
            auth_id2vocab = {v : k for k, v in auth_vocab2id.items()}
            vocab_embed = numpy.array(data_instance[args.vocab_field]).sum(1)
            logger.info(f"embed size: {vocab_embed.shape} word2id size: {len(auth_vocab2id)}")
            for auth_id in range(vocab_embed.shape[0]):
                vocab_id = vocab2id[auth_id2vocab[auth_id]]
                vocab_topic_matrix[vocab_id] += vocab_embed[auth_id]

    elif isinstance(data, dict):
        raise Exception("unimplemented")
    
    numpy.savetxt(args.output_dir, vocab_topic_matrix, delimiter=',')