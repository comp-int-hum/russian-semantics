import argparse, logging, gzip, pickle, json, numpy
from tqdm import tqdm

logger = logging.getLogger("order_by_auth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-ia", "--id2auth_dir", type=str, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-oa", "--id2auth_output_dir", type=str, default=None)

    # input data fields
    parser.add_argument("-af", "--author_field", type=str, default="author")
    parser.add_argument("-ef", "--embedding_field", type=str, default="embeddings")
    parser.add_argument("-if", "--id2author_field", type=str, default="id2author")

    # auxiliary info
    parser.add_argument("-e", "--embed_size", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-6)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # aither the id2auth is known or needs to be extracted from pickle
    # assert not args.id2auth_dir or not args.id2auth_output_dir
    assert args.output_dir.endswith('.csv')

    # read input
    if args.input_dir.endswith('.gz'):
        ungziped_input_dir : str = args.input_dir[:-3]
        with gzip.open(args.input_dir, "rb") as ifd:
            
            if ungziped_input_dir.endswith('.pkl'):
              data = pickle.loads(ifd.read())
            
            elif ungziped_input_dir.endswith('.jsonl'):
                data = []
                for i, line in tqdm(enumerate(ifd)):
                    data.append(json.loads(line))
            
            else:
                raise Exception("unidentifiable source of input")
    
    elif args.input_dir.endswith('.csv'):
        data = numpy.loadtxt(args.input_dir, delimiter=',')
    
    if args.id2auth_dir.endswith('.json'):
        with open(args.id2auth_dir, 'r') as f:
            id2auth = json.load(f)
    else:
        # must be a dictionary to get the auth2id data
        assert isinstance(data, dict)
        id2auth = data[args.id2author_field]

    id2auth = {int(k): v for k, v in id2auth.items()}
    id2auth = dict(sorted(id2auth.items()))
    auth2id = {v: k for k, v in id2auth.items()}

    if args.id2auth_output_dir.endswith('.json'):
        with open(args.id2auth_output_dir, 'w') as f:
            json.dump(id2auth, f, indent=4, ensure_ascii=False)

    # the first is for ordering by embed_size,
    # the second is for ordering by auth2auth matrix

    # try to parse input
    if isinstance(data, list):
        # it it is a list, then needs to extract the embedding of each 
        # and correspond to the author idx
        numpy_embeds = numpy.zeros((len(id2auth), args.embed_size))

        check_bool = [False] * len(auth2id)
        for data_instance in tqdm(data):
            author_name = data_instance[args.author_field]
            if author_name in auth2id:
                numpy_embeds[auth2id[author_name]] = numpy.array(data_instance[args.embedding_field])
                check_bool[auth2id[author_name]] = True
            else:
                logger.info(f"unable to find author name {author_name}, skipping... ")
        
        missing_authors = [author for author, included in zip(id2auth.values(), check_bool) if not included]
        logger.info(f"Completes list, missing {len(missing_authors)} authors: {missing_authors}")

    elif isinstance(data, dict):
        # if it is a dict, most likely it comes directly out of matrice
        # get the distribution of topics per author from the author_window_topic column
        author_window_topic = data[args.author_field]
        numpy_embeds = numpy.array(((author_window_topic.transpose(2, 0, 1) / 
                                     (author_window_topic.sum(2) + args.epsilon)).transpose(1, 2, 0)).sum(1))
    
    numpy.savetxt(args.output_dir, numpy_embeds, delimiter=',')