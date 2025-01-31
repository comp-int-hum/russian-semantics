import argparse, gzip, pickle, json
from detm import load_embeddings


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-ia", "--id2author", required=True)
    parser.add_argument("-iw", "--id2word", required=True)
    parser.add_argument("-e", "--embedding_dir", required=True)

    parser.add_argument("-a", "--auth_word_top", default="auth_word_top")

    args = parser.parse_args()

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
    
    with open(args.id2author, 'r') as f:
        id2author = json.load(f)
    
    with open(args.id2word, 'r') as f:
        id2word = json.load(f)

    embedding = load_embeddings(args.embedding_dir)

    id2author = {int(k) : v for k, v in id2author.items()}
    id2word = {int(k) : v for k, v in id2word.items()}
    id2embed = {k : embedding[v] for k, v in id2word.items()}
    auth_word_top = data[args.auth_word_top]
    del data, embedding, id2word
    