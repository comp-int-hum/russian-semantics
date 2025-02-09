import argparse, logging, json, gzip, torch, numpy
from detm import load_embeddings
from tqdm import tqdm

logger = logging.getLogger('generate per topic vocab embed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-ia", "--id2auth_dir", type=str, required=True)
    parser.add_argument("-e", "--embedding_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-t", "--topic_spec", type=int, required=True)
    parser.add_argument("-af", "--embed_field", type=str, default="embedding")
    parser.add_argument("-iw", "--word2id_field", type=str, default="word2id")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    embeddings = load_embeddings(args.embedding_dir)

    with open(args.id2auth_dir, 'r') as f:
        id2auth = json.load(f)
    id2auth = {int(k) : v for k, v in id2auth.items()}

    auth_topic_vocab_embeds = numpy.zeros((len(id2auth), embeddings.vector_size))

    with gzip.open(args.input_dir, "rt") as ifd:
        for auth_id, line in tqdm(enumerate(ifd)):
            data = json.loads(line)
            embed_data = numpy.array(data[args.embed_field])
            embed_data = (embed_data.sum(axis=1))[:, args.topic_spec]
            if embed_data.sum() == 0:
                # logger.info(f"author {id2auth[auth_id]} has no vocab classified as topic {args.topic_spec}")
                continue
            embed_data /= embed_data.sum()
            vocab2id = data[args.word2id_field]
            id2vocab = {v : k for k, v in vocab2id.items()}
            for vocab_id, vocab_embed_ratio in enumerate(embed_data):
                if vocab_embed_ratio > 0:
                    vocab_embed = embeddings[id2vocab[vocab_id]]
                    auth_topic_vocab_embeds[auth_id] += vocab_embed_ratio * vocab_embed

    numpy.savetxt(args.output_dir, auth_topic_vocab_embeds, delimiter=',')