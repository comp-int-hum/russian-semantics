import logging
import argparse
import pandas
from detm import load_embeddings


logger = logging.getLogger("generate_word_similarity_table")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--embeddings", dest="embeddings", help="W2V embeddings file")
    parser.add_argument("--output", dest="output", help="File to save table")
    parser.add_argument("--top_neighbors", dest="top_neighbors", default=10, type=int, help="How many neighbors to return")
    parser.add_argument('--target_words', default=[], nargs="*", help='Words to consider')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    embeddings = load_embeddings(args.embeddings)

    neighbors = []
    for w in args.target_words:
        row = [w]
        for ow, op in embeddings.most_similar(w, topn=args.top_neighbors):
            row.append("{}:{:.02f}".format(ow, op))
        neighbors.append(row)
        
    pd = pandas.DataFrame(neighbors)
    with open(args.output, "wt") as ofd:
        ofd.write(pd.to_latex(index_names=False, index=False, header=False))
