import argparse, logging, json, gzip, torch, numpy
from detm import load_embeddings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from googletrans import Translator

logger = logging.getLogger('generate per topic vocab embed')

def _get_translation(rus_word, translator):
    result = translator.translate(rus_word, src='ru', dest='en')
    return rus_word + '-' + result.text if result else rus_word

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-t", "--topic_spec", type=int, required=True)
    parser.add_argument('-f', "--filtered_author", type=str, required=True)
    parser.add_argument("-n", "--top_n", type=int, default=10)
    parser.add_argument("-af", "--author_field", type=str, default="author")
    parser.add_argument("-ef", "--embed_field", type=str, default="embedding")
    parser.add_argument("-iw", "--word2id_field", type=str, default="word2id")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    translator = Translator()

    filtered_auth = args.filtered_author.split(";")
    auth2id = {v: k for k, v in enumerate(filtered_auth)}
    auth_topn_data = [[] for _ in range(len(auth2id))]

    with gzip.open(args.input_dir, "rt") as ifd:
        for i, line in tqdm(enumerate(ifd)):
            data = json.loads(line)
            author_name = data[args.author_field]

            if author_name in auth2id:
                vocab2id = data[args.word2id_field]
                id2vocab = {v : k for k, v in vocab2id.items()}
                vocab_list = [id2vocab[k] for k in sorted(id2vocab.keys())]
                embed_data = numpy.array(data[args.embed_field])
                embed_data = (embed_data.sum(axis=1))[:, args.topic_spec]
                top_n_indices = (numpy.argsort(embed_data)[-args.top_n:][::-1]).astype(numpy.int16)
                top_n_vocab = [_get_translation(vocab_list[i], translator) for i in top_n_indices]
                auth_topn_data[auth2id[author_name]].extend(top_n_vocab)

    fig, ax = plt.subplots(figsize=(20, 10))
    table = ax.table(cellText=auth_topn_data, 
                     rowLabels=[auth_instance.split(', ')[0] for auth_instance in filtered_auth],
                     loc='center', 
                     cellLoc='center')
    
    # for idx in range(len(filtered_auth)):
    #     cell = table.get_celld().get((0, idx))  # +1 for header offset
    #     if cell is not None:
    #         cell.set_text_props(
    #             fontproperties=FontProperties(weight='bold'),
    #         )
    #     cell = table.get_celld().get((idx + 1, -1))  # +1 for header offset
    #     if cell is not None:
    #         cell.set_text_props(
    #             fontproperties=FontProperties(weight='bold'),
    #         )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    # Add padding to cells
    for cell in table._cells.values():
        cell.PAD = 0.05
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output_dir, bbox_inches='tight', dpi=300)
    plt.close()