import argparse, logging, gzip, pickle, json, numpy, math, re, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from googletrans import Translator

logger = logging.getLogger("order_by_auth")

def _get_translation(rus_word, translator):
    result = translator.translate(rus_word, src='ru', dest='en')
    return rus_word + '-' + result.text if result else rus_word

def _get_prop_and_dist_helper(data_matrice, id2data, top_n, epsilon=1e-6):

    translator = Translator()
    nvocabs, ntopics = data_matrice.shape
    vocab_dist_of_topic = (data_matrice.transpose() / 
                            (data_matrice.sum(1) + epsilon)).transpose()
    # topic_prop_of_vocab = data_matrice / (data_matrice.sum(0) + epsilon)
    # mean_others = ((vocab_dist_of_topic.sum(1, keepdims=True) - vocab_dist_of_topic) 
    #                 / (ntopics - 1))
    # exclusivity = vocab_dist_of_topic - mean_others

    return_data = [[''] * ntopics for _ in range(top_n)]
    
    def get_top_n(data, starter=False):
        
        for topic_idx in range(ntopics):
            data_slice = data[:, topic_idx]
            top_n_indices = numpy.argsort(data_slice)[-top_n:][::-1]
            # top_n_data = score_per_topic_per_time[top_n_indices]
            for data_idx, sort_idx in enumerate(top_n_indices):
                if not starter:
                    return_data[data_idx][topic_idx] += ' '
                return_data[data_idx][topic_idx] += _get_translation(id2data[sort_idx], translator) 
        
    get_top_n(vocab_dist_of_topic, starter=True)
    # get_top_n(vocab_dist_of_topic)
    # get_top_n(exclusivity)
    
    return return_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # auxiliary info
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--num_topic", type=int, default=20)
    parser.add_argument("--per_chart", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with gzip.open(args.model_dir, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device("cpu"), weights_only=False)
    
    vocab_list = model.word_list
    del model
    id2vocab = {k: v for k, v in enumerate(vocab_list)}
    vocab2id = {v: k for k, v in id2vocab.items()}
    vocab_topic_counts = numpy.loadtxt(args.input_dir, delimiter=',')

    data = numpy.array(_get_prop_and_dist_helper(vocab_topic_counts, id2vocab, 
                                                top_n=args.top_n, epsilon=args.epsilon))

    total_chart_num = int(args.num_topic // args.per_chart)
    for starter_idx in range(total_chart_num):
        fig, ax = plt.subplots(figsize=(20, 20))  # Adjust size as needed
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=data[:, starter_idx*args.per_chart:(starter_idx+1)*args.per_chart], cellLoc="center", loc="center",
                        colLabels=[f'topic #{idx}' for idx in range(starter_idx*args.per_chart, (starter_idx+1)*args.per_chart)], 
                        rowLabels=[f'#{idx}' for idx in range(args.top_n)])
        table.auto_set_font_size(False)
        table.set_fontsize(15)
        table.scale(1.5, 1.5)
        plt.savefig(re.sub(r'\[num\]', rf'#{starter_idx*args.per_chart}-#{(starter_idx+1)*args.per_chart}', args.output_dir), bbox_inches="tight", dpi=300)