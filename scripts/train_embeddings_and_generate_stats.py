import gzip, json, argparse, re
from typing import Dict, List, Tuple, Any

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from googletrans import Translator
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

DEBUG_SENTENCE_SPLITTER: bool = False


def get_translate_untranslate_tuple(
    rus_word: str, translator: Translator = Translator()
) -> str:
    try:
        assert type(rus_word) == str
        eng_word: str = translator.translate(rus_word)
        #                                      , src="ru", dest="en").text
    except Exception as e:
        print(f"meeting error on russian word {rus_word}")
        print(str(e))
        # exit(0)
        return rus_word
    return f"{rus_word}({eng_word})"


def graph_tabular_data(
    full_interest_table_data,
    image_dest: str,
    show_flag: bool = False,
):
    df = pd.DataFrame(full_interest_table_data)

    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.8))
    ax.axis("tight")
    ax.axis("off")

    tbl = table(
        ax,
        df,
        loc="center",
        cellLoc="center",
        colWidths=[0.5] * len(df.columns),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))

    plt.savefig(image_dest, bbox_inches="tight", dpi=300)

    if show_flag is True:
        plt.show()


class TrainingCallback(CallbackAny2Vec):
    def __init__(self):
        self.epochs = 0

    def on_epoch_end(self, model):
        self.epochs += 1
        print(f"Epoch {self.epochs} completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--model_output", type=str, default=None)
    parser.add_argument("--stats_output", type=str, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--num_docs", type=int, default=None)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--skip_model", action="store_true", default=False)
    parser.add_argument(
        "--window_size", type=int, default=5, help="Skip-gram window size"
    )

    args = parser.parse_args()

    assert args.input and args.model_output and args.stats_output

    if args.skip_model is False:

        # step 1: saving and training model
        print(f"step 1: saving and training model")

        sentences = []

        with gzip.open(args.input, "rt") as ifd:
            idx_counter: int = 0
            terminating_flag: bool = False

            for entry in tqdm(ifd):
                j = json.loads(entry)
                sentences.append(j["content"].split())
                idx_counter += 1

                if args.num_docs is not None and args.num_docs > 0 and args.num_docs <= idx_counter:
                    print(f"--- iterating to idx {idx_counter} ---")
                    terminating_flag = True
                    break

            if terminating_flag is False:
                print(f"--- iterating to idx {idx_counter} ---")

        model = Word2Vec(
            sentences=sentences,
            vector_size=args.embedding_size,
            window=args.window_size,
            min_count=1,
            workers=4,
            sg=1,
            epochs=args.epochs,
            seed=args.random_seed,
            callbacks=[TrainingCallback()],
        )

        print(f"--- complete training. Saving model to {args.model_output} ---")
        model.save(args.model_output)

    # step 2: use model for embedding variance check
    print(f"step 2: use model for embedding variance check")

    if args.skip_model is True:
        print(f"---- loading model from {args.model_output} ----")
        model = Word2Vec.load(args.model_output)
        print(f"---- successfully loaded model ----")

    words_of_interest: List[str] = [
        # 1. some interesting imageries that might be used
        "мост",  # bridge
        "бронза",  # bronze
        "поезд",  # train
        "Нигилизм",  # nihilism
        "Бесы",  # Demons
        "красный",  # red
        "Земля",  # Land
        "Власть",  # power
        "Тело",  # body
    ]

    translator = Translator()

    full_interest_table_data: Dict[str, List[str]] = {
        "Original Word": [],
    }
    for i in range(args.topn):
        full_interest_table_data.setdefault(f"Sim #{i+1}", [])

    # epoch: int = 0
    # random.seed(args.random_seed)
    # random_word = [random.choice(model.wv.index_to_key)]
    # exit(0)
    for woi in tqdm(words_of_interest):
        # for woi in tqdm(random_word):

        try:

            woi: str = woi.lower()
            translated_tuple_woi: str = get_translate_untranslate_tuple(woi, translator)
            print(
                f" ----- getting similarity matrix of word {translated_tuple_woi} ----- "
            )
            sims: List | Any | List[Tuple[Any, float]] = model.wv.most_similar(
                woi, topn=args.topn
            )
            sims_modified: List[str] = [
                (
                    get_translate_untranslate_tuple(w, translator)
                    + "\n"
                    + str(float(int(f * 1000)) / 1000.0)
                )
                for (w, f) in sims
            ]

            full_interest_table_data["Original Word"].append(translated_tuple_woi)
            for i in range(args.topn):
                full_interest_table_data[f"Sim #{i+1}"].append(sims_modified[i])
        except KeyError as e:
            print(str(e))
            pass

    print(full_interest_table_data)
    print(f"--- saving image to {args.stats_output} ---")
    graph_tabular_data(full_interest_table_data, args.stats_output)
