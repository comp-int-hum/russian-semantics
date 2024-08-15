import logging, gzip, math, json, argparse, random, re
from typing import Dict, List, Set, Optional, Tuple, Any, Generator

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from googletrans import Translator
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

DEBUG_SENTENCE_SPLITTER: bool = False


def get_translate_untranslate_tuple(
    rus_word: str, translator: Translator = Translator()
) -> str:
    try:
        assert type(rus_word) == str
    except Exception as e:
        print(f"meeting error on russian word {rus_word}")
        return rus_word
    eng_word: str = translator.translate(rus_word, src="ru", dest="en").text
    return f"{rus_word}({eng_word})"


def graph_tabular_data(
    full_interest_table_data,
    image_dest: str,
    # wrap_width: int = 15,
    show_flag: bool = False,
):
    df = pd.DataFrame(full_interest_table_data)
    # df = df.applymap(lambda x: "\n".join(textwrap.wrap(str(x), wrap_width)))

    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.8))
    ax.axis("tight")
    ax.axis("off")

    # Create the table with better formatting
    tbl = table(
        ax,
        df,
        loc="center",
        cellLoc="center",
        # colWidths=[0.15] * len(df.columns)
        colWidths=[0.5] * len(df.columns),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    # tbl.auto_set_column_height(col=list([0.3] * len(df.columns)))

    # Save the table as an image with high resolution
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
    parser.add_argument("--num_docs", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--window_size", type=int, default=5, help="Skip-gram window size"
    )

    args = parser.parse_args()

    assert args.input
    assert args.model_output
    assert args.stats_output
    assert args.num_docs != 0

    # \u0000-\u0020 --> all kinds of random space / tap/ etc.
    # \u0021 --> exclamation mark
    # \u0022 --> double quotation mark
    # \u0027 --> single quotation mark
    # \u0028-\u0029 --> small brackets
    # \u005B, \u005D --> mid brackets
    # \u007B-\u007D --> big brackets
    # \u002C --> comma
    # \u002E --> period
    # \u003A -->
    # \u003B --> semicolon
    # \u003F --> question mark

    # step 1: saving and training model
    print(f"step 1: saving and training model")

    non_punctuation_units = re.compile(
        r"[^\u0000-\u0020\u0021\u0022\u0027-\u0029\u005B\u005D\u007B\u007D\u002C\u002E\u003A\u003B\u003F]+"
    )

    sentences = []

    with gzip.open(args.input, "rt") as ifd:
        idx_counter: int = 0

        for entry in ifd:
            j = json.loads(entry)
            local_sentences: List[str] = non_punctuation_units.findall(j["content"])
            local_sentences = [
                local_sentence.lower() for local_sentence in local_sentences
            ]
            sentences.append(local_sentences)
            idx_counter += 1

            if args.num_docs <= idx_counter:
                print(f"--- iterating to idx {idx_counter} ---")
                break

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
    words_of_interest: List[str] = [
        "поезд",  # train
        "Бесы",  # Demons
        "церковь",  # church
        "красный"  # red
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
