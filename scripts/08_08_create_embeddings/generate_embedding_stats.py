import argparse
from typing import Dict, List, Set, Optional, Tuple, Any, Generator

from gensim.models import Word2Vec
from googletrans import Translator

# from tabulate import tabulate
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import textwrap


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
    epoch_number: Optional[int] = None,
    # wrap_width: int = 15,
    show_flag: bool = False,
):
    epoch_number = (
        epoch_number
        if epoch_number is not None
        else len(full_interest_table_data["Original Word"])
    )
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
    plt.savefig(f"mytable_{epoch_number}.png", bbox_inches="tight", dpi=300)

    if show_flag is True:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--topn", type=int, default=5)
    # parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    assert args.input
    # assert args.output

    print(f" ----- loading model from {args.input} ----- ")
    trained_model = Word2Vec.load(args.input)
    print(f" ----- successfully loaded the model ----- ")

    words_of_interest: List[str] = [
        # 1. some interesting imageries that might be used
        "мост",  # bridge
        "бронза",  # bronze
        "поезд",  # train
        "бричка",  # The brichka is a type of light, four-wheeled carriage
        # commonly used in Russia during the 19th century.
        # In Dead Souls, the protagonist Chichikov travels
        # around in a brichka as he goes about his dubious
        # business of acquiring "dead souls" (deceased serfs).
        # 2. some terminologies of particular significance to 19th century
        # novelists
        "Нигилизм",  # nihilism
        "Бесы",  # Demons
        "Земля",  # Land
        "Власть",  # power
        "Тело",  # body
    ]

    translator = Translator()

    # full_interest_table_data: List[List[str]] = []
    # headers: List[str] = ["Original Word"]
    # headers.extend([f"Sim #{i + 1}" for i in range(10)])
    full_interest_table_data: Dict[str, List[str]] = {
        "Original Word": [],
    }
    for i in range(args.topn):
        full_interest_table_data.setdefault(f"Sim #{i+1}", [])

    epoch: int = 0
    for woi in tqdm(words_of_interest):
        # try:

        woi: str = woi.lower()
        translated_tuple_woi: str = get_translate_untranslate_tuple(woi, translator)
        print(f" ----- getting similarity matrix of word {translated_tuple_woi} ----- ")
        sims: List | Any | List[Tuple[Any, float]] = trained_model.wv.most_similar(
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

        # graph_tabular_data(full_interest_table_data)

        # full_interest_table_data.append(woi_data)
        # print(f"{translated_tuple_woi}: {sims}")
        # print(tabulate(full_interest_table_data, headers=headers, tablefmt="grid"))
    # except Exception as e:
    #     print(str(e))
    #     pass
    # raise Exception(e)

    # headers: List[str] = ["Original Word"]
    # headers.extend([f"Sim #{i + 1}" for i in range(10)])
    # print(tabulate(full_interest_table_data, headers=headers, tablefmt="grid"))

    full_data = pd.DataFrame(full_interest_table_data)
    full_data.to_excel("mytable.xlsx")
