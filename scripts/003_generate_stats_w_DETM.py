import argparse, requests, re
from nltk.stem.snowball import SnowballStemmer
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

from detm import load_embeddings

def translate_text_deepl(text, target_lang='EN'):
    api_key = '0e1ffe54-80c7-462d-9bac-b2962074ba50:fx'
    url = 'https://api-free.deepl.com/v2/translate'
    
    data = {
        'auth_key': api_key,
        'text': text,
        'target_lang': target_lang
    }

    response = requests.post(url, data=data)
    if response.status_code == 200:
        result = response.json()
        translation = result['translations'][0]['text']
        return f"{text}({translation})"
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return text


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # io
    parser.add_argument("--input", type=str, required=True, default=None)
    parser.add_argument("--output", type=str, required=True, default=None)
    # used to select tabular result
    parser.add_argument("--topn", type=int, default=5)
    args = parser.parse_args()

    # use model for embedding variance check
    embedding_model = load_embeddings(args.input)

    words_of_interest: List[str] = [
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

    is_stemmed = re.search(r"stemmed", args.input) is not None
    assert is_stemmed is True

    if is_stemmed:
        stemmer = SnowballStemmer("russian")
        words_of_interest = [stemmer.stem(word) for word in words_of_interest]


    full_interest_table_data: Dict[str, List[str]] = {
        "Original Word": [],
    }
    for i in range(args.topn):
        full_interest_table_data.setdefault(f"Sim #{i+1}", [])

    for woi in words_of_interest:
        try:
            woi: str = woi.lower()
            translated_tuple_woi: str = translate_text_deepl(woi)
            print(f"getting similarity matrix of word {translated_tuple_woi}")
            sims = embedding_model.wv.most_similar(woi, topn=args.topn)
            sims_modified: List[str] = [
                (
                    translate_text_deepl(w)
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
    print(f"saving image to {args.output}")
    graph_tabular_data(full_interest_table_data, args.output)