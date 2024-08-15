import sys, argparse, io, json, re

# from tqdm import tqdm
from tqdm.autonotebook import tqdm, trange
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gzip, unicodedata

# used to check similarity between reported author names
import Levenshtein

def open_dataframe(df_dir: str, printable: bool = False) -> pd.DataFrame:
    metadata_df: pd.DataFrame = pd.read_json(df_dir, lines=True)
    if printable is True:
        print(metadata_df.head())
    return metadata_df


def dump_dataframe(df_dir: str, df_data: pd.DataFrame, printable: bool = False) -> None:
    if printable is True:
        print(df_data.head())
    json_str: str = df_data.to_json(orient="records", lines=True)
    with gzip.open(df_dir, "wt", encoding="utf-8") as f:
        f.write(json_str)
    print(f"successfully dumped into file {df_dir}")

def preprocess_str(orig_str: str) -> str:
    nfkd_form = unicodedata.normalize("NFKD", orig_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def regularize_title(orig_title: str) -> str:
    re_pattern = re.compile(r"[^.;:/\u0020]+")
    matches = re_pattern.findall(orig_title)
    return " ".join(matches)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--step", type=int, default=0)
    parser.add_argument("-l", "--levratio", type=float, default=0.95)
    args = parser.parse_args()

    input_dir: str = (
        f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.jsonl.gz"
    )
    dump_dir: str = (
        f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.jsonl.gz"
    )

    if args.step == 0:

        raise Exception("step completed")

        # step 1:
        # filter out:
        #   - without author name
        #   - of copyright year 1000 - 1950
        #   - less than 1e4 words

        input_dir: str = (
            "/home/zxia15/data_zxia15/russian-semantics/work/full_metadata_russian_documents.jsonl.gz"
        )
        df: pd.DataFrame = open_dataframe(input_dir, printable=True)

        filtered_author_df: pd.DataFrame = df[~df["author"].isna()]
        filtered_year_df: pd.DataFrame = filtered_author_df[
            ~filtered_author_df["year"].isna()
        ]
        filtered_year_df: pd.DataFrame = filtered_year_df[
            (filtered_year_df["year"] >= 1000) & (filtered_year_df["year"] <= 1950)
        ]
        filtered_content_len_df: pd.DataFrame = filtered_year_df[
            filtered_year_df["content_length"] >= 1e4
        ]

        filtered_content_len_df["regularized_author"] = filtered_content_len_df[
            "author"
        ].map(lambda x: preprocess_str(x))
        filtered_content_len_df["regularized_title"] = filtered_content_len_df[
            "title"
        ].map(lambda x: preprocess_str(x))

        # reduced the size of dataframe from 89353 to 41331
        print(
            f"reduced the size of dataframe from {len(df)} to {len(filtered_content_len_df)}"
        )
        dump_dataframe(dump_dir, filtered_content_len_df, printable=True)

    elif args.step == 1:

        raise Exception("step completed")

        # step 2: filter based on author names

        df: pd.DataFrame = open_dataframe(input_dir)
        # df = df.reset_index().drop(['index'], axis=1)
        unique_author_names: List[str] = df["regularized_author"].unique().tolist()
        print(f"there are {len(unique_author_names)} unique authors")

        # now need to find a way to group authors.
        # a trick may be to use abreviations

        # given each name is of format:
        # xxx, xxx (xxx)

        # for each kind of abbreviation,
        # give the author names, range of their publication years
        author_dictionary: Dict[str, Dict[str, List[int]]] = {}

        for author_name in tqdm(unique_author_names):

            # get the author name abbreviation
            last_first_name_split: List[str] = author_name.split(", ")
            last_name: str = last_first_name_split[0]
            inits_abbr: str = last_name[0] + ".,"

            if len(last_first_name_split) > 1:
                other_names: str = ", ".join(author_name.split(", ")[1:])
                other_names: List[str] = other_names.split(" ")

                for other_name in other_names:
                    inits_abbr += (
                        " " + other_name[0] + "." if len(other_name) > 0 else ""
                    )

            # get all the years associated with the author name
            all_years: List[int] = (
                (df[df["regularized_author"] == author_name])["year_copywright"]
                .unique()
                .tolist()
            )

            author_dictionary.setdefault(inits_abbr, {})
            author_dictionary[inits_abbr][author_name] = [
                min(all_years),
                max(all_years),
            ]

        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        with open(dump_dir, "w") as f:
            json.dump(author_dictionary, f, indent=4)

    elif args.step == 2:

        raise Exception("step completed")

        # step 3: filter based on author names
        # perform analysis on the dictionary

        input_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        with open(input_dir, "r") as f:
            author_data: Dict[str, Dict[str, List[int]]] = json.load(f)

        re_generated_author_data: Dict[str, Dict[str, int]] = {}

        for abbrev, author_dict in tqdm(author_data.items()):
            all_author_names: List[str] = [key for key, _ in author_dict.items()]
            correspondence_dict: Dict[str, int] = {}
            # correspondence_list : Tuple[List[str]] = ()

            # think all the last names are unabbreviated
            group_idx = 0

            if len(all_author_names) == 1:
                correspondence_dict[all_author_names[0]] = group_idx
                group_idx += 1

            for i in range(len(all_author_names) - 1):
                for j in range(i + 1, len(all_author_names)):
                    name_01: str = all_author_names[i]
                    name_02: str = all_author_names[j]

                    similarity_flag: bool = False

                    last_name_sim_ratio = Levenshtein.ratio(
                        name_01.split(", ")[0],
                        name_02.split(", ")[0],
                    )

                    if last_name_sim_ratio > args.levratio:
                        similarity_flag = True
                        other_names_01 = (", ".join(name_01.split(", ")[1:])).split(" ")
                        other_names_01 = [name for name in other_names_01 if name != ""]
                        other_names_02 = (", ".join(name_02.split(", ")[1:])).split(" ")
                        other_names_02 = [name for name in other_names_02 if name != ""]

                        try:
                            assert len(other_names_01) == len(other_names_02)
                        except AssertionError as e:
                            print(other_names_01, other_names_02)
                            raise AssertionError()

                        for idx in range(len(other_names_01)):
                            other_name_01 = other_names_01[idx]
                            other_name_02 = other_names_02[idx]

                            # abbreviation
                            if (
                                # len(other_name_01) == 2 and
                                other_name_01[-1]
                                == "."
                            ) or (  # len(other_name_02) == 2 and
                                other_name_02[-1] == "."
                            ):
                                if other_name_01[0] != other_name_02[0]:
                                    similarity_flag = False
                                    break

                            # non abbreviation
                            else:
                                if (
                                    Levenshtein.ratio(other_name_01, other_name_02)
                                    < args.levratio
                                ):
                                    similarity_flag = False
                                    break

                    if similarity_flag is True:

                        group_01 = correspondence_dict.get(name_01, None)
                        group_02 = correspondence_dict.get(name_02, None)

                        if group_01 is None and group_02 is None:
                            correspondence_dict[name_01] = group_idx
                            correspondence_dict[name_02] = group_idx
                            group_idx += 1
                        elif group_02 is None:
                            correspondence_dict[name_02] = group_01
                        elif group_01 is None:
                            correspondence_dict[name_01] = group_02
                        else:
                            if group_01 != group_02:
                                group_num = (
                                    group_01 if group_01 < group_02 else group_02
                                )
                                changeable_arr = []
                                for key, value in correspondence_dict.items():
                                    if value == group_01 or value == group_02:
                                        changeable_arr.append(key)

                                for changeable in changeable_arr:
                                    correspondence_dict[changeable] = group_num
                    else:
                        if correspondence_dict.get(name_01, None) is None:
                            correspondence_dict[name_01] = group_idx
                            group_idx += 1
                        if correspondence_dict.get(name_02, None) is None:
                            correspondence_dict[name_02] = group_idx
                            group_idx += 1

            # correspondence_dict["_max"] = group_idx
            re_generated_author_data[abbrev] = correspondence_dict

        with open(dump_dir, "w") as f:
            json.dump(re_generated_author_data, f, indent=4)

    elif args.step == 3:

        raise Exception("step completed")

        # use the data in the list to give more uniform names

        input_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        with open(input_dir, "r") as f:
            author_data: Dict[str, Dict[str, int]] = json.load(f)

        # organize author data
        all_name_dict: Dict[str, List[str]] = {}

        for _, name_dict in tqdm(author_data.items()):
            # max_idx: int = name_dict.get("_max", 0)
            reversed_data: Dict[int, Tuple[str, List[str]]] = {}

            for name, group_num in name_dict.items():
                # if name == "_max":
                #     continue

                reversed_data.setdefault(group_num, ("", []))

                first_name = (
                    name
                    if len(reversed_data[group_num][0]) < len(name)
                    else reversed_data[group_num][0]
                )
                second_list = reversed_data[group_num][1]
                second_list.append(name)

                reversed_data[group_num] = (first_name, second_list)

            for _, (representative_name, name_arr) in reversed_data.items():
                all_name_dict[representative_name] = name_arr

        with open(dump_dir, "w") as f:
            json.dump(all_name_dict, f, indent=4)

    elif args.step == 4:

        # raise Exception("step completed")

        input_json_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        input_df_dir: str = (
            "/home/zxia15/data_zxia15/russian-semantics/work/filtered1_metadata_russian_documents.jsonl.gz"
        )

        df: pd.DataFrame = open_dataframe(input_df_dir, printable=True)
        df = df.reset_index().drop(["index"], axis=1)

        # initialize a authorship column list
        authorship_column: List[str | None] = [None for _ in range(len(df))]

        with open(input_json_dir, "r") as f:
            author_dict: Dict[str, List[str]] = json.load(f)

        for normalized_author, author_list in tqdm(author_dict.items()):
            author_filtered_indeces: List[int] = []
            for author_name in author_list:
                author_filtered_indeces.extend(
                    df[df["regularized_author"] == author_name].index
                )
            authorship_column = [
                normalized_author if idx in author_filtered_indeces else author_name
                for idx, author_name in enumerate(authorship_column)
            ]

        df["normalized_author"] = authorship_column
        assert len(df[df["normalized_author"].isna()]) == 0
        df["regularized_title"] = df["regularized_title"].map(
            lambda x: regularize_title(x)
        )

        dump_dataframe(dump_dir, df, printable=True)

    elif args.step == 5:

        # raise Exception("step completed")

        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        df: pd.DataFrame = open_dataframe(input_dir, printable=True)

        # there are 14635 unique authors previously,
        # there are 9996 unique authors after normalization
        print(
            f"there are {len(df['regularized_author'].unique())} unique authors previously, \n"
            + f"there are {len(df['normalized_author'].unique())} unique authors after normalization"
        )

        author_work_name: Dict[str, List[str]] = {}

        # create a collection of all the works each author has created
        unique_authorship_list: List[str] = df["normalized_author"].unique().tolist()
        for author_name in tqdm(unique_authorship_list):
            author_work_name[author_name] = (
                (df[df["normalized_author"] == author_name])["regularized_title"]
                .unique()
                .tolist()
            )

        with open(dump_dir, "w") as f:
            json.dump(author_work_name, f, indent=4)

    elif args.step == 6:

        # raise Exception("step completed")
        # step 7: do the same filtering process, but this time through authors

        input_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        with open(input_dir, "r") as f:
            title_data_per_author: Dict[str, List[str]] = json.load(f)

        re_generated_title_data: Dict[str, Dict[str, int]] = {}

        for author_name, title_arr in tqdm(title_data_per_author.items()):

            correspondence_dict: Dict[str, int] = {}
            # correspondence_list : Tuple[List[str]] = ()

            # think all the last names are unabbreviated
            group_idx = 0
            if len(title_arr) == 1:
                correspondence_dict[title_arr[0]] = group_idx
                group_idx += 1
            for i in range(len(title_arr) - 1):
                for j in range(i + 1, len(title_arr)):
                    title_01: str = title_arr[i]
                    title_02: str = title_arr[j]

                    if len(title_01) < len(title_02):
                        title_02_arr = title_02.split(" ")

                        idx_before = 0
                        shortened_title_02 = ""
                        while len(shortened_title_02) < len(title_01):
                            idx_before += 1
                            shortened_title_02 = " ".join(title_02_arr[:idx_before])
                        shortened_title_01 = title_01
                    elif len(title_01) > len(title_02):
                        title_01_arr = title_01.split(" ")

                        idx_before = 0
                        shortened_title_01 = ""
                        while len(shortened_title_01) < len(title_02):
                            idx_before += 1
                            shortened_title_01 = " ".join(title_01_arr[:idx_before])
                        shortened_title_02 = title_02
                    else:
                        shortened_title_02 = title_02
                        shortened_title_01 = title_01
                    sim_ratio = Levenshtein.ratio(
                        shortened_title_01, shortened_title_02
                    )

                    if sim_ratio > args.levratio:

                        group_01 = correspondence_dict.get(title_01, None)
                        group_02 = correspondence_dict.get(title_02, None)

                        if group_01 is None and group_02 is None:
                            correspondence_dict[title_01] = group_idx
                            correspondence_dict[title_02] = group_idx
                            group_idx += 1
                        elif group_02 is None:
                            correspondence_dict[title_02] = group_01
                        elif group_01 is None:
                            correspondence_dict[title_01] = group_02
                        else:
                            if group_01 != group_02:
                                group_num = (
                                    group_01 if group_01 < group_02 else group_02
                                )
                                changeable_arr = []
                                for key, value in correspondence_dict.items():
                                    if value == group_01 or value == group_02:
                                        changeable_arr.append(key)

                                for changeable in changeable_arr:
                                    correspondence_dict[changeable] = group_num
                    else:
                        if correspondence_dict.get(title_01, None) is None:
                            correspondence_dict[title_01] = group_idx
                            group_idx += 1
                        if correspondence_dict.get(title_02, None) is None:
                            correspondence_dict[title_02] = group_idx
                            group_idx += 1

            re_generated_title_data[author_name] = correspondence_dict

        with open(dump_dir, "w") as f:
            json.dump(re_generated_title_data, f, indent=4)

    elif args.step == 7:

        # raise Exception("step completed")

        # use the data in the list to give more uniform names

        input_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        dump_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        )

        with open(input_dir, "r") as f:
            title_data: Dict[str, Dict[str, int]] = json.load(f)

        # organize author data
        all_name_dict: Dict[str, Dict[str, List[str]]] = {}

        for author_name, name_dict in tqdm(title_data.items()):

            reversed_data: Dict[int, Tuple[str, List[str]]] = {}

            for title_name, group_num in name_dict.items():

                reversed_data.setdefault(group_num, ("", []))

                first_name = (
                    title_name
                    if len(reversed_data[group_num][0]) < len(title_name)
                    else reversed_data[group_num][0]
                )
                second_list = reversed_data[group_num][1]
                second_list.append(title_name)

                reversed_data[group_num] = (first_name, second_list)

            all_title_dict: Dict[str, List[str]] = {}

            for _, (representative_name, name_arr) in reversed_data.items():
                all_title_dict[representative_name] = name_arr

            all_name_dict[author_name] = all_title_dict

        with open(dump_dir, "w") as f:
            json.dump(all_name_dict, f, indent=4)

    elif args.step == 8:

        # raise Exception("step completed")

        input_json_dir: str = (
            f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step}_metadata_russian_documents.json"
        )
        input_df_dir: str = (
            "/home/zxia15/data_zxia15/russian-semantics/work/filtered5_metadata_russian_documents.jsonl.gz"
        )

        # dump_dir: str = (
        #     f"/home/zxia15/data_zxia15/russian-semantics/work/filtered{args.step + 1}_metadata_russian_documents.json"
        # )

        df: pd.DataFrame = open_dataframe(input_df_dir, printable=True)
        df = df.reset_index().drop(["index"], axis=1)

        # initialize a authorship column list
        normalized_title_column: List[str | None] = [None for _ in range(len(df))]

        with open(input_json_dir, "r") as f:
            author_dict: Dict[str, Dict[str, List[str]]] = json.load(f)

        for author, title_dict in tqdm(author_dict.items()):

            for normalized_title, title_arr in title_dict.items():
                title_filtered_indeces: List[int] = []

                for title_name in title_arr:

                    title_filtered_indeces.extend(
                        df[
                            (df["normalized_author"] == author)
                            & (df["regularized_title"] == title_name)
                        ].index
                    )

                normalized_title_column = [
                    normalized_title if idx in title_filtered_indeces else title_name
                    for idx, title_name in enumerate(normalized_title_column)
                ]

        df["normalized_title"] = normalized_title_column
        assert len(df[df["normalized_title"].isna()]) == 0
        dump_dataframe(dump_dir, df, printable=True)

    elif args.step == 9:

        # raise Exception("step completed")

        # finally, time to do another round of redundancy elimination
        df: pd.DataFrame = open_dataframe(input_dir)

        df["normalized_author"] = df["normalized_author"].map(
            lambda x: x if x[-1] != "," else x[:-1]
        )

        non_cyrillic_authors: List[str] = [
            # French
            "Voltaire",
            "Gautier, Theophile",
            "Margeret, Jacques",
            "Beraud, Henri",
            "Laboulaye, Edouard",
            "Le Roux, Hugues",
            "Londres, Albert",
            "Barbusse, Henri",
            "Saint-John Perse",
            "Maeterlinck, Maurice",
            "Zola, Emile",
            "Hovelacque, Abel",
            "Patouillet, Jules.",
            "Gerard, Philippe Louis",
            "Hamon, Augustin Frederic",
            "La Bruyere, Jean de",
            "Moliere",
            "Racine, Jean",
            "Berwick, James Fitzjames",
            "Rulhiere, Claude Carloman de",
            "Compan, Charles",
            "Choffin, David Etienne",
            "Fenelon, Francois de Salignac de La Mothe-",
            "Jaucourt, Louis de",
            "Mably",
            "Prevost",
            "Rulhie  re, Claude Carloman de",
            "Le Sage, Alain Rene",
            "Rousseau, Jean Jacques",
            "Chardin, John",
            "Luchet, Jean-Pierre-Louis de",
            "Barthelemy, J.-J.",
            "Louvet de Couvray, Jean-Baptiste",
            "Guyon, Jeanne Marie (Bouvier de La Motte)",
            "Mason, John",
            "Proudhon, P.-J.",
            "Pataud, Emile",
            "Balzac, Honore de",
            "Guyau, Jean-Marie",
            "Michiels, Alfred",
            "Le Play, Frederic",
            "Septans, Albert",
            "Ferriere, Emile",
            "Beaumont-Vassy, Edouard Ferdinand de la Bonniere",
            "Vigouroux, Louis",
            "Tocqueville, Alexis de",
            "Maynial, Edouard",
            "Surowiecki, Lorenz",
            "Carrel, Armand",
            "Mignet",
            "Rodet, Julien",
            "Ghio, Paul",
            "Renan, Ernest",
            # Italian
            "Dante Alighieri",
            "Nitti, Francesco Saverio",
            "Sacchetti, Franco",
            "Antonii",
            "Petrarca, Francesco",
            "Polo, Marco",
            "Giovanni",
            "Ariosto, Lodovico",
            "Machiavelli, Niccolo",
            "Cellini, Benvenuto",
            "Condivi, Ascanio",
            "Brancati, Francesko",
            "Beccaria, Cesare",
            "Goldoni, Carlo",
            "Zoccoli, Ettore",
            # English
            "Darwin, Charles",
            "Shakespeare, William, 1564-1616.",
            "Shakespeare, William",
            "Trollope, Frances Milton",
            "Bunyan, John",
            "Coleridge, Samuel Taylor",
            "Wilde, Oscar",
            "Thomas",
            "Smith, Thomas",
            "Milton, John",
            "Fletcher, Giles",
            "Horsey, Jerome",
            "Defoe, Daniel",
            "Macpherson, James",
            "Franklin, Benjamin",
            "Fielding, Henry",
            "Kimber, Edward",
            "Ridley, Charles",
            "Robertson, William",
            "Plaisted, Bartholomew",
            "Hearne, Samuel",
            "Juel, Just",
            "Kain, Vaʹnka",
            "Smith, Adam",
            "Swift, Jonathan",
            "Addison, Joseph",
            "Young, Edward",
            "Longfellow, Henry Wadsworth",
            "Varb, E.",
            "Irving, Washington",
            "Dickens, Charles",
            "Moore, Thomas",
            "Laurie, Joseph",
            "Ruddock, E. H.",
            "Farrington, E. A.",
            "Ricardo, David",
            "Clarke, Alexander Ross",
            "Ward, Lester Frank",
            "Marshall, William",
            "Ticknor, George",
            "Blank, R. M.",
            "Kaler, Emil",
            "Cooper, James Fenimore",
            "Byron, George Gordon Byron",
            # Latin / Czech / Latvian / Polish / Dutch
            "Comenius, Johann Amos",
            "Butter, Oskar",
            "Endzelins, Janis",
            "Dometiian",
            "Symeon",
            "Herberstein, Sigmund",
            "Tasso, Torquato",
            "Piotrowski, Jan",
            "Barezzi, Barezzo",
            "Rodes, Johann de",
            "Spinoza, Benedictus de",
            "Helmont, Franciscus Mercurius van",
            "Zołkiewski, Stanisław",
            "Niemojewski, Stanisław",
            "Syrokomla, Władysław",
            "Siestrzencewicz-Bohusz, Stanisław",
            "Stanisław",
            "Krizanic, Juraj",
            "Van Dyck, Anthony",
            "Maryna Mniszchowna",
            "Vratislav z Mitrovic, Vaclav",
            "Manstein, Cristof Hermann",
            "Suhm, Peter Frederik",
            "Augustine",
            "Cicero, Marcus Tullius.",
            "Horace.",
            "Mallet, Paul Henri",
            "Berwick y Liria, Jacobo Francisco Fitz James Stuart",
            "Jezbera, Frantisek Jan",
            "Gilliard, Pierre",
            "Curtius Rufus, Quintus",
            "Bruun, Filip Jakob",
            "Meyer, Conrad Ferdinand",
            # German
            "Schlechtendal, Diedrich Franz Leonard von",
            "Heine, Heinrich",
            "Altenberg, Peter",
            "Scherr, Johannes",
            "Storm, Theodor",
            "Suttner, Bertha von",
            "Liebknecht, Wilhelm",
            "Graetz, Heinrich",
            "Olearius, Adam",
            "Tectander von der Jabel, Georg",
            "Drexel, Jeremias",
            "Meyerberg, Augustin",
            "Roth, Eberhard Rudolph",
            "Baumeister, Friedrich Christian",
            "Bayer, T. S.",
            "Brincken, Julain",
            "Will, Georg Andreas",
            "Achenwall, Gottfried",
            "Freyer, Hieronymus",
            "Halle, Johann Samuel",
            "Krafft, Georg Wolfgang",
            "Wolff, Christian",
            "Leibniz, Gottfried Wilhelm",
            "Staehlin, Jakob von",
            "Petiscus, A. H.",
            "Turbia",
            "Schleicher, August",
            "Adelung, Friedrich von",
            "Seidlitz, Karl Johann von",
            "Stirner, Max",
            "Miller, Orest",
            "Marx, Karl",
            "Engels, Friedrich",
            "Jost, I. M.",
            "Klassen, E.",
            "Regel, Eduard August",
            "Reutz, Alexander von",
            "Muller, Clotar Moriz",
            "Fellenberg-Ziegler, Ferdinand Albert von",
            "Jahr, G. H. G.",
            "Hahnemann, Samuel",
            "Ritter, Carl",
            "Reiff, Ch. Ph.",
            "Neumann, Karl Friedrich",
            "Lubker, Friedrich Heinrich Christian",
            "Perwolf, Josef",
            "Schlosser, Friedrich Christoph",
            "Helbig, Georg Adolf Wilhelm von",
            "Spielhagen, Friedrich",
            "Fresenius, C. Remigius",
            "Eckartshausen, Karl von",
            "Brennecke, Ludwig",
            "Regel, E.",
            "Miklosich, Franz",
            "Hipping, Anders Johann",
            "Lebzeltern, Ludwig",
            "Hoffmann, E. T. A.",
            "Kotzebue, August von",
            "Chamisso, Adelbert von",
            "Stritter, Johann Gotthelf von",
            "Huber, Johannes",
            "Beer, Adolf",
            "Mayr, Georg von",
            "Hellwald, Friedrich von",
            "Lupke, Robert Theodor Wilhelml",
            "Thun, Alphon",
            # Danish
            "Brandes, Georg Morris Cohen",
            # Swiss
            "Platten, Fritz",
            # Armenian
            "Nerses",
            # Greek
            "Gregory Palamas",
            "Gregory",
            "Kirakos",
            "Theophylactus",
            "Choniates, Nicetas",
            "Bryennius, Nicephorus",
            "Gavriil of Nazareth",
            "Kalloudes, Arsenios",
            "Paul",
            "Meniates, Elias",
            "Meletios",
            "Leichoudes, Ioannikios",
            "Altschul, Elias",
            "Anacreon",
            "Pachymeres, George",
            # Spanish
            "Cid",
            "Calderon de la Barca, Pedro",
            "Cervantes Saavedra, Miguel de",
            # Portuguese
            "Camoes, Luis de",
            "Sanches, Antonio Nunes Ribeiro",
            # Arabic
            "Babur",
            "Ebulgazi Bahadir Han",
            "Pavel, Aleppskii",
            "Bakri, Abu \u02bbUbayd \u02bbAbd Allah ibn \u02bbAbd al-\u02bbAziz",
            # Hebrew
            "Hannover, Nathan Nata",
            "Pines, Meyer Isser",
            # Scandinavian
            "Lundberg, E.",
            # other
            "Kenesaryuly, Akhmet",
        ]

        # for author in non_cyrillic_authors:
        #     if len(df[df["normalized_author"] == author]) == 0:
        #         print(f"unable to find author {author}")

        filtered_df = df[~df["normalized_author"].isin(non_cyrillic_authors)].copy()
        filtered_df = filtered_df.reset_index().drop(["index"], axis=1)

        print(
            f"filtering of authors reduces data size from {len(df)} to {len(filtered_df)}"
        )

        filtered_df["author_plus_title"] = filtered_df.apply(
            lambda x: x["normalized_title"] + " -- " + x["normalized_author"], axis=1
        )

        unique_title_plus_author_instances: List[str] = (
            filtered_df["author_plus_title"].unique().tolist()
        )

        print(
            f"among those {len(unique_title_plus_author_instances)} unique title and author"
        )

        filtered_indices: List[int] = []

        for unique_instance in tqdm(unique_title_plus_author_instances):

            filtered_indices.append(
                filtered_df[filtered_df["author_plus_title"] == unique_instance]
                .sort_values(by="content_length", ascending=False)
                .index[0]
            )

        filtered_indices = list(set(filtered_indices))
        # re_filtered_df = filtered_df[filtered_df.index.isin(filtered_indices)]
        re_filtered_df = filtered_df.iloc[filtered_indices]
        dump_dataframe(dump_dir, re_filtered_df)

    elif args.step == 10:

        # raise Exception("step completed")

        df = open_dataframe(input_dir, printable=True)
        df = df.reset_index().drop(
            [
                "index",
                # "year",
                "year_copywright",
                "regularized_author",
                "regularized_title",
                "author_plus_title",
                "author",
                "title",
            ],
            axis=1,
        )

        print(
            f"there are currently {len(df)} instances. The columns of the dataframe includes {df.columns}"
        )

        dump_dataframe(dump_dir, df)

    else:
        raise Exception("unimplemented")
