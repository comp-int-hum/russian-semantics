import sys, argparse, io, json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gzip

# used to check similarity between reported author names
import Levenshtein


def open_dataframe(df_dir: str, printable: bool = False) -> pd.DataFrame:
    metadata_df = pd.read_json(df_dir, lines=True)
    if printable is True:
        print(metadata_df.head())
    return metadata_df


def dump_dataframe(df_dir: str, df_data: pd.DataFrame) -> None:
    json_str = df_data.to_json(orient="records", lines=True)
    with gzip.open(df_dir, "wt", encoding="utf-8") as f:
        f.write(json_str)
    print(f"successfully dumped into file {df_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--step", type=int, default=0)

    # Parse the arguments
    args = parser.parse_args()

    if args.step == 0:
        print(f"no valid filter step provided.")
        exit(1)

    if args.step == 1:
        raise Exception("already completed this step!")

        # Filtering 01: Get Rid of all Instances without years
        input_dir = "/home/zxia15/data_zxia15/russian-semantics/work/metadata_russian_documents.jsonl.gz"
        dump_dir = "/home/zxia15/data_zxia15/russian-semantics/work/f01_metadata_russian_documents.jsonl.gz"
        df = open_dataframe(input_dir, printable=True)
        print(f"there are a total of {len(df)} instances of data")

        filted_01_df = df[~df["year"].isna()]
        print(
            f"after filtering our all null years, there are a total of {len(filted_01_df)} instances of data"
        )

        dump_dataframe(dump_dir, filted_01_df)
        exit(0)

    # for every filtering afterwards, the input_dir would be the prior step
    input_dir = f"/home/zxia15/data_zxia15/russian-semantics/work/f0{str(args.step - 1)}_metadata_russian_documents.jsonl.gz"
    dump_dir = f"/home/zxia15/data_zxia15/russian-semantics/work/f0{str(args.step)}_metadata_russian_documents.jsonl.gz"

    # Filtering 02: Get Rid of all repeated instances
    if args.step == 2:
        raise Exception("already completed this step!")
        df = open_dataframe(input_dir, printable=False)

        # first get the total number of unique titles
        # unique_titles = df["title"].unique()
        # print(
        #     f"there are a total of {len(unique_titles)} unique titles in the dataframe"
        # )

        # first check the number of redundant titles v.s. the number of not redundant ones
        title_counts = df["title"].value_counts()
        non_unique_titles = title_counts[title_counts > 1].index

        # print(
        #     f"there are a total of {len(non_unique_titles)} redundant titles and {len(title_counts) - len(non_unique_titles)} unique titles"
        # )

        # a possibly good assumption would be the year number associated with a particular
        # title would be a good indicator of whether the title is a general one or a repeated
        # one

        # here we assume three situations:
        #   1. the year is unique, so it becomes highly likely that it is a redundant work
        #   2. the year is not unique, but there are only two years, likely corresponding to the fact that some
        #       puts publication year whereas others put written year
        #   3. the year is not unique, and there are more than two years, making it most likely to be a common title

        redundant_title_df = df[df["title"].isin(non_unique_titles)]
        unique_years = redundant_title_df.groupby("title")["year"].nunique()
        redundant_or_two_years = unique_years[unique_years <= 2].index
        common_titles = unique_years[unique_years > 2].index

        print(
            f"there are a total of {len(redundant_or_two_years)} likely redundant titles with one or two year numbers;\n"
            + f"and a total of {len(common_titles)} likely common title with more than two year numbers."
        )

        # let's add a little bit of nuance by checking how likely an author is the same for the first case
        likely_redundant_df = df[df["title"].isin(redundant_or_two_years)]
        unique_authors = redundant_title_df.groupby("title")["author"].unique()
        unique_authors = unique_authors.apply(lambda x: x.tolist())

        redundant_filter_cases = (
            unique_authors[unique_authors.apply(len) == 1].index
        ).tolist()
        print(
            f"there are a total of {len(redundant_filter_cases)} securely filterable cases"
        )
        outlier_redundant_cases = unique_authors[
            unique_authors.apply(len) > 1
        ].to_dict()

        for redundant_title, redudant_author_arr in tqdm(
            outlier_redundant_cases.items()
        ):
            sim_ratio = []
            for i in range(len(redudant_author_arr) - 1):
                if redudant_author_arr[i] is None:
                    continue
                for j in range(i + 1, len(redudant_author_arr)):
                    if redudant_author_arr[j] is None:
                        continue
                    shorter_len = (
                        len(redudant_author_arr[i])
                        if len(redudant_author_arr[i]) < len(redudant_author_arr[j])
                        else len(redudant_author_arr[j])
                    )
                    sim_ratio.append(
                        Levenshtein.ratio(
                            redudant_author_arr[i][:shorter_len],
                            redudant_author_arr[j][:shorter_len],
                        )
                    )

            similarity_score = sum(sim_ratio) / len(sim_ratio)
            if similarity_score >= 0.75:
                redundant_filter_cases.append(redundant_title)

        print(
            f"there are {len(redundant_filter_cases)} filterable titles after second round"
        )
        # print(redundant_filter_cases[:5])
        # print(redundant_filter_cases[-5:])

        # after securing the number of filterable cases, split the data into those needs filters and those that does not
        non_filter_needed_df_index = (
            df[~df["title"].isin(redundant_filter_cases)].index
        ).tolist()

        filter_needed_df = df[df["title"].isin(redundant_filter_cases)]
        usable_index = (
            filter_needed_df.groupby("title")["russian text ratio"].idxmax()
        ).tolist()

        print(
            f"there are a total of {len(non_filter_needed_df_index)} non-filter-needed indexes and {len(usable_index)} filtered indexes"
        )

        non_filter_needed_df_index.extend(usable_index)

        filtered_02_df = (
            df.iloc[non_filter_needed_df_index].reset_index().drop(["index"], axis=1)
        )

        dump_dataframe(dump_dir, filtered_02_df)
        exit(0)

    if args.step == 3:
        raise Exception("already completed this step!")
        # step 3: check some outliers:
        #   1. the expected duration of publication is between 1500 and 1950.
        #      filter all those outside the range

        df = open_dataframe(input_dir, printable=True)
        df_year_too_early = df[df["year"] < 1500]
        df_year_too_late = df[(df["year"] > 1950)]

        print(
            f"there are a total of {len(df_year_too_early)} instances of year before 1500"
        )
        # df_year_too_early.to_json("output.json", orient="records", lines=True)
        df_year_too_early.to_csv("output.csv")

        # here areaa list of such words that do not need to be eliminated
        retainable_index = [
            1730,
            6720,
            9599,
            12727,
            14756,
            15318,
            15888,
            17032,
            17842,
            18789,
            18799,
            19607,
            20532,
            20533,
        ]

        df_filtered_early_translations = df[
            (df["year"] >= 1500) | (df.index.isin(retainable_index))
        ]

        print(
            f"there are a total of {len(df_filtered_early_translations)} instances retained and"
            + f"a total of {len(df) - len(df_filtered_early_translations)} filtered out"
        )
        dump_dataframe(dump_dir, df_filtered_early_translations)

    if args.step == 4:
        raise Exception("already completed this step!")
        # filter out
        input_dir = f"/home/zxia15/data_zxia15/russian-semantics/work/f03_metadata_russian_documents.jsonl.gz"
        df = open_dataframe(input_dir, printable=True)

        # find the most frequently appearing authors among a bin of timescale
        year_bin = [
            1000,
            1500,
            1600,
            1700,
            1750,
            1800,
            1820,
            1840,
            1860,
            1880,
            1900,
            1920,
            1950,
            2024,
        ]
        writer_stats = {}

        for i in range(len(year_bin) - 1):
            start_year = year_bin[i]
            end_year = year_bin[i + 1]

            df_filtered = df[(df["year"] >= start_year) & (df["year"] < end_year)]

            value_counts = df_filtered[
                "author"
            ].value_counts()  # Get the top 10 most common values
            writer_stats[str(start_year) + " to " + str(end_year)] = (
                value_counts.head(25).index
            ).tolist()

        with open("output.json", "w") as f:
            json.dump(writer_stats, f, indent=4)

    if args.step == 5:
        # filter out ratio that are not high enough
        input_dir = f"/home/zxia15/data_zxia15/russian-semantics/work/f03_metadata_russian_documents.jsonl.gz"
        dump_dir = f"/home/zxia15/data_zxia15/russian-semantics/work/f04_metadata_russian_documents.jsonl.gz"

        df = open_dataframe(input_dir, printable=True)

        filtered_df = df[df["russian text ratio"] != 0]
        text_ratios = filtered_df["russian text ratio"]
        plt.hist(x=text_ratios, bins=100, density=True)
        plt.savefig("output.png")

        filtered_above_95_df = df[df["russian text ratio"] >= 95]
        print(
            f"previously there are {len(filtered_df)} non-zero instances, "
            + f"ultimately filtered to {len(filtered_above_95_df)} instances"
        )
        dump_dataframe(dump_dir, filtered_above_95_df)
