import gzip, json, argparse, re, csv
from typing import Dict, List, Tuple, Any
from collections import Counter
from tqdm import tqdm
import pandas as pd
from jsonl_file_utils import write_jsonl_file


def dump_counter_data(file_name: str, counter: Counter) -> None:
    with open(file_name, "w") as f:
        fieldnames = ["vocab", "frequency"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for vocab_instance in tqdm(counter.items()):
            writer.writerow(
                {"vocab": vocab_instance[0], "frequency": vocab_instance[1]}
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_vocab", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vocabfreq_cap", type=int, default=10)
    parser.add_argument("--num_docs", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--skip_vocabcounter", action="store_true")

    args = parser.parse_args()

    assert args.input and args.input_vocab and args.output
    assert args.vocabfreq_cap > 0 and args.num_docs > 0
    assert args.skip_vocabcounter

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

    non_punctuation_units = re.compile(
        r"[^\u0000-\u0020\u0021\u0022\u0027-\u0029\u005B\u005D\u007B\u007D\u002C\u002E\u003A\u003B\u003F]+"
    )

    if args.skip_vocabcounter is False:

        vocab_counter: Counter = Counter()

        with gzip.open(args.input, "rt") as ifd:
            idx_counter: int = 0
            terminating_flag: bool = False

            for entry in tqdm(ifd):
                j = json.loads(entry)
                local_vocabs: List[str] = non_punctuation_units.findall(j["content"])
                local_vocabs = [local_vocab.lower() for local_vocab in local_vocabs]
                vocab_counter += Counter(local_vocabs)

                idx_counter += 1

                if idx_counter >= args.num_docs:
                    terminating_flag = True
                    print(f" --- iterated to idx {idx_counter} --- ")
                    break

                if idx_counter % 5000 == 0:
                    less_used_vocabs = [
                        vocab_entry
                        for vocab_entry in vocab_counter.items()
                        if vocab_entry[1] <= args.vocabfreq_cap
                    ]

                    print(
                        f"at idx {idx_counter}, there are a total of {len(vocab_counter)} vocabs, among those {len(less_used_vocabs)} has less than {args.vocabfreq_cap} frequency"
                    )

                    dump_counter_data(f"output_at_idx_{idx_counter}.csv", vocab_counter)

            if terminating_flag is False:
                print(f" --- iterated to idx {idx_counter} --- ")

        # do some statistics of vocab_counter
        less_used_vocabs = [
            vocab_entry
            for vocab_entry in vocab_counter.items()
            if vocab_entry[1] <= args.vocabfreq_cap
        ]

        print(
            f"there are a total of {len(vocab_counter)} vocabs, among those {len(less_used_vocabs)} has less than {args.vocabfreq_cap} frequency"
        )
        # print(f"the less frequent terms are {less_used_vocabs}")

        print(f" --- writing to output file {args.output} --- ")
        dump_counter_data(args.output, vocab_counter)

    vocab_freq_df = pd.read_csv(args.input_vocab)
    # print(vocab_freq_df.head())
    less_than_cap_vocabs = vocab_freq_df[
        vocab_freq_df["frequency"] <= args.vocabfreq_cap
    ]["vocab"].unique()
    more_than_cap_vocabs = set(
        vocab_freq_df[vocab_freq_df["frequency"] > args.vocabfreq_cap]["vocab"].unique()
    )
    # print(
    #     f"there are initially {len(vocab_freq_df)} instances of vocab, of which {len(less_than_cap_vocabs)} are with less than {args.vocabfreq_cap} times used"
    # )
    # print(f"some of the vocabs include : {less_than_cap_vocabs[:15]}")

    with gzip.open(args.input, "rt") as ifd:

        idx_counter: int = 0
        terminating_flag: bool = False

        for entry in tqdm(ifd):
            j = json.loads(entry)
            local_vocabs: List[str] = non_punctuation_units.findall(j["content"])
            local_vocabs_filtered = [
                local_vocab.lower()
                for local_vocab in local_vocabs
                if local_vocab.lower() in more_than_cap_vocabs
            ]

            j["content"] = " ".join(local_vocabs_filtered)

            write_jsonl_file(args.output, j)

            idx_counter += 1

            if idx_counter >= args.num_docs:
                terminating_flag = True
                print(f" --- iterated to idx {idx_counter} --- ")
                break

        if terminating_flag is False:
            print(f" --- iterated to idx {idx_counter} --- ")
