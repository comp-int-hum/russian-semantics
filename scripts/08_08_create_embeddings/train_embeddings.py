import logging, gzip, math, json, argparse, random, re
from typing import Dict, List, Set, Optional, Tuple, Any, Generator
from jsonl_file_utils import open_jsonl_file

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# logger = logging.getLogger("train_dtm")

DEBUG_SENTENCE_SPLITTER: bool = False


class SentenceGenerator:
    def __init__(self, data_generator, non_punctuation_pattern):
        self.data_generator = data_generator
        self.non_punctuation_pattern = non_punctuation_pattern

    def __iter__(self):
        for record in self.data_generator:
            local_sentences = self.non_punctuation_pattern.findall(
                json.loads(record)["content"]
            )
            local_sentences = [sentence.lower() for sentence in local_sentences]
            yield local_sentences


class TrainingCallback(CallbackAny2Vec):
    def __init__(self):
        self.epochs = 0

    def on_epoch_end(self, model):
        self.epochs += 1
        print(f"Epoch {self.epochs} completed")


def prior_attempt_main():
    sentences = []
    idx_counter: int = 0

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
    try:
        while True:
            record: Dict[str, any] = json.loads(next(data_generator))
            # print(record.keys())
            # break

            # first try filtering out all punctuations
            local_sentences: List[str] = non_punctuation_units.findall(
                record["content"]
            )
            lowered_local_sentences: List[str] = [
                local_sentence.lower() for local_sentence in local_sentences
            ]

            if DEBUG_SENTENCE_SPLITTER:
                print("case of using re: ")
                print(f"has a length of {len(local_sentences)}")
                print(f"some of the initial instnaces : {local_sentences[10:20]}")

                print("case of using re with lower : ")
                print(f"has a length of {len(lowered_local_sentences)}")
                print(
                    f"some of the initial instnaces : {lowered_local_sentences[10:20]}"
                )

                split_local_sentences = record["content"].split()
                print("case of using split: ")
                print(f"has a length of {len(split_local_sentences)}")
                print(f"some of the initial instnaces : {split_local_sentences[10:20]}")

            sentences.append(lowered_local_sentences)
            idx_counter += 1

            if idx_counter % 1000 == 0:
                print(f"navigating to idx {idx_counter}")

            # break
            # print(sentences)
            # sentences. j["full_text"]
    # for x in sentences:
    #   print(x)
    #  for y in x:
    #     print(y)
    except StopIteration:
        print(f"end of input at {idx_counter} line")
        pass
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught. Currently at {idx_counter} line")
        raise KeyboardInterrupt
    except Exception as e:
        raise Exception(f"encounters exception at {idx_counter} line : {str(e)}")
    # exit(0)

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.embedding_size,
        window=args.window_size,
        min_count=1,
        workers=4,
        sg=1,
        epochs=args.epochs,
        seed=args.random_seed,
    )

    model.save(args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--window_size", type=int, default=5, help="Skip-gram window size"
    )

    args = parser.parse_args()

    assert args.input
    assert args.output

    # logging.basicConfig(
    #     format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    # )

    if args.random_seed:
        random.seed(args.random_seed)

    model = Word2Vec(
        vector_size=args.embedding_size,
        window=args.window_size,
        min_count=1,
        workers=4,
        sg=1,
        seed=args.random_seed,
    )

    data_generator = open_jsonl_file(args.input)
    non_punctuation_units = re.compile(
        r"[^\u0000-\u0020\u0021\u0022\u0027-\u0029\u005B\u005D\u007B\u007D\u002C\u002E\u003A\u003B\u003F]+"
    )
    sentences = SentenceGenerator(data_generator, non_punctuation_units)
    # Build vocabulary
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=args.epochs,
    callbacks=[TrainingCallback()])
    model.save(args.output)
