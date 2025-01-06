import argparse, json
from detm import Corpus, open_jsonl_file, train_embeddings, save_embeddings

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # io
    parser.add_argument("--input", type=str, required=True, default=None)
    parser.add_argument("--output", type=str, required=True, default=None)
    # process_input
    parser.add_argument("--num_docs", type=int, default=None)
    # embedding model train params
    parser.add_argument("--content_field", type=str, default='text')
    parser.add_argument("--max_subdoc_length", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--window_size", type=int, default=5, help="Skip-gram window size"
    )
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    # generate corpus
    document_corpus = Corpus()
    data_generator = open_jsonl_file(args.input)
    idx_counter = 0
    try:
        while True:
            entry = next(data_generator)
            document_corpus.append(json.loads(entry))
            idx_counter += 1
            if (args.num_docs is not None and 
                args.num_docs > 0 and 
                args.num_docs <= idx_counter):

                print(f"--- iterating to idx {idx_counter} ---")
                break
    
    except StopIteration:
        print(f"--- iterating to idx {idx_counter} ---")
        pass

    # train embedding
    embedding_model = train_embeddings(document_corpus, args.content_field, 
                                       epochs=args.epochs, 
                                       window_size=args.window_size, 
                                       embedding_size=args.embedding_size, 
                                       random_seed=args.random_seed)
    save_embeddings(embedding_model, args.output)