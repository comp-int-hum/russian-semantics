import argparse, gzip, json, re, numpy
from detm import load_embeddings
from tqdm import tqdm
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embeddings_dir', type=str, default='/home/zxia15/detm-shootout/work/russian/embeddings.bin.gz')
    parser.add_argument('-i', '--input_dir', type=str, default='/home/zxia15/detm-shootout/corpora/russian.jsonl.gz')
    parser.add_argument("-o0", "--output_s0", type=str, default="author_vocab_counter.jsonl.gz")
    parser.add_argument("-o1", "--output_s1", type=str, default="author_embeddings.jsonl.gz")
    parser.add_argument("-c", '--content_field', type=str, default='text')
    parser.add_argument("-a", "--author_field", type=str, default="author_info")
    parser.add_argument("-s", "--step", type=int, default=0)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)

    args = parser.parse_args()

    if args.step == 0:

        embeddings = load_embeddings(args.embeddings_dir)
        tok2idx = embeddings.key_to_index
        del embeddings
        print(f"the embedding size is {len(tok2idx)}")

        author_vocab_counter = {}
    
        with gzip.open(args.input_dir, "rt") as ifd:
            for i, line in tqdm(enumerate(ifd)):
                j = json.loads(line)
                try:
                    author_vocab_counter.setdefault(j[args.author_field], Counter())
                    for tok in re.split(r"\s+", j[args.content_field].lower()):
                        if tok in tok2idx:
                            author_vocab_counter[j[args.author_field]][tok2idx[tok]] += 1
                
                except Exception as e:
                    print(str(e))
                    exit(0)
                word_frequency = Counter()
        
        word_freq = Counter()

        for _, vocab in author_vocab_counter.items():
            word_frequency.update(vocab.keys())

        num_authors = len(author_vocab_counter)
        words_to_remove = {word for word, count 
                           in word_frequency.items() 
                           if count / num_authors > args.threshold}
                    
        with gzip.open(args.output_s0, 'wt', encoding='utf-8') as f:
            for author, vocab in tqdm(author_vocab_counter.items()):
                author_data = {
                    'author': author,
                    'vocab': {word: count for word, count in 
                              vocab.items() 
                              if word not in words_to_remove}
                }
                f.write(json.dumps(author_data) + '\n')

    if args.step == 1:
        embeddings = load_embeddings(args.embeddings_dir)
        num_vocab = len(embeddings.key_to_index)

        all_data = []
        idx2author = {}
        author2idx = {}

        with gzip.open(args.output_s0, "rt") as ifd:
            for idx, line in tqdm(enumerate(ifd)):
                j = json.loads(line)
                idx2author[idx] = j['author']
                author2idx[j['author']] = idx
                data = numpy.zeros(num_vocab)
                vocab_indices = [int(k) for k in list(j['vocab'].keys())]
                vocab_counts = list(j['vocab'].values())
                data[vocab_indices] = vocab_counts
                all_data.append(data)
        
        num_auth = len(all_data)
        author_counters = numpy.array(all_data)
        
        embedding_size = embeddings.vector_size
        vocab_embedding_matrix = numpy.zeros((num_vocab, embedding_size))

        for vocab_term, vocab_idx in embeddings.key_to_index.items():  # Mapping index to word
            vocab_embedding_matrix[vocab_idx] = embeddings[vocab_term]
        
        assert vocab_embedding_matrix.shape == (num_vocab, embedding_size)
        assert author_counters.shape == (num_auth, num_vocab)

        author_embedding_dot = numpy.dot(author_counters, vocab_embedding_matrix) 
        assert author_embedding_dot.shape == (num_auth, embedding_size)
        author_vocab_counts = numpy.sum(author_counters, axis=1, keepdims=True)
        assert author_vocab_counts.shape == (num_auth, 1)
        author_embeddings = author_embedding_dot / numpy.maximum(author_vocab_counts, 1) 
        assert author_embeddings.shape == (num_auth, embedding_size)

        with gzip.open(args.output_s1, 'wt', encoding='utf-8') as f:
            for idx in range(num_auth):
                author_data = {
                    'author': idx2author[idx],
                    'embeddings': (author_embeddings[idx]).tolist()
                }
                f.write(json.dumps(author_data) + '\n')