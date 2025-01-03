import math, random, gzip, json, torch
from tqdm import tqdm
from collections import Counter
import numpy as np

class CustomDataloader:
    
    def __init__(self, args, logger):
        
        self.min_time = args.min_time if args.min_time else 0
        self.max_time = args.max_time if args.max_time else 2000
        self.random_seed = args.random_seed
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.batch_preprocess = args.batch_preprocess
        self.device = args.device
        self.max_subdoc_length = args.max_subdoc_length
        self.min_word_occurrence = args.min_word_occurrence
        self.max_word_proportion = args.max_word_proportion
        self.epoch = 0

        self.all_window_ranges = [f"{self.min_time + idx * self.window_size}-" + 
                                  f"{self.min_time + (idx + 1) * self.window_size if self.min_time + (idx + 1) * self.window_size <= self.max_time else self.max_time}" 
                                  for idx in range(math.ceil((self.max_time - self.min_time) / self.window_size))]
        
        self.logger = logger
    
    def preprocess_instance(self, line, data, token2subdoccount, is_eval=False):

        j = json.loads(line)
        time = int(j["written_year"])
        
        if self.min_time <= time < self.max_time:
            time_intereval_key = self.all_window_ranges[(time - self.min_time) // self.window_size]
            title = j["title"]
            author = j["author_info"]
            htid = j['htid']
            docs = j["text"].split()

            num_subdocs = math.ceil(len(docs) / self.max_subdoc_length)
            subdocs = [docs[i * self.max_subdoc_length : (i + 1) * self.max_subdoc_length] 
                       for i in range(num_subdocs)]
            for subdoc_num, subdoc in enumerate(subdocs):
                utokens = {t for t in subdoc}
                for t in utokens:
                    token2subdoccount[t] = token2subdoccount.get(t, 0) + 1
                
                data_instance = {"time": time,
                                 "tokens": subdoc,
                                 "title": title,
                                 "author": author,
                                 'htid': htid,
                                 "subdoc_number": subdoc_num,
                                 'window': time_intereval_key,
                                 'window_idx': (time - self.min_time) // self.window_size
                                 }
                
                if is_eval:
                    data['eval'].setdefault(time_intereval_key, [])
                    data['eval'][time_intereval_key].append(data_instance)
                    
                else:
                    data['train'].setdefault(time_intereval_key, [])
                    data['train'][time_intereval_key].append(data_instance)

    def preprocess_data(self, train_dir, eval_dir, train_proportion, 
                        id2token=None, token2id=None):

        self.logger.info(f"----- starts preprocessing data ----- ")

        data = {"train": {},
                "eval": {}}
        token2subdoccount = {}

        if eval_dir:
            for name in ["train", "eval"]:
                with gzip.open(train_dir if name == 'train' else eval_dir, "rt") as ifd:
                    for line in tqdm(ifd):
                        self.preprocess_instance(line, data, token2subdoccount, is_eval= (name=='eval'))
            self.total_subdocs = len(data['train']) + len(data['eval'])

        else:
            with gzip.open(train_dir, "rt") as ifd:
                for line in tqdm(ifd):
                    self.preprocess_instance(line, data, token2subdoccount)
            print(f"currently having: {sum([value for _, value in token2subdoccount.items()])}")
            self.total_subdocs = sum([len(values) for _, values in data['train'].items()])
            self.split_train_eval(data, train_proportion)

        train_counter = Counter([instance['window'] for instance in data['train']])
        eval_counter = Counter([instance['window'] for instance in data['eval']])

        self.logger.info(f"train counter:\n {train_counter}", )
        self.logger.info(f"evalidation counter:\n {eval_counter}")

        if not id2token or not token2id:
            vocab_to_keep = self.filter_vocabs_to_keep(token2subdoccount)
            id2token = self.filter_windows_to_keep(data, vocab_to_keep=vocab_to_keep)
        else:
            self.logger.info(f" --- among the {len(token2subdoccount)} vocabs, retaining {len(token2id)} in the model")
            self.filter_windows_to_keep(data, token2id=token2id)

        rnn_input = self.get_rnn(id2token,
                                 (len(self.window_counts['eval']) if token2id else len(self.window_counts['train'])))
        self.logger.info(f"----- completes preprocessing data ----- ")
        return id2token, rnn_input, (len(self.window_counts['eval']) if token2id else len(self.window_counts['train'])), len(self.subdoc_counts['train']), len(self.subdoc_counts['eval'])
    
    def split_train_eval(self, data, train_proportion, by_category=True):
        
        self.logger.info(f"----- starts splitting train and eval ----- ")

        random.seed(self.random_seed)

        data['train'] = dict(sorted(data['train'].items()))
        train_data = []
        eval_data = []

        if by_category:
            train_data = []
            eval_data = []
            for _, values in data['train'].items():
                train_count = math.ceil(train_proportion * len(values))
                random.shuffle(values)
                train_data.extend(values[:train_count])
                eval_data.extend(values[train_count:])
        else:
            all_data = []
            for _, values in data['train'].items():
                all_data.extend(values)
            random.shuffle(all_data)
            train_count = math.ceil(train_proportion * len(all_data))
            train_data = all_data[:train_count]
            eval_data = all_data[train_count:]

        data['train'] = train_data
        data['eval'] = eval_data
        self.logger.info(f"----- complete splitting train and eval ----- ")

    def filter_vocabs_to_keep(self, token2subdoccount):

        self.logger.info(f"----- starts filtering vocabs to keep ----- ")
        vocab_to_keep = set()

        for t, count in token2subdoccount.items():
            if (
                count >= self.min_word_occurrence
                and (count / self.total_subdocs) <= self.max_word_proportion
            ):
                vocab_to_keep.add(t)
        self.logger.info(
        "Keeping %d words from a vocabulary of %d",
        len(vocab_to_keep),
        len(token2subdoccount),
        )

        self.logger.info(f"----- complete filtering vocabs to keep ----- ")
        return vocab_to_keep

    def filter_windows_to_keep(self, data, by_category=True, 
                               vocab_to_keep = None, 
                               token2id = None):
    
        self.logger.info(f"----- starts filtering windows to keep ----- ")
        self.subdoc_counts = {}
        self.window_counts = {}
        window_transform = {}
        token2id = token2id if token2id else {}

        for name, vs in data.items():
            self.subdoc_counts.setdefault(name, [])
            self.window_counts.setdefault(name, {})
           
            for subdoc in vs:
                window = subdoc['window_idx'] 
                token_counts = Counter(subdoc["tokens"])
                if vocab_to_keep:
                    subdoc["counts"] = {tid: count
                                        for t, count in token_counts.items() 
                                        if (t in vocab_to_keep 
                                        and (tid := token2id.setdefault(t, len(token2id))))}
                else:
                    subdoc["counts"] = {token2id[t]: count 
                                        for t, count in token_counts.items() 
                                        if t in token2id}
                
                if subdoc["counts"]:
                    self.subdoc_counts[name].append(subdoc)
                    self.window_counts[name][window] = self.window_counts[name].get(window, 0) + 1

        if by_category:
            for name in ['train', 'eval']:
                sorted_by_key = dict(sorted(self.window_counts[name].items()))
                accum_idx = 0
                for key, counts in sorted_by_key.items():
                    windows = [item['window_idx'] for item in self.subdoc_counts[name][accum_idx:accum_idx + counts]]
                    try:
                        assert len(set(windows)) == 1 and windows[0] == int(key)
                    except Exception as e:
                        print(f"expected all instances from {accum_idx} to {accum_idx + counts} to be {key}, but instead got:\n" + 
                                    f"{windows}")
                        raise Exception(str(e))

                    accum_idx += counts
        
        all_windows = set(w for v in self.window_counts.values() for w in v.keys())
        if vocab_to_keep:
            windows_to_keep = {
                w
                for w in self.window_counts['train'].keys()
                if all([w in v for v in self.window_counts.values()])
            }
        else:
            windows_to_keep =   {
                    w
                    for w in self.window_counts['eval'].keys()
                    if all([v for v in self.window_counts['eval'].values()])
                }
            
        self.logger.info(f"windows to keep: {windows_to_keep}")
        self.logger.info(f"windows with no instances: {all_windows - windows_to_keep}")
        window_transform = {w: i for i, w in enumerate(sorted(windows_to_keep))}

        for name in self.subdoc_counts.keys():
            self.subdoc_counts[name] = [
        {**{k: v for k, v in s.items() if k != "window_idx"}, "window_idx": window_transform[s["window_idx"]]}
        for s in self.subdoc_counts[name] if s["window_idx"] in windows_to_keep
    ]
        self.window_counts[name] = {window_transform[k]: v for k, v in self.window_counts[name].items() if k in windows_to_keep}

        id2token = {v: k for k, v in token2id.items()} if vocab_to_keep else None

        self.logger.info(f"----- complete filtering windows to keep ----- ")
        return id2token

    def get_rnn(self, id2token, window_count):
        self.logger.info(f"----- starts get rnn input ----- ")
        rnn_input = {}
        for name in ["train", "eval"]:
            batch_size = self.batch_size if name == 'train' else self.eval_batch_size
            indices = torch.arange(0, len(self.subdoc_counts[name]), dtype=torch.int)
            indices = torch.split(indices, batch_size)
            rnn_input[name] = torch.zeros(window_count, len(id2token)).to(
                self.device
            )
            cnt = torch.zeros(
                window_count,
            ).to(self.device)
            for _, ind in enumerate(indices):
                batch_size = len(ind)
                data_batch = np.zeros((batch_size, len(id2token)))
                times_batch = np.zeros((batch_size,))
                for i, doc_id in enumerate(ind):
                    subdoc = self.subdoc_counts[name][doc_id]
                    times_batch[i] = subdoc["window_idx"]  # timestamp
                    for k, v in subdoc["counts"].items():
                        data_batch[i, k] = v
                data_batch = torch.from_numpy(data_batch).float().to(self.device)
                times_batch = torch.from_numpy(times_batch).to(self.device)
                for t in range(window_count):
                    tmp = (times_batch == t).nonzero()
                    docs = data_batch[tmp].squeeze().sum(0)
                    rnn_input[name][t] += docs
                    cnt[t] += len(tmp)
            rnn_input[name] = rnn_input[name] / cnt.unsqueeze(1)
        self.logger.info(f"----- completes get rnn input ----- ")
        return rnn_input

    def batch_generator(self, vocab_size, is_train=True, by_category=True):
        name = "train" if is_train else "eval"
        batch_size = self.batch_size if is_train else self.eval_batch_size

        self.epoch += 1
        random.seed(self.epoch)
        torch.manual_seed(self.epoch)
        
        if by_category and is_train:
            if len(self.window_counts[name]) > batch_size:
                raise Exception("unable to form batch instances because there are more time windows than single batch size")
            total_batch_num = math.ceil(len(self.subdoc_counts[name]) / batch_size)
            batch_indices_list = [[] for _ in range(total_batch_num)]
            sorted_by_key = dict(sorted(self.window_counts[name].items()))
            buffer = 0

            for idx, (_, curr_count) in enumerate(sorted_by_key.items()):
                if curr_count == 0:
                    continue
              
                prev_section = buffer
                buffer += curr_count
                cur_section = buffer

                all_indices = torch.arange(prev_section, cur_section)
                division = total_batch_num // len(all_indices)
                if division > 0:
                    all_indices = all_indices.repeat(division)
                num_sample = total_batch_num - len(all_indices) if division > 0 else total_batch_num
                sampled_indices = all_indices[torch.randint(0, len(all_indices), (num_sample,))]
                all_indices = torch.cat([all_indices, sampled_indices]) if division > 0 else sampled_indices
                permuted_indices = all_indices[torch.randperm(all_indices.size(0))]

                for idx in range(len(permuted_indices)):
                    batch_indices_list[idx].append(permuted_indices[idx])
        
            total_used_idx = set(idx for batch in batch_indices_list for idx in batch)
            remaining_indices = [idx for idx in range(len(self.subdoc_counts[name])) if idx not in total_used_idx]
            remaining_indices = torch.tensor(remaining_indices)[torch.randperm(len(remaining_indices))]
            ptr = 0

            for idx in range(len(batch_indices_list)):
                num_missing_items = batch_size - len(batch_indices_list[idx])
                batch_indices_list[idx].extend(remaining_indices[ptr:ptr+num_missing_items])
                ptr += num_missing_items
        else:
            all_batch_indices = torch.randperm(len(self.subdoc_counts[name]))
            batch_indices_list = torch.split(all_batch_indices, batch_size) 

        all_missing_time_counter = Counter()

        for idx, batch_indices in enumerate(batch_indices_list):
            current_batch_size = len(batch_indices)
            data_batch = torch.zeros((current_batch_size, vocab_size), device=self.device)
            times_batch = torch.zeros((current_batch_size,), device=self.device)
            for i, doc_id in enumerate(batch_indices):
                subdoc = self.subdoc_counts[name][doc_id]
                times_batch[i] = subdoc["window_idx"]
                for k, v in subdoc["counts"].items():
                    data_batch[i, k] = v

            if is_train:
                time_included = Counter(times_batch.tolist()).keys()
                time_missed = [window_range for (idx, window_range) in enumerate(self.all_window_ranges) if idx not in time_included]
                all_missing_time_counter.update(time_missed)

                if idx == len(batch_indices_list) - 1:
                    self.logger.info(f"all missing time in the current counter: {all_missing_time_counter}")

            sums = data_batch.sum(dim=1, keepdim=True)
            sums[sums == 0] = 1
            normalized_data_batch = data_batch / sums

            yield data_batch, normalized_data_batch, times_batch, batch_indices
        
    def update_subdoc_counts(self, liks, inds, token2id):
        try:
            for lik, ind in zip(liks, inds):
                lik = lik.argmax(0)
                self.subdoc_counts['eval'][ind]["tokens"] = [
                    (
                            (tok, lik[token2id[tok]].item())
                            if tok in token2id
                            else (tok, None)
                        )
                        for tok in self.subdoc_counts['eval'][ind]["tokens"]
                    ]
                del self.subdoc_counts['eval'][ind]["counts"]
        except Exception as e:
            self.logger.info(f"received error when enumerating text : {str(e)}")
            raise Exception(e)