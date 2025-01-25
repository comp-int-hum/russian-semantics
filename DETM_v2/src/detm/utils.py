import copy, logging, random, torch, math, numpy, gzip, pickle
from tqdm import tqdm
from torch import autograd

logger = logging.getLogger("utils")

def _yield_data(subdocs, times, vocab_size, batch_size=64):
    assert len(subdocs) == len(times)
    
    indices = torch.randperm(len(subdocs))
    indices = torch.split(indices, batch_size)

    for ind in tqdm(indices):
        actual_batch_size = len(ind)
        data_batch = numpy.zeros((actual_batch_size, vocab_size))
        times_batch = numpy.zeros((actual_batch_size, ))

        for i, doc_id in enumerate(ind):
            subdoc = subdocs[doc_id]
            times_batch[i] = times[doc_id]
            for k, v in subdoc.items():
                data_batch[i, k] = v
        data_batch = torch.from_numpy(data_batch).float()
        times_batch = torch.from_numpy(times_batch)
        # sums = data_batch.sum(1).unsqueeze(1)
        # normalized_data_batch = data_batch / sums

        yield times_batch, data_batch, ind
        # sums, normalized_data_batch

def train_model(
        subdocs,
        times,
        model,
        optimizer,
        max_epochs,
        clip=2.0,
        lr_factor=2.0,
        batch_size=32,
        device="cpu",
        val_proportion=0.2,
        detect_anomalies=False
):
    #times = [model.represent_time(t) for t in times]
    model = model.to(device)
    
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    val_subdocs = [x for x, _ in pairs[:int(val_proportion*len(subdocs))]]

    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    val_times = [x for _, x in pairs[:int(val_proportion*len(times))]]
    
    best_state = copy.deepcopy(model.state_dict())
    best_val_ppl = float("inf")
    since_annealing = 0
    since_improvement = 0
    for epoch in range(1, max_epochs + 1):
        logger.info("Starting epoch %d", epoch)
        model.train(True)
        model.prepare_for_data(train_subdocs, train_times)
        
        acc_loss = 0
        acc_nll = 0
        acc_kl_theta_loss = 0
        acc_kl_eta_loss = 0
        acc_kl_alpha_loss = 0
        cnt = 0

        train_generator = _yield_data(train_subdocs, train_times, model.vocab_size,
                                      batch_size)
        while True:
            try:
                times_batch, data_batch, _ = next(train_generator)
                optimizer.zero_grad()
                model.zero_grad()

                with autograd.set_detect_anomaly(detect_anomalies):

                    loss, nll, kl_alpha, kl_eta, kl_theta = model(
                        data_batch,
                        times_batch,
                    )
                    loss.backward()
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                acc_loss += torch.sum(loss).item()
                acc_nll += torch.sum(nll).item()
                acc_kl_theta_loss += torch.sum(kl_theta).item()
                acc_kl_eta_loss += torch.sum(kl_eta).item()
                acc_kl_alpha_loss += torch.sum(kl_alpha).item()
                cnt += data_batch.shape[0]
            
            except StopIteration:
                break

        cur_loss = round(acc_loss / cnt, 2) 
        cur_nll = round(acc_nll / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
        cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
        lr = optimizer.param_groups[0]['lr']

        logger.info("Computing perplexity...")
        _, val_ppl = apply_model(
            model,
            val_subdocs,
            val_times,
            batch_size,
            detect_anomalies=detect_anomalies
        )
        logger.info(
            '{}: LR: {}, Train doc losses: mix_prior={:.3f}, mix={:.3f}, embs={:.3f}, recon={:.3f}, NELBO={:.3f} Val doc ppl: {:.3f}'.format(
                epoch,
                lr,
                cur_kl_eta,
                cur_kl_theta,
                cur_kl_alpha,
                cur_nll,
                cur_loss,
                val_ppl
            )
        )

        if val_ppl < best_val_ppl:
            logger.info("Copying new best model...")
            best_val_ppl = val_ppl
            best_state = copy.deepcopy(model.state_dict())
            since_improvement = 0
            logger.info("Copied.")
        else:
            since_improvement += 1
        since_annealing += 1
        if since_improvement > 5 and since_annealing > 5 and since_improvement < 10:
            optimizer.param_groups[0]['lr'] /= lr_factor
            model.load_state_dict(best_state)
            since_annealing = 0
        elif numpy.isnan(val_ppl):
            logger.error("Perplexity was NaN: reducing learning rate and trying again...")
            model.load_state_dict(best_state)
            optimizer.param_groups[0]['lr'] /= lr_factor
        elif since_improvement >= 10:
            break

    return best_state

def test_for_lr(
        subdocs,
        times,
        model_class,
        word_list,
        min_time, max_time,
        num_topics, window_size,
        learning_rate, wdecay,
        embeddings,
        clip=2.0,
        lr_factor=2.0,
        batch_size=32,
        device="cpu",
        optimizer_type='adam',
        val_proportion=0.2,
        detect_anomalies=False
):
    
    # initialize data
    pairs = list(zip(subdocs, times))
    random.shuffle(pairs)
    
    train_subdocs = [x for x, _ in pairs[int(val_proportion*len(subdocs)):]]
    train_times = [x for _, x in pairs[int(val_proportion*len(times)):]]
    
    train_generator = _yield_data(train_subdocs, train_times, 
                                  len(word_list), batch_size)
    
    nan_flag = True
    while nan_flag:
        logger.info(f"starting new trial. Current learning rate {learning_rate}")

        acc_loss, acc_nll, acc_kl_theta_loss = 0, 0, 0
        acc_kl_eta_loss, acc_kl_alpha_loss, cnt = 0, 0, 0

        nan_flag = False
        model = model_class(
        word_list=word_list,
        num_topics=num_topics,
        window_size=window_size,
        min_time=min_time,
        max_time=max_time,
        embeddings=embeddings,
    )

        model = model.to(device)
        model.prepare_for_data(train_subdocs, train_times)

        optimizer = (torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wdecay) 
                     if optimizer_type == 'adam' 
                     else torch.optim.SGD(model.parameters(), lr=learning_rate,  weight_decay=wdecay))
  
        batch_idx = 0
        while nan_flag:
            try:
                times_batch, data_batch, _ = next(train_generator)
                batch_idx += 1
        
                with autograd.set_detect_anomaly(detect_anomalies):

                    loss, nll, kl_alpha, kl_eta, kl_theta = model(
                        data_batch,
                        times_batch,
                    )
                    
                    if torch.isnan(loss).any():
                    
                        logger.info(f"got nan in batch #{batch_idx} with lr {learning_rate}, reducing ...")
                        learning_rate /= lr_factor 
                        nan_flag = True
                        del model
                        del optimizer
                        break
                    loss.backward()
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                acc_loss += torch.sum(loss).item()
                acc_nll += torch.sum(nll).item()
                acc_kl_theta_loss += torch.sum(kl_theta).item()
                acc_kl_eta_loss += torch.sum(kl_eta).item()
                acc_kl_alpha_loss += torch.sum(kl_alpha).item()
                cnt += data_batch.shape[0]
            except StopIteration:
                break

        if not nan_flag:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            nan_flag = False
        
            logger.info(
                'LR: {}, Train doc losses: mix_prior={:.3f}, mix={:.3f}, embs={:.3f}, recon={:.3f}, NELBO={:.3f}'.format(
                    lr,
                    cur_kl_eta,
                    cur_kl_theta,
                    cur_kl_alpha,
                    cur_nll,
                    cur_loss
                )
            )

    return learning_rate

def apply_model(
        model,
        subdocs,
        times,
        batch_size=32,
        device="cpu",
        detect_anomalies=False
):
    model.train(False)
    model.prepare_for_data(subdocs, times)

    ppl = 0
    cnt = 0
    appl_generator = _yield_data(subdocs, times, model.vocab_size, batch_size)
    while True:
        try:
            times_batch, data_batch, _ = next(appl_generator)
            with autograd.set_detect_anomaly(detect_anomalies):
                _, nll, _, _, _ = model(
                    data_batch,
                    times_batch,
                )

                ppl += torch.sum(nll).item()
                cnt += data_batch.shape[0]
        except StopIteration:
            break
    return (), ppl / cnt

def get_matrice(model, subdocs, times, auxiliaries,
                output_dir, batch_size,
                workid_field, time_field, 
                author_field=None, workname_field=None,
                logger=None, get_prob=False,
                detect_anomalies=False):

    if logger:
        logger.info("----- starts applying model ------ ")

    min_time, max_time, window_size = model.min_time, model.max_time, model.window_size
    num_topics, token_list = model.num_topics, model.word_list
    subdoc_topics = [[
        (idx, None) for idx, _ in subdoc.items()] 
        for subdoc in subdocs]
        
    model.train(False)
    model.prepare_for_data(subdocs, times)

    appl_generator = _yield_data(subdocs, times, model.vocab_size, batch_size)

    while True:
        try:
            times_batch, data_batch, inds = next(appl_generator)
            with autograd.set_detect_anomaly(detect_anomalies):
                liks = model.get_likelihood(data_batch, times_batch)
            
                for lik, ind in zip(liks, inds):
                    if get_prob:
                        subdoc_topics[ind] = [
                            (idx, lik[:, idx].cpu().detach().numpy())
                            for idx, _ in subdoc_topics[ind]
                        ]
                    else:
                        lik = lik.argmax(0)
                        subdoc_topics[ind] = [
                            (idx, lik[idx].item())
                            for idx, _ in subdoc_topics[ind]
                            ]
        
        except StopIteration:
            break

    if logger:
        logger.info("----- completes applying model ------ ")
        logger.info(f"----- starts getting matrice, using {'likihood probability' if get_prob else 'most likely topic'}  ------ ")
    
    docs, doc2year = {}, {}
    token2id = {tok: idx for idx, tok in enumerate(token_list)}

    if author_field:
        doc2author, unique_authors = {}, set()
    if workname_field:
        doc2title = {}
    
    for subdoc, auxiliary in tqdm(zip(subdoc_topics, auxiliaries)):
        workid = auxiliary[workid_field]
        docs.setdefault(workid, [])
        docs[workid].append(subdoc)
        doc2year[workid] = auxiliary[time_field]
 
        if author_field:
            author = auxiliary[author_field]
            doc2author[workid] = author
            unique_authors.add(author)
        
        if workname_field:
            doc2title[workid] = auxiliary[workname_field]
        

    # print(window_rangs)
    nwins, nwords, ntopics, nauths, ndocs = (math.ceil((max_time - min_time) / window_size), 
                                             len(token2id), num_topics, 
                                             len(unique_authors), len(docs))
    
    workid2id = {d : i for i, d in enumerate(docs.keys())}
    time2window = {}
    
    if author_field:
        author2id = {a : i for i, a in enumerate(unique_authors)}

    if logger:
        logger.info(
            f"Found {nwins} windows, {nwords} unique words, {ntopics} unique topics, " +
            f"{ndocs} unique documents, and {nauths} unique authors"
        )

    words_wins_topics = numpy.zeros(shape=(nwords, nwins, ntopics))
    works_wins_topics = numpy.zeros(shape=(ndocs, nwins, ntopics))
    works_words_topics = numpy.zeros(shape=(ndocs, nwords, ntopics))

    if author_field:
        auths_wins_topics = numpy.zeros(shape=(nauths, nwins, ntopics))
        auths_words_topics = numpy.zeros(shape=(nauths, nwords, ntopics))

    for workid, subdocs in docs.items():
        win = (doc2year[workid] - min_time) // window_size
        idx = workid2id[workid]
        
        for subdoc in subdocs:
            for wid, topic in subdoc:
                if get_prob and isinstance(topic, numpy.ndarray):
                    words_wins_topics[wid, win] += topic
                    works_wins_topics[idx, win] += topic
                    works_words_topics[idx, wid] += topic

                    if author_field:
                        aid = author2id[doc2author[workid]]
                        auths_words_topics[aid, wid] += topic
                        auths_wins_topics[aid, win] += topic

                elif (not get_prob) and topic != None:
                    words_wins_topics[wid, win, topic] += 1
                    works_wins_topics[idx, win, topic] += 1
                    works_words_topics[idx, wid, topic] += 1

                    if author_field:
                        aid = author2id[doc2author[workid]]
                        auths_words_topics[aid, wid, topic] += 1
                        auths_wins_topics[aid, win, topic] += 1
    
    del docs             

    if logger:
        logger.info(f"----- completes getting matrice. Writing matrice to {output_dir} ------ ")

    with gzip.open(output_dir, 'wb') as f:

        pickle.dump({"word_win_top": words_wins_topics}, f)
        del words_wins_topics
        pickle.dump({"work_win_top": works_wins_topics}, f)
        del works_wins_topics
        pickle.dump({"work_word_top": works_words_topics}, f)
        del works_words_topics

        if author_field:
            pickle.dump({"auth_word_top": auths_words_topics}, f)
            del auths_words_topics 
            pickle.dump({"auth_win_top": auths_wins_topics}, f)
            del auths_wins_topics 

            pickle.dump({"id2author": {i: a for a, i in author2id.items()}}, f)
            del author2id 
            pickle.dump({"doc2author": doc2author}, f)
            del doc2author
        
        pickle.dump({"id2word": {i: w for w, i in token2id.items()}}, f)
        del token2id
        pickle.dump({"id2workid": {i: d for d, i in workid2id.items()}}, f)
        del workid2id 
        pickle.dump({"doc2year": doc2year}, f)
        del doc2year 

        if workname_field:
            pickle.dump({"doc2title": doc2title}, f)
            del doc2title 

        pickle.dump({"start_time": min_time}, f)
        pickle.dump({"end_time": max_time}, f)
        pickle.dump({"window_size": window_size}, f)

    if logger:
        logger.info(f"----- completes writing to {output_dir} ------ ")

def get_top_topic_info(matrice, top_n=5):
        min_time, max_time, window_size = matrice['start_time'], matrice['end_time'], matrice['window_size']
        start_finish_per_window = [(min_time + idx * window_size, 
                                (min_time + (idx + 1) * window_size
                                 if (min_time + (idx + 1) * window_size <= max_time) 
                                 else max_time)
                                 )
                                for idx in range(math.ceil((max_time - min_time) / window_size))]
        range_str_per_window = [f"{start_year}-{end_year}" 
                                for (start_year, end_year) in start_finish_per_window]
        
        nwords, nwins, ntopics = matrice['wwt'].shape
        
        def _get_prop_and_dist_helper(data_window_topic_col, id2data_col, top_n, epsilon=1e-6):

            data_window_topic, id2data = matrice[data_window_topic_col], matrice[id2data_col]
            return_data = {}

            topic_dist_per_data = (data_window_topic.transpose(2, 0, 1) / 
                                   (data_window_topic.sum(2) + epsilon)).transpose(1, 2, 0)

            data_dist_per_top = data_window_topic / (data_window_topic.sum(0) + epsilon)

            const_str = "Data proportion in single topic "

            for topic_idx in range(ntopics):
                for temp_idx in range(nwins):
                    score_per_topic_per_time = data_dist_per_top[:, temp_idx, topic_idx]
                    top_n_indices = numpy.argsort(score_per_topic_per_time)[-top_n:][::-1]
                    top_n_data = score_per_topic_per_time[top_n_indices]
                    key = const_str + f"#{topic_idx} time {range_str_per_window[temp_idx]}"
                    return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            # only per topic
            for topic_idx in range(ntopics):
                # summing over the time axis
                score_per_topic = (data_dist_per_top.sum(1))[:, topic_idx]
                top_n_indices = numpy.argsort(score_per_topic)[-top_n:][::-1]
                top_n_data = score_per_topic[top_n_indices]
                key = const_str + f"#{topic_idx} across all window"
                return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            const_str = "Topic proportion in single data "

            for topic_idx in range(ntopics):
                for temp_idx in range(nwins):
                    score_per_topic_per_time = topic_dist_per_data[:, temp_idx, topic_idx]
                    top_n_indices = numpy.argsort(score_per_topic_per_time)[-top_n:][::-1]
                    top_n_data = score_per_topic_per_time[top_n_indices]
                    key = const_str + f"#{topic_idx} time {range_str_per_window[temp_idx]}"
                    return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            # only per topic
            for topic_idx in range(ntopics):
                # summing over the time axis
                score_per_topic = (topic_dist_per_data.sum(1))[:, topic_idx]
                top_n_indices = numpy.argsort(score_per_topic)[-top_n:][::-1]
                top_n_data = score_per_topic[top_n_indices]
                key = const_str + f"#{topic_idx} across all window"
                return_data[key] = {
                        f"# {idx}": {'score': round(top_n_data[idx], 4), 
                                    'name': id2data[top_n_indices[idx]]}
                        for idx in range(top_n)}
        
            return return_data
        
        data = {}
        data['per work'] = _get_prop_and_dist_helper("hwt", "id2htid", top_n)
        data['per vocab'] = _get_prop_and_dist_helper("wwt", "id2word", top_n)
        data['per author'] = _get_prop_and_dist_helper("awt", "id2author", top_n)
        return data