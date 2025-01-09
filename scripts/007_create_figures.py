import argparse, gzip, warnings, pickle, logging, numpy, ruptures, requests
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

warnings.simplefilter("ignore")

# used for matrix division
EPSILON = 1e-10

logger = logging.getLogger("create_figures")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--log", dest="log")
    parser.add_argument("--top_n", dest="top_n", type=int, default=8)
    parser.add_argument("--topics_per_plot", dest="topics_per_plot", type=int, default=2)
    parser.add_argument("--step_size", dest="step_size", type=int, default=1)
    parser.add_argument("--temporal_image", dest="temporal_image", help="Output file")
    parser.add_argument("--latex", dest="latex", help="Output file")
    parser.add_argument("--figure_type", dest="figure_type", default="default")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log)

    # load in the various precomputed counts/lookups/etc    
    label_size = 20
    tick_size = 20
    title_size = 25
    with gzip.open(args.input, "rb") as ifd:
        precomp = pickle.loads(ifd.read())        

    # unpack the precomputed info

    # the start year and years-per-window (integers)
    start = precomp["start"]
    window_size = precomp["window_size"]

    # dictionary lookups for authors, words, htids
    id2author = precomp["id2author"]
    author2id = {v : k for k, v in id2author.items()}
    id2htid = precomp["id2htid"]
    htid2id = {v : k for k, v in id2htid.items()}
    id2word = precomp["id2word"]
    word2id = {v : k for k, v in id2word.items()}

    # 3-d count matrices (this could have been one 4-d matrix, but since authors only occur in one window it would be inefficient)
    word_win_topic = precomp["wwt"]
    auth_win_topic = precomp["awt"]
    htid_win_topic = precomp["hwt"]

    word_counts = word_win_topic.sum(1).sum(1)

    if args.figure_type == "topic_per_window_dist":
        # sum up all the words over every window
        per_window_topic_dist = htid_win_topic.sum(0)
        per_window_topic_dist = (per_window_topic_dist.T / per_window_topic_dist.sum(1)).T

        logger.info(per_window_topic_dist.shape)
        logger.info(per_window_topic_dist[0, :])
        logger.info(per_window_topic_dist[:, 0])

        fig, ax = plt.subplots(figsize=(10, 6))
        time_windows = numpy.arange(5)
        cmap = plt.get_cmap('tab20')  # Use 'tab20' for 20 distinct colors
        colors = cmap(numpy.arange(20))  # Get 20 colors from the colormap

        # Use stackplot to create the area plot
        ax.stackplot(time_windows, per_window_topic_dist.T, labels=[f"Category {i}" for i in range(20)], alpha=0.8, colors=colors)

        # Step 4: Customize plot
        ax.set_title("Category Proportions Over Time Windows", fontsize=14)
        ax.set_xlabel("Time Windows", fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)


        ax.set_xticks([0, 1, 2, 3, 4])  # Set 5 ticks corresponding to the 5 time windows
        ax.set_xticklabels(['1860', '1870', '1880', '1890', "1900"]) 

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)  # Adjust legend position

        plt.tight_layout()
        plt.savefig(args.output)
    if args.figure_type == "dist_divergence":
        # try implementing KL divergence and genet divergence

        # step 1: get the distribution of word per topic for each time
        # shape of word_win_topic: (87504, 5, 20)
        word_win_topic_sum_per_time_per_topic = word_win_topic.sum(axis=0)
        word_win_topic_dist = word_win_topic / word_win_topic_sum_per_time_per_topic
        divergence_arr = numpy.zeros((3, 4, 20))
        epsilon = 1e-10

        # for each topic for each time interval, get the KL divergence and the genet divergence
        for topic_idx in range(word_win_topic.shape[2]):
            for time_idx in range(word_win_topic.shape[1] - 1):
                time_p, time_q = time_idx, time_idx + 1

                # KL(P || Q) =  sum x in X P(x) * log(P(x) / Q(x))
                p_x = word_win_topic_dist[:, time_p, topic_idx]
                q_x = word_win_topic_dist[:, time_q, topic_idx]
                p_x = numpy.clip(p_x, epsilon, 1)
                q_x = numpy.clip(q_x, epsilon, 1)

                # forward
                divergence_arr[0, time_p, topic_idx] = numpy.sum(p_x * numpy.log(p_x / q_x))
                # backward
                divergence_arr[1, time_p, topic_idx] = numpy.sum(q_x * numpy.log(q_x / p_x))

                # JS(P || Q) = 1/2 KL(P || M) + 1/2 KL(Q || M)
                # where M = 1/2 (P + Q)
                m_x = (p_x + q_x) / 2
                m_x = numpy.clip(m_x, epsilon, 1)
                divergence_arr[2, time_p, topic_idx] = numpy.sum(p_x * numpy.log(p_x / m_x)) / 2 + numpy.sum(q_x * numpy.log(q_x / m_x)) / 2
        
        logger.info(divergence_arr)

        reshaped_array = divergence_arr.reshape(12, 20)
        indexed_array = numpy.column_stack((numpy.repeat(numpy.arange(3), 4), reshaped_array))
        numpy.savetxt("output.csv", indexed_array, delimiter=",", header="divergence+idx, " + ", ".join([f"topic #{idx}" for idx in range(20)]))
    
        # create graphs
        time_intervals = numpy.arange(1, 5)  # 4 time intervals
        cmap = plt.get_cmap('tab10')  # Just use one argument here
        colors = [cmap(i % cmap.N) for i in range(20)]

        for i in range(20 // args.topics_per_plot):  # 4 plots, each with 5 topics
            plt.figure(figsize=(10, 6))

            scaled_divergence = [(divergence_arr[fb_idx, :, topic_idx] - divergence_arr[fb_idx, :, topic_idx].min()) / (divergence_arr[fb_idx, :, topic_idx].max() - divergence_arr[fb_idx, :, topic_idx].min() + epsilon) for topic_idx in range(20) for fb_idx in range(2)]
    
            for j in range(args.topics_per_plot):
                topic_idx = i * args.topics_per_plot + j
                plt.plot(time_intervals, scaled_divergence[topic_idx * 2], label=f"Topic {topic_idx + 1} DL Forward", linestyle='-', linewidth=1.5, color=colors[topic_idx])
                plt.plot(time_intervals, scaled_divergence[topic_idx * 2 + 1], label=f"Topic {topic_idx + 1} DL Backward", linestyle=':', linewidth=1.5, color=colors[topic_idx])
                plt.plot(time_intervals, divergence_arr[2, :, topic_idx], label=f"Topic {topic_idx + 1} JS", linestyle='-.', linewidth=1.5, color=colors[topic_idx])
            
            plt.xlabel("Time Interval")
            plt.ylabel("Divergence Value")
            plt.title(f"Topics {i * args.topics_per_plot + 1} to {(i + 1) * args.topics_per_plot} KL & JS Divergence")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"images/topics_{i * args.topics_per_plot + 1}_to_{(i + 1) * args.topics_per_plot}_KL_&_JS_divergence.png")

            plt.figure(figsize=(10, 6))
    
            for j in range(args.topics_per_plot):
                topic_idx = i * args.topics_per_plot + j
                plt.plot(time_intervals, divergence_arr[0, :, topic_idx], label=f"Topic {topic_idx + 1} DL Forward", linestyle='-', linewidth=1.5, color=colors[topic_idx])
                plt.plot(time_intervals, divergence_arr[1, :, topic_idx], label=f"Topic {topic_idx + 1} DL Backward", linestyle=':', linewidth=1.5, color=colors[topic_idx])
                # plt.plot(time_intervals, divergence_arr[2, :, topic_idx], label=f"Topic {topic_idx + 1} JS", linestyle='-.', linewidth=1.5, color=colors[topic_idx])
            
            plt.xlabel("Time Interval")
            plt.ylabel("Divergence Value")
            plt.title(f"Topics {i * args.topics_per_plot + 1} to {(i + 1) * args.topics_per_plot} KL Divergence")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"images/topics_{i * args.topics_per_plot + 1}_to_{(i + 1) * args.topics_per_plot}_KL_divergence.png")


    if args.figure_type == "default":
        win_topic = word_win_topic.sum(0)
        win_topic = (win_topic.T / win_topic.sum(1)).T

        # each author's distribution over topics
        auth_topic_dist = auth_win_topic.sum(1)
        auth_topic_dist = (auth_topic_dist.T / auth_topic_dist.sum(1)).T

        # each author's *preceding* window's distribution over topics
        auth_wins = auth_win_topic.sum(2).argmax(1)
        auth_prev_background_topic_dist = win_topic[auth_wins - 1]

        # each topic's top word ids
        topic_word = numpy.transpose(word_win_topic.sum(1), axes=(1, 0))
        topic_word_dist = (topic_word.T / topic_word.sum(1)).T
        topic_word_ids = numpy.flip(topic_word_dist.argsort(1)[:, -args.top_n:], 1)

        # each word's distribution over topics over time
        word_win = word_win_topic.sum(2)
        word_win_topic_dist = numpy.zeros(shape=word_win_topic.shape)
        for win in range(word_win.shape[1]):
            word_win_topic_dist[:, win, :] = (word_win_topic[:, win, :].T / word_win[:, win]).T
        word_win_topic_maxes = numpy.sort(word_win_topic_dist, axis=2)[:, :, -2:]

        words = []
        word_win_modality = (word_win_topic_maxes[:, :, -1] - word_win_topic_maxes[:, :, -2]) + (1 - (word_win_topic_maxes[:, :, -1] + word_win_topic_maxes[:, :, -2]))


        word_modality_changepoint_delta = numpy.zeros(shape=(word_win_modality.shape[0],))
        for i, mod in enumerate(word_win_modality):
            if numpy.isnan(mod).sum() < 3:
                modalities = numpy.array([v for v in mod if not numpy.isnan(v)])
                #ent = word_win_entropies[i]
                cps = ruptures.Dynp(model="l2", min_size=1, jump=1).fit_predict(modalities.T, 1)
                cp = cps[0]
                cpd = abs(modalities[:cp].mean() - modalities[cp:].mean())
                if word_counts[i] > 50 and cpd > 0.0 and cpd < 1.0:
                    words.append((cpd, cp, modalities.mean(), modalities.std(), id2word[i], word_counts[i]))
                    #print(id2word[i], word_counts[i])

        with open(args.latex, "wt") as latex_ofd, open(args.output, "wb") as temporal_image_ofd:

            width = 12
            height = 6
        
            window_modality_counts = {}
            words_by_modality_shift = list(reversed(sorted(words)))
            for cpd, cp, mmean, mstd, w, c in words:
                year = start + window_size * cp
                window_modality_counts[year] = window_modality_counts.get(year, 0.0) + cpd

            # plot modality shift over time
            fig = Figure(figsize=(width, height))
            ax = fig.add_subplot(frameon=False)
            pairs = list(sorted(window_modality_counts.items()))

            ax.plot([x for x, _ in pairs], [y for _, y in pairs], linewidth=5)
            ax.set_xticks([x for x, _ in pairs], labels=[str(x) for x, _ in pairs], fontsize=tick_size)
            ax.set_xlabel("Year", fontsize=label_size)
            ax.set_ylabel("Amount of bimodal shift", fontsize=label_size)
            fig.savefig(temporal_image_ofd, bbox_inches="tight")


            latex_ofd.write("""\\begin{tabular}{l l l}\n""")
            latex_ofd.write("""\\hline\n""")
            latex_ofd.write("""Word & Changepoint & Delta \\\\\n""")
            latex_ofd.write("""\\hline\n""")
            for cpd, cp, _, _, w, c in words_by_modality_shift[:10]:
                w_bilingual = translate_text_deepl(w.lower())
                year = start + window_size * cp
                latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(w_bilingual, year, cpd))
            latex_ofd.write("""\\hline\n""")
            for cpd, cp, _, _, w, c in words_by_modality_shift[-10:]:
                w_bilingual = translate_text_deepl(w.lower())
                year = start + window_size * cp
                latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(w_bilingual, year, cpd))
            latex_ofd.write("""\\hline\n""")
            latex_ofd.write("""\\end{tabular}\n""")


            jsds = jensenshannon(auth_prev_background_topic_dist, auth_topic_dist, axis=1)
            authors_by_novelty = []
            for i in jsds.argsort():
                authors_by_novelty.append((id2author[i], jsds[i], auth_wins[i]))
            authors_by_novelty = list(reversed(authors_by_novelty))


            latex_ofd.write("""\\begin{tabular}{l l}\n""")
            latex_ofd.write("""\\hline\n""")
            latex_ofd.write("""Author & JSD \\\\\n""")
            latex_ofd.write("""\\hline\n""")
            for name, val, cp in authors_by_novelty[:5]:
                year = start + window_size * cp
                latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(name, year, val))
            latex_ofd.write("""\\hline\n""")
            for name, val, cp in authors_by_novelty[-5:]:
                year = start + window_size * cp
                latex_ofd.write("""{} & {} & {:.3f} \\\\\n""".format(name, year, val))
            latex_ofd.write("""\\hline\n""")
            latex_ofd.write("""\\end{tabular}\n""")


            # topic evolutions
            topic_win_word = numpy.transpose(word_win_topic, axes=(2, 1, 0))
            indices = numpy.arange(topic_win_word.shape[1], step=args.step_size)

            for tid, topic in enumerate(topic_win_word):
                tt = (topic.T / topic.sum(1)).T
                top_words = numpy.flip(topic.argsort(1), 1)

                topic_states = []
                for j in indices:
                    word_ids = top_words[j][:args.top_n]
                    #topic_states.append(["{}:{:.03}".format(id2word[wid], tt[j][wid]) for wid in word_ids])
                    topic_states.append(["{}".format(
                        translate_text_deepl(id2word[wid].lower())) for wid in word_ids])

                latex_ofd.write(
                    """\\topicevolution{%s}{0}{0}{{%s}}\n""" % (
                        tid,
                        ",".join(["{%s}" % (",".join(state)) for state in topic_states])
                    ) + "\n"
                )
            