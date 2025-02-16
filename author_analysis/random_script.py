import numpy, csv
from tqdm import tqdm


def data_generator(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in tqdm(reader):
            yield next(reader)


if __name__ == "__main__":
    filename = "data/auth_voc_win_top_one_hot/auth_mat.csv"

    gen = data_generator(filename)
    out_data = []

    for _ in range(100):
        out_data.append(next(gen))

    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["author_id", "topic_id", "vocab_id", "window_id", "counter"])
        # writer.writerow([num_topics, num_words, num_windows, -1])
        for data in out_data:
            writer.writerow(data)
