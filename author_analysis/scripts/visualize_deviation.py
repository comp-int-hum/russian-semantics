import argparse, numpy, logging, json

logger = logging.getLogger("visualize deviation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--id2auth_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        # filename="output.out"
    )

    with open(args.id2auth_dir, "r") as f:
        id2auth = json.load(f)

    id2auth = {int(k): v for k, v in id2auth.items()}

    # first use js matrix to get the particular row and col information needed
    js_matrix = numpy.load(args.input_dir)
    num_author, num_topic, num_window = js_matrix.shape
    per_win_per_top_auth_list = {}
    for top_id in range(num_topic):
        for win_id in range(num_window):
            auth_data = [
                (auth_id, auth_js)
                for auth_id, auth_js in enumerate(js_matrix[:, top_id, win_id])
                if auth_js != 2
            ]
            if len(auth_data) < 10:
                per_win_per_top_auth_list[f"top{top_id}-win{win_id}"] = {
                    "deviate": [],
                    "close": [],
                }
                continue

            deviate_auths = [
                f"id{auth_id}-{id2auth[auth_id]}"
                for auth_id, _ in sorted(auth_data, key=lambda x: x[1], reverse=True)[
                    : args.top_n
                ]
            ]
            close_auths = [
                f"id{auth_id}-{id2auth[auth_id]}"
                for auth_id, _ in sorted(auth_data, key=lambda x: x[1], reverse=True)[
                    -args.top_n :
                ]
            ]

            per_win_per_top_auth_list[f"top{top_id}-win{win_id}"] = {
                "deviate": deviate_auths,
                "close": close_auths,
            }

    with open(args.output_dir, "w") as f:
        json.dump(per_win_per_top_auth_list, f, indent=4, ensure_ascii=False)
