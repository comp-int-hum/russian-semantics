import argparse, numpy, logging, json

logger = logging.getLogger("find deviation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--id2auth_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename="output.out")

    with open(args.id2auth_dir, "r") as f:
        id2auth = json.load(f)

    id2auth = {int(k): v for k, v in id2auth.items()}

    js_matrix = numpy.load(args.input_dir)
    num_author, num_topic, num_window = js_matrix.shape

    for top_id in range(num_topic):
        for win_id in range(num_window):
            auth_data = [
                (auth_id, auth_js)
                for auth_id, auth_js in enumerate(js_matrix[:, top_id, win_id])
                if auth_js != 2
            ]
            if len(auth_data) < 10:
                continue
            deviate_auths = [
                id2auth[auth_id]
                for auth_id, _ in sorted(auth_data, key=lambda x: x[1], reverse=True)[
                    :3
                ]
            ]
            close_auths = [
                id2auth[auth_id]
                for auth_id, _ in sorted(auth_data, key=lambda x: x[1], reverse=True)[
                    -3:
                ]
            ]

            logger.info(
                f"the 3 authors whose use of words most deviate from topic {top_id} in time window {win_id} are:\n {deviate_auths}"
            )

            logger.info(
                f"the 3 authors whose use of words most align with topic {top_id} in time window {win_id} are:\n {close_auths}"
            )
