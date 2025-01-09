from detm import DETM_Matrice
import argparse, logging, torch, json

# used for matrix division
EPSILON = 1e-10

logger = logging.getLogger("create_figures")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--min_time", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--content_field", type=str, required=True)
    parser.add_argument("--time_field", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_subdoc_length", type=int, default=200)
    parser.add_argument("--min_word_occurrence", type=int, default=0)
    parser.add_argument("--max_word_proportion", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--device")  # , choices=["cpu", "cuda"], help='')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log)

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    matrice = DETM_Matrice(args.model, args.input)
    matrice.get_matrice(args.device, args.min_time, args.max_time, args.window_size,
                    args.num_topics, args.content_field, args.time_field,
                    args.batch_size, args.max_subdoc_length, 
                    args.min_word_occurrence, args.max_word_proportion,
                    logger, args.random_seed)
    analytics_data = {
        "vocab": matrice.get_top_vocab_for_topic(args.top_n, EPSILON),
        "htid": matrice.get_top_work_for_topic(args.top_n, EPSILON),
        "author": matrice.get_top_author_for_topic(args.top_n, EPSILON)
    }
    
    with open(args.output, 'w') as f:
        json.dump(analytics_data, f, ensure_ascii=False, indent=4)