import os.path
from steamroller import Environment
from SCons.Environment import OverrideEnvironment

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("RANDOM_SEED", "", 0),
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("MAX_SUBDOC_LENGTH", "", 100),
    ("MIN_WORD_COUNT", "", 10),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("TOP_NEIGHBORS", "", 15),
    ("TOP_TOPIC_WORDS", "", 5),
    ("TOPIC_COUNT", "", 50),
    ("MAX_EPOCHS", "", 200),
    ("CUDA_DEVICE", "", "cpu"),
    ("BATCH_SIZE", "", 512),
    ("LEARNING_RATE", "", 0.0001),
    ("OPTIMIZER", "", "adam"),
    ("LOWERCASE", "", True),
    (
        "TEST_PROPORTION",
        "",
        0.1
    ),
    (
        "VAL_PROPORTION",
        "This proportion of *non-test* data is used for validation (early stop etc)",
        0.1
    ),
    ("STUDIES", "", []),
    ("MODEL_TYPES", "", [])
)

env = Environment(
    variables=vars,
    BUILDERS={
        "SplitData" : Builder(
            action="python scripts/split_data.py --input ${SOURCES[0]} --first_output ${TARGETS[0]} --second_output ${TARGETS[1]} --second_proportion ${PROPORTION} --split_field ${SPLIT_FIELD} --random_seed ${RANDOM_SEED} --content_field ${CONTENT_FIELD} --down_sample ${DOWN_SAMPLE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} ${'--lowercase' if LOWERCASE else ''}"
        ),
        "TrainEmbeddings" : Builder(
            action="python scripts/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]} --content_field ${CONTENT_FIELD}"
        ),
        "TestForLR": Builder(
            action="python scripts/test_lr_for_model.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --max_epochs ${MAX_EPOCHS} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_count ${MIN_WORD_COUNT} --max_word_proportion ${MAX_WORD_PROPORTION} --content_field ${CONTENT_FIELD} --time_field ${TIME_FIELD} --model_type ${MODEL_TYPE} --optimizer ${OPTIMIZER} --min_time ${MIN_TIME} --max_time ${MAX_TIME}"
        ),
        "TrainModel" : Builder(
            action="python scripts/train_model.py --train ${SOURCES[0]} ${'--val ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --embeddings ${SOURCES[1]} --window_size ${WINDOW_SIZE} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE} --max_epochs ${MAX_EPOCHS} --output ${TARGETS[0]} --num_topics ${TOPIC_COUNT} --learning_rate ${LEARNING_RATE} --min_word_count ${MIN_WORD_COUNT} --max_word_proportion ${MAX_WORD_PROPORTION} --content_field ${CONTENT_FIELD} --time_field ${TIME_FIELD} --model_type ${MODEL_TYPE} --optimizer ${OPTIMIZER} --min_time ${MIN_TIME} --max_time ${MAX_TIME}"
        ),
        "ApplyModel" : Builder(
           action="python scripts/apply_model.py --model ${SOURCES[0]} --input ${SOURCES[1]} --device ${CUDA_DEVICE} --time_field ${TIME_FIELD} --content_field ${CONTENT_FIELD} --output ${TARGETS[0]}"
        ),
        "GenerateWordSimilarityTable" : Builder(
            action="python scripts/generate_word_similarity_table.py --embeddings ${SOURCES[0]} --output ${TARGETS[0]} --target_words ${WORD_SIMILARITY_TARGETS} --top_neighbors ${TOP_NEIGHBORS} ${'--language_code ' + LANGUAGE_CODE if LANGUAGE_CODE else ''}"
        ),
        "GetMatrice": Builder(
            action="python scripts/matrice.py --model ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]} --time_field ${TIME_FIELD} --content_field ${CONTENT_FIELD} --device ${CUDA_DEVICE} --batch_size ${BATCH_SIZE}"
        ),
        "GetTopTopicData": Builder(
            action="python scripts/top_dist_info.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        )
    }
)

for study in env["STUDIES"]:

    env = OverrideEnvironment(env, study)
    
    data = "${DATA_ROOT}/${NAME}.jsonl.gz"
    
    # train_val_data, test_data = env.SplitData(
    #     [
    #         "work/${NAME}/train_val_data.jsonl.gz",
    #         "work/${NAME}/test_data.jsonl.gz"
    #     ],
    #     data,
    #     PROPORTION=env["TEST_PROPORTION"]
    # )

    # embeddings = env.TrainEmbeddings(
    #     "work/${NAME}/embeddings.bin.gz",
    #     train_val_data
    # )

    train_val_data = "work/${NAME}/train_val_data.jsonl.gz"
    test_data = "work/${NAME}/test_data.jsonl.gz"
    embeddings = "work/${NAME}/embeddings.bin.gz"

    # word_similarity_table = env.GenerateWordSimilarityTable(
    #    "work/${NAME}/word_similarity.tex",        
    #    embeddings,       
    #    TOP_NEIGHBORS=5,
    # )

    for model_type in env["MODEL_TYPES"]:
        # env.TestForLR(['output.txt'],
        #     [
        #         train_val_data,
        #         embeddings
        #     ],
        #     MODEL_TYPE=model_type)
        # topic_model = env.TrainModel(
        #     "work/${NAME}/${MODEL_TYPE}_model.bin.gz",
        #     [
        #         train_val_data,
        #         embeddings
        #     ],
        #     MODEL_TYPE=model_type
        # )

        topic_model = f"work/{env['NAME']}/{model_type}_model.bin.gz"

        # perplexity = env.ApplyModel(
        #     "work/${NAME}/${MODEL_TYPE}_perplexity.jsonl.gz",
        #     [topic_model, test_data if test_data else train_val_data],
        #     MODEL_TYPE=model_type
        # )

        matrice = env.GetMatrice(
            "work/${NAME}/${MODEL_TYPE}_all_data_by_prob_matrice.pkl.gz",
            [topic_model, data],
            MODEL_TYPE=model_type
        )

        # output = "output.json"

        # output_json = env.GetTopTopicData(
        #     [output],
         #    [matrice]
        # )