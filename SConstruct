import os, os.path
from steamroller import Environment, Variables, Builder


vars: Variables = Variables("custom.py")
vars.AddVariables(
    # data
    ("DATA_ROOT", "", "/home/zxia15/russian-semantics/work"),
    ("DOC_DIR", "", "${DATA_ROOT}/dataset/metadata_with_full_text.jsonl.gz"),
    ("EMBEDDING_DIR", "", "${DATA_ROOT}/embeddings/word_2_vec_embeddings.bin"),
    # Running status
    ("USE_PREASSEMBLED_DATA", "", False),
    ("USE_PRETRAINED_EMBEDDING", "", False),
    ("USE_PREASSEMBLED_STATS", "", False),
    ("USE_PREEXISTING_DETM", "", False),
    ("DEBUG_LDA", "", False),
    ("COMPARE_MODELS", "", False),
    # embedding & debug vocab params
    ("DEBUG_EMBEDDING_VOCAB", "", False),
    ("USE_PART_OF_DOCS", "", False),
    ("VOCAB_COUNTER_NUM_DOCS", "", 5),
    ("NUMBERS_OF_DOC", "", -1),
    ("EMBEDDING_SIZE", "", 100),
    # model detm training params
    ("CONTENT_FIELD", "", "text"),
    ("TIME_FIELD", "", "written_year"),
    ("USE_SBATCH", "", False),
    ("NUM_TOP_WORDS", "", 8),
    ("MIN_WORD_OCCURRENCE", "", 100),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("MIN_TIME", "", 1850),
    ("MAX_TIME", "", 1910),
    ("NUMBER_OF_TOPICS", "", 10),
    ("WINDOW_SIZE", "", 10),
    ("MAX_SUBDOC_LENGTH", "", 500),
    ("CHUNK_SIZE", "", [500]),
    ("USE_GRID", "", 1),
    ("RANDOM_LENGTH", "", 2000),
    ("NUM_ID_SPLITS", "", 500),
    ("LIMIT_SPLITS", "", None),
    ("GPU_ACCOUNT", "", "tlippin1_gpu"),
    ("GPU_QUEUE", "", "a100"),
    ("BATCH_SIZE", "", 64),  # 2048?
    ("EPOCHS", "", 200),
    ("LEARNING_RATE", "", 0.015),
    ("LR_IDENTIFIER", "", "0015"),
    ("BIMODAL_ATTESTATION_LEVEL", "", 1),
    ("RANDOM_SEED", "", 42),
    ("BATCH_PREPROCESS", "", True),
    ("BATCH_PREPROCESS_COMMAND", "", "--batch_preprocess"),
    # create figures params
    ("FIGURE_TYPE", "", "topic_top_titles"),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "FilterHathiTrust": Builder(
            action="python scripts/000_filter_hathitrust.py --hathitrust_marc ${HATHITRUST_MARC} --language ${LANGUAGE} --output ${TARGETS[0]}"
        ),
        "PopulateHathiTrust": Builder(
            action="python scripts/001_populate_hathitrust.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT}"
        ),
        "TrainEmbeddings": Builder(
            action="python scripts/002_train_embeddings.py --input ${SOURCES} --output ${TARGETS} --num_docs ${NUMBERS_OF_DOC} --embedding_size ${EMBEDDING_SIZE}"
        ),
        "GetEmbeddingStats": Builder(
            action="python scripts/003_generate_stats.py --input ${SOURCES} --output ${TARGETS}"
        ),
        "BuildDETMSlurm": Builder(
            action=(
                "python scripts/004v2_create_detm_sbatch_script.py "
                + " --sbatch_script train_detm.sh --logger train_detm.out "
                + "--command 'python scripts/train_detm.py --embeddings ${SOURCES[0]} --train ${SOURCES[1]}  "
                + "--output ${TARGETS[0]} --log ${TARGETS[1]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} "
                + "--max_word_proportion ${MAX_WORD_PROPORTION} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --epochs ${EPOCHS} "
                + "--emb_size ${EMBEDDING_SIZE} --rho_size ${EMBEDDING_SIZE} --learning_rate ${LEARNING_RATE} --random_seed ${RANDOM_SEED} "
                + "--min_time ${MIN_TIME} --max_time ${MAX_TIME} ${BATCH_PREPROCESS_COMMAND}' "
                + "--job_name detm_model --use_gpu --account ${GPU_ACCOUNT} --partition ${GPU_QUEUE} "
                + "--gres gpu:1 --time 24:00:00 --mem_alloc 180G"
            )
        ),
        "TrainDETM": Builder(
            action=(
                "python scripts/004_train_detm.py --embeddings ${SOURCES[0]} --train ${SOURCES[1]}  "
                + "--output ${TARGETS[0]} --log ${TARGETS[1]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} "
                + "--min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION} "
                + "--window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --epochs ${EPOCHS} "
                + "--emb_size ${EMBEDDING_SIZE} --rho_size ${EMBEDDING_SIZE} --learning_rate ${LEARNING_RATE} "
                + "--random_seed ${RANDOM_SEED} --min_time ${MIN_TIME} --max_time ${MAX_TIME} ${BATCH_PREPROCESS_COMMAND} "
                + "--content_field ${CONTENT_FIELD} --time_field ${TIME_FIELD}"
            )
        ),
        "GetMatriceAnalysis": Builder(
            action="python scripts/005_get_analysis_results.py --input ${SOURCES[0]} --model ${SOURCES[1]} --output ${TARGETS[0]} --log ${TARGETS[1]} --min_time ${MIN_TIME} --max_time ${MAX_TIME} " +
            "--window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --max_word_proportion ${MAX_WORD_PROPORTION} --min_word_occurrence ${MIN_WORD_OCCURRENCE} " + 
            "--content_field ${CONTENT_FIELD} --time_field ${TIME_FIELD} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE}"
        )
    },
)


# populated full russian document data from hathitrust
if not env["USE_PREASSEMBLED_DATA"]:
    filtered = env.FilterHathiTrust("work/russian_documents.jsonl.gz", [])
    populated = env.PopulateHathiTrust("work/full_russian_documents.jsonl.gz", filtered)

if env["USE_PREASSEMBLED_DATA"] and not env["USE_PREASSEMBLED_STATS"]:
    
    jsonl_russian_doc_dir = env["DOC_DIR"]

    embedding_dir = (
        f"work/embeddings/word_2_vec_embeddings_doc{env['NUMBERS_OF_DOC']}.bin"
        if env["USE_PART_OF_DOCS"] and type(env["NUMBERS_OF_DOC"]) == int
        else env["EMBEDDING_DIR"]
    )

    stats_dir = (
        f"images/embedding_similarity_table_doc{env['NUMBERS_OF_DOC']}.png"
        if env["USE_PART_OF_DOCS"] and type(env["NUMBERS_OF_DOC"]) == int
        else "images/embedding_stemmed_similarity_table.png"
    )

    if not env["USE_PRETRAINED_EMBEDDING"]:
        env.TrainEmbeddings([embedding_dir], jsonl_russian_doc_dir)

    else:
        env.GetEmbeddingStats([stats_dir], embedding_dir)


if env["USE_PREASSEMBLED_STATS"] and not env["USE_PREEXISTING_DETM"]:

    embeddings_dir = env["EMBEDDING_DIR"]
    jsonl_russian_doc_dir = env["DOC_DIR"]

    model_file = f"work/detm_model_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}.bin"
    model_log = (f"train_detm_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}.out")
    slurm_file = f"train_detm.sh"
    if not env["USE_SBATCH"]:
        env.TrainDETM(
            [model_file, model_log],
            [embeddings_dir, jsonl_russian_doc_dir],
            LR_IDX=env["LR_IDENTIFIER"],
            SLURM_FILE=slurm_file,
            LEARNING_RATE=env["LEARNING_RATE"],
        )

    else:
        env.BuildDETMSlurm(
            [model_file, model_log],
            [embeddings_dir, jsonl_russian_doc_dir],
            LEARNING_RATE=env["LEARNING_RATE"],
            STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
            STEAMROLLER_GPU_COUNT=1,
            STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
            STEAMROLLER_MEMORY="180G",
            )

if env["USE_PREEXISTING_DETM"]:
        jsonl_russian_doc_dir = env["DOC_DIR"]
        model_file = f"work/detm_model_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}.bin"
        output_json = f"work/matrice_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}.json"
        output_log = f"work/matrice_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}.out"
        env.GetMatriceAnalysis([output_json, output_log], [jsonl_russian_doc_dir, model_file])
 