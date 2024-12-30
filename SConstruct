import os, os.path
from steamroller import Environment, Variables, Builder


vars: Variables = Variables("custom.py")
vars.AddVariables(
    # data
    ("DATA_ROOT", "", "/home/zxia15/russian-semantics/work"),
    ("DOC_DIR", "", "${DATA_ROOT}/12_24_stemmed_metadata_with_full_text.jsonl.gz"),
    (
        "EMBEDDING_DIR",
        "",
        "${DATA_ROOT}/stemmed_embeddings/word_2_vec_embeddings.bin",
    ),
    # Running status
    ("USE_PREASSEMBLED_DATA", "", False),
    ("USE_PRETRAINED_EMBEDDING", "", False),
    ("USE_PREASSEMBLED_STATS", "", False),
    ("USE_PREEXISTING_DETM", "", False),
    ("DEBUG_LDA", "", False),
    # embedding & debug vocab params
    ("DEBUG_EMBEDDING_VOCAB", "", False),
    ("USE_PART_OF_DOCS", "", False),
    ("VOCAB_COUNTER_NUM_DOCS", "", 5),
    ("NUMBERS_OF_DOC", "", -1),
    ("EMBEDDING_SIZE", "", 100),
    # model detm training params
    ("USE_SBATCH", "", False),
    ("NUM_TOP_WORDS", "", 8),
    ("MIN_WORD_OCCURRENCE", "", 100),
    ("MAX_WORD_PROPORTION", "", 0.7),
    ("MIN_TIME", "", 1775),
    ("MAX_TIME", "", 1800),
    ("NUMBER_OF_TOPICS", "", 20),
    ("WINDOW_SIZE", "", 5),
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
    ("LEARNING_RATES", "", [0.015]),
    # ("LEARNING_RATES", "", [0.1, 0.05, 0.015]),
    ("LR_IDENTIFIERS", "", ["0015"]),
    # ("LR_IDENTIFIERS", "", ["01", "005", "0015"]),
    ("BIMODAL_ATTESTATION_LEVEL", "", 1),
    ("RANDOM_SEED", "", 42),
    # create figures params
    ("FIGURE_TYPE", "", "topic_top_titles"),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "FilterHathiTrust": Builder(
            action="python scripts/filter_hathitrust.py --hathitrust_marc ${HATHITRUST_MARC} --language ${LANGUAGE} --output ${TARGETS[0]}"
        ),
        "PopulateHathiTrust": Builder(
            action="python scripts/populate_hathitrust.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT}"
        ),
        "DebugEmbeddingVocab": Builder(
            action="python scripts/check_vocab_counter.py --input ${SOURCES} --output ${TARGETS} --num_docs ${VOCAB_COUNTER_NUM_DOCS}"
        ),
        "TrainEmbeddingsAndStats": Builder(
            action="python scripts/train_embeddings_and_generate_stats.py --input ${SOURCES} --model_output ${TARGETS[0]} --stats_output ${TARGETS[1]} --num_docs ${NUMBERS_OF_DOC} --embedding_size ${EMBEDDING_SIZE}"
        ),
        "GetEmbeddingStats": Builder(
            action="python scripts/train_embeddings_and_generate_stats.py --input ${SOURCES} --model_output ${TARGETS[0]} --stats_output ${TARGETS[1]} --num_docs ${NUMBERS_OF_DOC} --embedding_size ${EMBEDDING_SIZE} --skip_model"
        ),
        "BuildDETMSlurm": Builder(
            action=(
                "python scripts/create_detm_sbatch_script.py "
                + " --sbatch_script train_detm.sh --logger train_detm.out "
                + "--command 'python scripts/train_detm.py --embeddings ${SOURCES[0]} --train ${SOURCES[1]}  "
                + "--output ${TARGETS[0]} --log ${TARGETS[1]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} --min_word_occurrence ${MIN_WORD_OCCURRENCE} "
                + "--max_word_proportion ${MAX_WORD_PROPORTION} --window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --epochs ${EPOCHS} "
                + "--emb_size ${EMBEDDING_SIZE} --rho_size ${EMBEDDING_SIZE} --learning_rate ${LEARNING_RATE} --random_seed ${RANDOM_SEED} "
                + "--min_time ${MIN_TIME} --max_time ${MAX_TIME}' "
                + "--job_name detm_model --use_gpu --account ${GPU_ACCOUNT} --partition ${GPU_QUEUE} "
                + "--gres gpu:1 --time 24:00:00 --mem_alloc 180G"
            )
        ),
        "TrainDETM": Builder(
            action=(
                "python scripts/train_detm.py --embeddings ${SOURCES[0]} --train ${SOURCES[1]}  "
                + "--output ${TARGETS[0]} --log ${TARGETS[1]} --num_topics ${NUMBER_OF_TOPICS} --batch_size ${BATCH_SIZE} "
                + "--min_word_occurrence ${MIN_WORD_OCCURRENCE} --max_word_proportion ${MAX_WORD_PROPORTION} "
                + "--window_size ${WINDOW_SIZE} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --epochs ${EPOCHS} "
                + "--emb_size ${EMBEDDING_SIZE} --rho_size ${EMBEDDING_SIZE} --learning_rate ${LEARNING_RATE} "
                + "--random_seed ${RANDOM_SEED} --min_time ${MIN_TIME} --max_time ${MAX_TIME}"
            )
        ),
        "ApplyDETM": Builder(
            action="python scripts/apply_detm.py --model ${SOURCES[0]} --input ${SOURCES[1]} --output ${TARGETS[0]}  --log ${TARGETS[1]} --max_subdoc_length ${MAX_SUBDOC_LENGTH} --min_time ${MIN_TIME} --max_time ${MAX_TIME}"
        ),
        "CreateMatrices": Builder(
            action="python scripts/create_matrices.py --topic_annotations ${SOURCES[0]} --log ${TARGETS[1]} --output ${TARGETS[0]} --window_size ${WINDOW_SIZE} --min_time ${MIN_TIME}"
        ),
        "CreateFigures": Builder(
            action="python scripts/create_figures.py --input ${SOURCES[0]} --latex ${TARGETS[0]} --output ${TARGETS[1]} --top_n ${NUM_TOP_WORDS} --log ${TARGETS[2]} --figure_type ${FIGURE_TYPE}"
        ),
        "TrainDebugLDAModel": Builder(
            action="python scripts/lda_verification_model.py --train ${SOURCES} --log ${TARGETS} --min_time ${MIN_TIME} --max_time ${MAX_TIME} "
            + "--min_word_occur ${MIN_WORD_OCCURRENCE} --max_doclen 5000 --random_seed ${RANDOM_SEED}"
        ),
    },
)


# populated full russian document data from hathitrust
if not env["USE_PREASSEMBLED_DATA"]:
    filtered = env.FilterHathiTrust("work/russian_documents.jsonl.gz", [])
    populated = env.PopulateHathiTrust("work/full_russian_documents.jsonl.gz", filtered)

# after data filtering
# The basic pattern for invoking a build rule is:
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"

if env["USE_PREASSEMBLED_DATA"] and not env["USE_PREASSEMBLED_STATS"]:
    jsonl_russian_doc_dir = env["DOC_DIR"]

    if env["DEBUG_EMBEDDING_VOCAB"]:
        env.DebugEmbeddingVocab("work/vocab_freq.csv", jsonl_russian_doc_dir)

    embedding_dir = (
        f"work/embeddings/word_2_vec_embeddings_doc{env['NUMBERS_OF_DOC']}.bin"
        if env["USE_PART_OF_DOCS"] and type(env["NUMBERS_OF_DOC"]) == int
        else env["EMBEDDING_DIR"]
    )
    stats_dir = (
        f"images/embedding_similarity_table_doc{env['NUMBERS_OF_DOC']}.png"
        if env["USE_PART_OF_DOCS"] and type(env["NUMBERS_OF_DOC"]) == int
        else "images/embedding_stemmed_no_punc_similarity_table.png"
    )

    if not env["USE_PRETRAINED_EMBEDDING"]:
        env.TrainEmbeddingsAndStats([embedding_dir, stats_dir], jsonl_russian_doc_dir)

    else:
        env.GetEmbeddingStats([embedding_dir, stats_dir], jsonl_russian_doc_dir)


if (
    env["USE_PREASSEMBLED_STATS"]
    and not env["USE_PREEXISTING_DETM"]
    and not env["DEBUG_LDA"]
):
    embeddings_dir = env["EMBEDDING_DIR"]
    jsonl_russian_doc_dir = env["DOC_DIR"]

    for idx in range(len(env["LEARNING_RATES"])):

        output_file = f"work/detm_model_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}_sublen_{env['MAX_SUBDOC_LENGTH']}_widsize_{env['WINDOW_SIZE']}_lr_{env['LR_IDENTIFIERS'][idx]}_epoch_{env['EPOCHS']}.bin"
        output_log = f"train_detm_{env['MIN_TIME']}-{env['MAX_TIME']}_topics_{env['NUMBER_OF_TOPICS']}_sublen_{env['MAX_SUBDOC_LENGTH']}_widsize_{env['WINDOW_SIZE']}_lr_{env['LR_IDENTIFIERS'][idx]}_epoch_{env['EPOCHS']}.out"
        slurm_file = f"train_detm_{env['LR_IDENTIFIERS'][idx]}.sh"
        if not env["USE_SBATCH"]:
            env.TrainDETM(
                [output_file, output_log],
                [embeddings_dir, jsonl_russian_doc_dir],
                LR_IDX=env["LR_IDENTIFIERS"][idx],
                SLURM_FILE=slurm_file,
                LEARNING_RATE=env["LEARNING_RATES"][idx],
            )

        else:
            env.BuildDETMSlurm(
                [output_file, output_log],
                [embeddings_dir, jsonl_russian_doc_dir],
                LEARNING_RATE=env["LEARNING_RATES"][idx],
                STEAMROLLER_ACCOUNT=env.get("GPU_ACCOUNT", None),
                STEAMROLLER_GPU_COUNT=1,
                STEAMROLLER_QUEUE=env.get("GPU_QUEUE", None),
                STEAMROLLER_MEMORY="180G",
            )

if env["USE_PREEXISTING_DETM"]:
    if not env["USE_SBATCH"]:
        jsonl_russian_doc_dir = env["DOC_DIR"]
        model_file = f"work/1850_1900_model/detm_model_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}_{env['LR_IDENTIFIERS'][0]}_{env['EPOCHS']}.bin"
        output_file = f"work/apply_model_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}_{env['LR_IDENTIFIERS'][0]}_{env['EPOCHS']}.bin"
        output_log = (
            f"apply_detm_{env['MIN_TIME']}_{env['MAX_TIME']}_Epoch_{env['EPOCHS']}.out"
        )
        slurm_file = f"apply_detm_{env['MIN_TIME']}_{env['MAX_TIME']}.sh"
        # env.ApplyDETM([output_file, output_log], [model_file, jsonl_russian_doc_dir])
        topic_annotations = output_file
        output_file = f"work/matrices_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}.pkl.gz"
        output_log = f"create_matrices_{env['MIN_TIME']}_{env['MAX_TIME']}_Epoch_{env['EPOCHS']}.out"
        # env.CreateMatrices([output_file, output_log], topic_annotations)
        matrices_input = output_file
        latex_output = f"work/tables_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}_{env['FIGURE_TYPE']}.tex"
        output_log = f"create_figures_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}_{env['FIGURE_TYPE']}.out"
        figure_output = f"work/temporal_image_{env['NUMBER_OF_TOPICS']}_{env['MAX_SUBDOC_LENGTH']}_{env['WINDOW_SIZE']}_{env['FIGURE_TYPE']}.png"

        env.CreateFigures([latex_output, figure_output, output_log], matrices_input)

if env["DEBUG_LDA"]:
    jsonl_russian_doc_dir = env["DOC_DIR"]
    output_log = (
        f"debug_lda_{env['MIN_TIME']}_{env['MAX_TIME']}_Epoch_{env['EPOCHS']}.out"
    )
    env.TrainDebugLDAModel(output_log, jsonl_russian_doc_dir)
