import os, os.path, logging, random, subprocess, shlex, gzip, re, functools, time, imp, sys, json
from steamroller import Environment, Variables, Builder


print(" ----- start defining variables ----- ")

vars: Variables = Variables("custom.py")
vars.AddVariables(
    # Running status
    ("USE_PREASSEMBLED_DATA", "", False),
    ("EXISTING_DOC_DIR", "", False),
    # debug vocab bool
    ("DEBUG_EMBEDDING_VOCAB", "", False),
    # debug vocab params
    ("VOCAB_COUNTER_NUM_DOCS", "", 5),
    # embedding bool
    ("USE_PRETRAINED_EMBEDDING", "", False),
    # embedding params
    ("NUMBERS_OF_DOC", "", 20000),
    # ("GET_EMBEDDING_STATS", False),
    # model training
    ("NUMBERS_OF_TOPICS", "", [5, 10, 15, 20, 30]),
    ("DATA_ROOT", "", "/home/zxia15/data_zxia15/russian-semantics/work"),
    ("DOC_DIR", "", "${DATA_ROOT}/final_cleaned_russian_documents.jsonl.gz"),
    ("CHUNK_SIZE", "", [500]),
    ("USE_GRID", "", 1),
    ("RANDOM_LENGTH", "", 2000),
    ("NUM_ID_SPLITS", "", 500),
    ("LIMIT_SPLITS", "", None),
    ("GPU_ACCOUNT", "", None),
    ("GPU_QUEUE", "", None),
    # ("GRID_TYPE","", "slurm"),
    # ("GRID_GPU_COUNT","", 1),
    # ("GRID_MEMORY", "", "64G"),
    # ("GRID_TIME", "", "24:00:00"),
    ("BIMODAL_ATTESTATION_LEVEL", "", 1),
    ("WINDOW_SIZE", "", [20]),
    # ("FOLDS", "", 1),
    # ("OUTPUT_WIDTH", "", 5000),
    # ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    # ("HATHITRUST_ROOT", "", "${DATA_ROOT}/hathi_trust"),
    # ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_index.tsv.gz"),
    # ("HATHITRUST_MARC", "", "${HATHITRUST_ROOT}/hathi_marc.json.gz"),
    # ("LANGUAGE", "", "rus"),
    # ("START_YEAR", "", 1500),
    # ("END_YEAR", "", 1950),
    # ("WINDOW_SIZE", "", 50),
)

print(" ----- start defining environment ----- ")

env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "FilterHathiTrust": Builder(
            action="python scripts/06_20_allocating_data/filter_hathitrust.py --hathitrust_marc ${HATHITRUST_MARC} --language ${LANGUAGE} --output ${TARGETS[0]}"
        ),
        "PopulateHathiTrust": Builder(
            action="python scripts/06_20_allocating_data/populate_hathitrust.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT}"
        ),
        "TrainEmbeddings": Builder(
            action="python scripts/08_08_create_embeddings/train_embeddings.py --input ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "GenerateEmbeddingStats": Builder(
            action="python scripts/08_08_create_embeddings/generate_embedding_stats.py --input ${SOURCES[0]}"
        ),
        "DebugEmbeddingVocab": Builder(
            action="python scripts/08_14_modified_create_embeddings_pipeline/check_vocab_counter.py --input ${SOURCES} --output ${TARGETS} --num_docs ${VOCAB_COUNTER_NUM_DOCS}"
        ),
        "TrainEmbeddingsAndStats": Builder(
            action="python scripts/08_14_modified_create_embeddings_pipeline/train_embeddings_and_generate_stats.py --input ${SOURCES} --model_output ${TARGETS[0]} --stats_output ${TARGETS[1]} --num_docs ${NUMBERS_OF_DOC}"
        ),
        "GetEmbeddingStats": Builder(
            action="python scripts/08_14_modified_create_embeddings_pipeline/train_embeddings_and_generate_stats.py --input ${SOURCES} --model_output ${TARGETS[0]} --stats_output ${TARGETS[1]} --num_docs ${NUMBERS_OF_DOC} --skip_model"
        ),
    },
)

print(" ----- start processing command ----- ")

# populated full russian document data from hathitrust
if not env["USE_PREASSEMBLED_DATA"]:
    print(" ----- start preassembing data ----- ")
    filtered = env.FilterHathiTrust("work/russian_documents.jsonl.gz", [])
    populated = env.PopulateHathiTrust("work/full_russian_documents.jsonl.gz", filtered)
    exit(0)

print(" ----- skipped preassembing data ----- ")

# after data filtering
# The basic pattern for invoking a build rule is:
#   "Rule(list_of_targets, list_of_sources, VARIABLE1=value, VARIABLE2=value...)"

assert env["EXISTING_DOC_DIR"]
jsonl_russian_doc_dir = env["DOC_DIR"]

if env["DEBUG_EMBEDDING_VOCAB"]:
    print(" ----- creating counters on debugging embedding vocabs ----- ")
    env.DebugEmbeddingVocab(
        "work/vocab_freq.csv",
        jsonl_russian_doc_dir,
    )
    exit(0)

if env["USE_PRETRAINED_EMBEDDING"]:
    print(" ----- skipped taining embedding, training embedding stats ----- ")
    env.GetEmbeddingStats(
        [
            f"work/embeddings/word_2_vec_embeddings_doc{env['NUMBERS_OF_DOC']}.bin",
            f"image/embedding_similarity_table_doc{env['NUMBERS_OF_DOC']}.png",
        ],
        jsonl_russian_doc_dir,
    )
else:
    print(" ----- taining embedding and stats ----- ")
    env.TrainEmbeddingsAndStats(
        [
            f"work/embeddings/word_2_vec_embeddings_doc{env['NUMBERS_OF_DOC']}.bin",
            f"image/embedding_similarity_table_doc{env['NUMBERS_OF_DOC']}.png",
        ],
        jsonl_russian_doc_dir,
    )

print(" ----- end of processing ----- ")
