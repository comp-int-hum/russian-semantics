import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from steamroller import Environment


vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 5000),
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("HATHITRUST_ROOT", "", "${DATA_ROOT}/hathi_trust"),
    ("HATHITRUST_INDEX", "", "${HATHITRUST_ROOT}/hathi_index.tsv.gz"),
    ("HATHITRUST_MARC", "", "${HATHITRUST_ROOT}/hathi_marc.json.gz"),
    ("LANGUAGE", "", "rus"),
    ("START_YEAR", "", 1500),
    ("END_YEAR", "", 1950),
    ("WINDOW_SIZE", "", 50),
    ("USE_PREASSEMBLED_DATA", "", False),
)


env = Environment(
    variables=vars,
    ENV=os.environ,
    BUILDERS={
        "FilterHathiTrust" : Builder(
            action="python scripts/filter_hathitrust.py --hathitrust_marc ${HATHITRUST_MARC} --language ${LANGUAGE} --output ${TARGETS[0]}"
        ),
        "PopulateHathiTrust" : Builder(
            action="python scripts/populate_hathitrust.py --input ${SOURCES[0]} --output ${TARGETS[0]} --hathitrust_root ${HATHITRUST_ROOT}"
        ),        
    }
)

if env["USE_PREASSEMBLED_DATA"]:
    populated = env.File("work/full_russian_documents.jsonl.gz")
else:
    filtered = env.FilterHathiTrust(
        "work/russian_documents.jsonl.gz",
        []
    )
    populated = env.PopulateHathiTrust(
        "work/full_russian_documents.jsonl.gz",
        filtered
    )
