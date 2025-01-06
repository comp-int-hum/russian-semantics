from demarc import Record
import random
import logging
import gzip
import os.path
import json
import zipfile
import re
import argparse
from pairtree import PairtreeStorageFactory
import fasttext
from huggingface_hub import hf_hub_download

logger = logging.getLogger("populate_hathitrust")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hathitrust_root", dest="hathitrust_root", help="HathiTrust root directory")
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)
    
    psf = PairtreeStorageFactory()
    
    seen = set()
    with gzip.open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            rec = Record(j)
            val = {
                "title" : rec.title,
                "htid" : rec.htid
            }
            if rec.author:
                val["author"] = rec.author
            if rec.date:
                val["year"] = rec.date
            if rec.literary_form:
                val["form"] = rec.literary_form

            print(i, val["title"])
            id_toks = rec.htid.split(".")
            subcollection = id_toks[0]
            pairtree_name = id_toks[0].replace('/', '.')
            pairtree_path = ".".join(id_toks[1:]).replace('/', '.')
            mid = os.path.join(pairtree_name, pairtree_path)
            ident = ".".join(id_toks[1:])
            try:
                store = psf.get_store(
                    store_dir=os.path.join(
                        args.hathitrust_root,
                        subcollection
                    )
                )
                obj = store.get_object(ident, create_if_doesnt_exist=False)
            except Exception as e:
                logger.error(
                    "Could not access HathiTrust document '%s'",
                    rec.htid
                )
                raise e
            full_content = []
            for subpath in obj.list_parts():
                for fname in obj.list_parts(subpath):
                    if fname.endswith("zip"):
                        with zipfile.ZipFile(
                                obj.get_bytestream(
                                    "{}/{}".format(subpath, fname),
                                    streamable=True
                                )
                        ) as izf:                            
                            for page in sorted(izf.namelist()):
                                if page.endswith("txt"):
                                    txt = izf.read(page).decode("utf-8")
                                    #if correct_line_breaks:
                                    #    txt = re.sub(r"\-\s*?\n\s*", "", txt)
                                    full_content.append(txt.replace("\n",  " "))
                                    

            labels, probs = model.predict([p.replace("\n", " ") for p in full_content])
            pages = []
            for page, label in zip(full_content, labels):
                if "Cyrl" in label[0]:
                    pages.append(page)
            val["content"] = "\n".join(pages)
            ofd.write(json.dumps(val) + "\n")
