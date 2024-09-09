#!/bin/sh
#SBATCH --job-name=train_detm
#SBATCH --time=24:00:00
#SBATCH --output=slurm_embed.out
#SBATCH --mem=180G

python scripts/train_embeddings_and_generate_stats.py --input /home/zxia15/data_zxia15/russian-semantics/work/stemmed_russian_documents_inuse.jsonl.gz --model_output /home/zxia15/data_zxia15/russian-semantics/work/embeddings/7500_stemmed_data/word_2_vec_embeddings.bin --stats_output images/embedding_stemmed_similarity_table.png --num_docs -1 --embedding_size 100
