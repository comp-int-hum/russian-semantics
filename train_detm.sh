#!/bin/sh
#SBATCH --job-name=detm_model
#SBATCH -A tlippin1_gpu
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --output=train_detm.out
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

python scripts/train_detm.py --embeddings "/home/zxia15/data_zxia15/russian-semantics/work/embeddings/7500_stemmed_data/word_2_vec_embeddings.bin" --train "/home/zxia15/data_zxia15/russian-semantics/work/stemmed_russian_documents_inuse.jsonl.gz" --output "work/detm_model_20_500_10_0015_150.bin" --log "traim_detm_20_500_10_0015_150.out" --num_topics 20 --batch_size 512 --min_word_occurrence 100 --max_word_proportion 0.7 --window_size 10 --max_subdoc_length 500 --epochs 150 --emb_size 100 --rho_size 100 --learning_rate 0.015 --random_seed 42 --min_time 1850 --max_time 1900