#!/bin/sh
python3 /home/acb19lh/diss21/pet-master/cli.py --method pet --task_name idiom-detection --pattern_ids 0 --data_dir /home/acb19lh/diss21/pet-master/magpie-corpus-master --model_type bert --model_name_or_path bert-base-uncased --output_dir /data/acb19lh/results/pet-bert/pet-bert-2500-p0 --do_train --do_eval --train_examples 2500 --unlabeled_examples 10000 --split_examples_evenly --pet_per_gpu_train_batch_size 2 --pet_per_gpu_unlabeled_batch_size 4 --pet_gradient_accumulation_steps 8 --pet_max_steps 250 --lm_training --pet_max_seq_lengt 512 --pet_repetitions 1 --sc_per_gpu_train_batch_size 2 --sc_per_gpu_unlabeled_batch_size 4 --sc_gradient_accumulation_steps 8 --sc_max_steps 5000 --sc_max_seq_length 512