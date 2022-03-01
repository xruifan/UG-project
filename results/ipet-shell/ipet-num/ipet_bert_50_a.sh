#!/bin/sh
python3 /home/acb19lh/diss21/pet-master/cli.py --method ipet --task_name idiom-detection --pattern_ids 0 1 2 3 --data_dir /home/acb19lh/diss21/pet-master/magpie-corpus-master --model_type bert --model_name_or_path bert-base-uncased --output_dir /data/acb19lh/results/ipet-bert/ipet-bert-50-a --do_train --do_eval --train_examples 50 --unlabeled_examples 19230 --split_examples_evenly --pet_per_gpu_train_batch_size 4 --pet_per_gpu_unlabeled_batch_size 8 --pet_gradient_accumulation_steps 2 --pet_max_steps 250 --lm_training --sc_per_gpu_train_batch_size 8 --sc_per_gpu_unlabeled_batch_size 8 --sc_gradient_accumulation_steps 2 --sc_max_steps 5000 --ipet_generations 5