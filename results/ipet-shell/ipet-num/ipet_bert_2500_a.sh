#!/bin/sh
python /home/acb19lh/diss21/pet-master/cli.py --method ipet --task_name idiom-detection --pattern_ids 0 1 2 3 --data_dir /home/acb19lh/diss21/pet-master/magpie-corpus-master --model_type bert --model_name_or_path bert-base-uncased --output_dir /data/acb19lh/results/ipet-bert/ipet-bert-2500-a --do_train --do_eval --train_examples 2500 --unlabeled_examples 10000 --split_examples_evenly --pet_per_gpu_train_batch_size 2 --pet_per_gpu_unlabeled_batch_size 4 --pet_gradient_accumulation_steps 8 --lm_training --ipet_generations 5 --pet_max_seq_lengt 512 --pet_repetitions 1 --sc_per_gpu_train_batch_size 2 --sc_per_gpu_unlabeled_batch_size 4 --sc_gradient_accumulation_steps 8 --sc_max_steps 5000 --sc_max_seq_length 512 --ipet_logits_percentage 0.25 --ipet_scale_factor 4


