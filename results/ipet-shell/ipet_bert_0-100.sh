#!/bin/sh
python3 /home/acb19lh/diss21/pet-master/cli.py --method ipet --task_name idiom-detection --pattern_ids 0 1 2 3 --data_dir /home/acb19lh/diss21/pet-master/magpie-corpus-master --model_type bert --model_name_or_path bert-base-uncased --output_dir /home/acb19lh/diss21/pet-master/results/ipet-bert/ipet-bert-0-100 --do_eval --train_examples 0 --split_examples_evenly --do_train --do_eval --pet_repetitions 3 --ipet_n_most_likely 100 --reduction mean --ipet_generations 5