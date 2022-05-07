source /home/acb19lh/diss21/ADAPET-master/bin/setup.sh
python $ADAPET_ROOT/cli.py \
--data_dir $ADAPET_ROOT/magpie-corpus-master/2500/ \
--pattern 'The following sentence is [LBL]. [TEXT1]' \
--dict_verbalizer '{"l": "literal", "i": "phrase"}' \
--eval_every 2 \
--max_text_length 512 \
--batch_size 2 \
--eval_batch_size 4 \
--grad_accumulation_factor 8 \
--num_batches 469