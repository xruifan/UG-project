"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 13 Feb 2022
"""
import os
import pandas as pd
import json
import tensorflow as tf
from tensorflow import data
from official.nlp.data import classifier_data_lib


class DataProcessor:
    def __init__(self, data_dir, config, tokenizer):
        # set the names of the files containing the examples
        self.data_dir = data_dir
        self.TRAIN_FILE_NAME = "train.jsonl"
        self.DEV_FILE_NAME = "dev.jsonl"
        self.TEST_FILE_NAME = "test.jsonl"
        self.config = config
        self.tokenizer = tokenizer

    # a function that reads data
    def create_examples(self, path):
        data = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label']
                # literal = 0, idiomatic = 1
                if (label == 'l'):
                    label = 0
                else:
                    label = 1
                id = example_json['no.']
                text = example_json['context']

                data.append([id,text,label])

        return pd.DataFrame(data=data, columns=['id','text','label'])

    # a function that converts row to input features and label
    def create_feature(self, text, label, label_list=None, max_seq_length=None, tokenizer=None):
        if label_list is None:
            label_list = self.config['label_list']
        if max_seq_length is None:
            max_seq_length = self.config['max_seq_length']
        if tokenizer is None:
            tokenizer = self.tokenizer
        # construct a InputExample as an example
        example = classifier_data_lib.InputExample(guid = None,
                                                text_a = text.numpy(), 
                                                text_b = None, 
                                                label = label.numpy())
        # convert the example with label list and tokenizer 
        feature = classifier_data_lib.convert_single_example(0, example, label_list, max_seq_length, tokenizer)

        return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

    # a functino that converts the features to the format that tensorflow uses (dataframe)
    def create_feature_map(self, text, label):

        input_ids, input_mask, segment_ids, label_id = tf.py_function(self.create_feature, inp=[text, label], Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

        max_seq_length = self.config['max_seq_length']

        # define shapes of inputs
        input_ids.set_shape([max_seq_length])
        input_mask.set_shape([max_seq_length])
        segment_ids.set_shape([max_seq_length])
        label_id.set_shape([])

        x = {
            'input_word_ids': input_ids,
            'input_mask': input_mask,
            'input_type_ids': segment_ids
        }
        return (x, label_id)

    # apply the transformation to batch dataset
    def trans_data(self, input_data, batch, shuffle):
        if (shuffle):
            input_data = (input_data.map(self.create_feature_map,
                                        num_parallel_calls=data.experimental.AUTOTUNE)
                                    .shuffle(10000)
                                    .batch(batch, drop_remainder=False)
                                    .prefetch(data.experimental.AUTOTUNE))
        else: 
            input_data = (input_data.map(self.create_feature_map,
                                        num_parallel_calls=data.experimental.AUTOTUNE)
                                    .batch(batch, drop_remainder=False)
                                    .prefetch(data.experimental.AUTOTUNE))
        return input_data

    def get_data(self):
        # read the data
        train_df = self.create_examples(os.path.join(self.data_dir, self.TRAIN_FILE_NAME))
        dev_df = self.create_examples(os.path.join(self.data_dir, self.DEV_FILE_NAME))
        test_df = self.create_examples(os.path.join(self.data_dir, self.TEST_FILE_NAME))

        train_data = data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))
        dev_data = data.Dataset.from_tensor_slices((dev_df['text'].values, dev_df['label'].values))
        test_data = data.Dataset.from_tensor_slices((test_df['text'].values, test_df['label'].values))
        
        return self.trans_data(train_data, self.config['train_batch'], True), self.trans_data(dev_data, self.config['dev_batch'], True), self.trans_data(test_data, self.config['test_batch'], False)
