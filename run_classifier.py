"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 13 Feb 2022
"""
from sre_parse import FLAGS
import tensorflow as tf
import os

from model import Model
from data_processor import DataProcessor
from tokenizer import Tokenizer
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

import json
import pandas as pd

class Run:
    def __init__(self):
        # enable xla devices
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

        self.DATA_DIR = "./magpie-corpus-master"

        self.config = { 'model_name': 'bert_idiom_finetuned',
                        'label_list' : [0, 1], 
                        'max_seq_length' : 500, # maximum length of (token) input sequences
                        'learning_rate': 3e-5,
                        'epochs': 3,
                        'optimizer': 'adam',
                        'regularization_penalty': 7e-4,
                        'train_batch': 2,
                        'dev_batch': 8,
                        'test_batch': 8,
                }
        
        self.handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'

    def run(self):
        # create tokenizer for data reading and creating bert layer
        tokenizer = Tokenizer(self.handle)

        # read data
        data_porcessor = DataProcessor(self.DATA_DIR, self.config, tokenizer.get_tokenizer())

        # taming the data
        with tf.device('/gpu:0'):
            train_data, dev_data, test_data = data_porcessor.get_data()

        # create model
        bert_layer = tokenizer.get_bert_layer()
        classification_model = Model(self.config, bert_layer)

        model = classification_model.create_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.PrecisionAtRecall(0.5),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall()])

        model.summary()

        # callbacks
        checkpoint = ModelCheckpoint(os.path.dirname(os.path.abspath(__file__)), monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=2),
        callbacks_list = [checkpoint, early_stopping]

        # train model
        history = model.fit(train_data,
                            validation_data=dev_data,
                            epochs=self.config['epochs'],
                            verbose=1,
                            callbacks = callbacks_list)

        # model evaluation on test set
        evaluation_results = model.evaluate(test_data, return_dict=True)

        # create resluts folder
        results_folder_path = './results'
        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)

        # draw graph
        Model.create_graphs(history, results_folder_path)

        # write result
        with open(os.path.join(results_folder_path,'results.txt'), 'w') as file:
            file.write(json.dumps(evaluation_results))

        # save model
        model.save(f"{self.config['model_name']}.h5")
        

if __name__ == "__main__":
    Run().run()