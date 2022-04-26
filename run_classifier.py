"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 13 Feb 2022
"""
import tensorflow as tf
import tensorflow_addons as tfa
import os

from model import Model
from data_processor import DataProcessor
from tokenizer import Tokenizer
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

import json


class Run:
    def __init__(self):
        # enable xla devices
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        
        # need to change DATA_DIR and config for different tasks
        self.config = { 'model_name': 'bert_idiom_finetuned',
                        'train_num' : 19280,
                        'label_list' : [0, 1], 
                        'max_seq_length' : 500, # maximum length of (token) input sequences
                        'learning_rate': 1e-5,
                        'epsilon': 1e-8,
                        'epochs': 3,
                        'optimizer': 'adam',
                        'train_batch': 2,
                        'dev_batch': 4,
                        'test_batch': 2,
                }

        self.DATA_DIR = "./magpie-corpus-master/{}".format(self.config['train_num'])
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

        # define metrics
        def f1_macro(y_true, y_pred):
            def recall(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            return 2*((p*r)/(p+r+K.epsilon()))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'],epsilon=self.config['epsilon']),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics = [f1_macro, 'acc'])

        model.summary()

        # callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=2),
        callbacks_list = [early_stopping]

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
        #Model.create_graphs(history, results_folder_path)

        # write result
        with open(os.path.join(results_folder_path,f"result_{self.config['train_num']}.txt"), 'w') as file:
            file.write(json.dumps(evaluation_results))

        # save model
        model.save(f"{self.config['model_name']}_{self.config['train_num']}.h5")
        

if __name__ == "__main__":
    Run().run()