"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 13 Feb 2022
"""
import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

class Model:
    def __init__(self, config, bert_layer):
        # let tensorflow allocate memory whenever it is needed
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth=True
        self.sess = tf.compat.v1.Session(config=tf_config)

        print("TF Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print("Hub version: ", hub.__version__)
        print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
        
        # NN config
        self.config = config
        self.bert_layer = bert_layer

    # a function that creates model
    def create_model(self):
        input_word_ids = tf.keras.layers.Input(shape=(self.config['max_seq_length'],), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.config['max_seq_length'],), dtype=tf.int32, name="input_mask")
        input_type_ids = tf.keras.layers.Input(shape=(self.config['max_seq_length'],), dtype=tf.int32, name="input_type_ids")

        # only use pooled-output for classification 
        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, input_type_ids])
        
        regularizer = tf.keras.layers.ActivityRegularization(l1=self.config['regularization_penalty'])(pooled_output)
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(regularizer)

        # inputs coming from the function
        model = tf.keras.Model(
            inputs={
                'input_word_ids': input_word_ids,
                'input_mask': input_mask,
                'input_type_ids': input_type_ids}, 
            outputs=output)

        return model

    @staticmethod
    def create_graphs(history):
        train_accuracy = history.history['binary_accuracy']
        val_accuracy = history.history['val_binary_accuracy'],
        train_losses = history.history['loss'],
        val_losses = history.history['val_loss']
        f1 = plt.figure(0)
        plt.plot(train_accuracy)
        plt.plot(val_accuracy)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves', fontsize=16)
        plt.savefig('Accuracy.png')
        plt.show()
        
        f2 = plt.figure(1)
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.savefig('Loss.png')
        plt.show()
        
