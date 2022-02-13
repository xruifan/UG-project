"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 13 Feb 2022
"""
import tensorflow_hub as hub
from official.nlp.bert import tokenization

class Tokenizer:
    def __init__(self, handle):
        self.handle = handle

    def get_bert_layer(self):
        # BERT layer and tokenizer
        return hub.KerasLayer(self.handle, trainable=True)
        
    def get_tokenizer(self):
        vocab_file = self.get_bert_layer().resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.get_bert_layer().resolved_object.do_lower_case.numpy() # checks if the bert layer is uncased or not
        return tokenization.FullTokenizer(vocab_file, do_lower_case)