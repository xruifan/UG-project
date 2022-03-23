import jsonlines
import os
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from official.nlp.bert import tokenization
import random

def main():

    dir_path = '.'
    file_name = 'MAGPIE_filtered_split_typebased.jsonl'

    idiomatic = []
    literal = []
    train = []
    dev = []
    test = []

    # create jsonlines reader
    with jsonlines.open(os.path.join(dir_path, file_name)) as reader:
        # BERT layer and tokenizer
        hub_handle = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
        bert_layer = hub.KerasLayer(hub_handle, trainable=False)

        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() # checks if the bert layer is uncased or not
        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        
        for obj in reader:
            # remove ' "" '
            obj["context"] = ' '.join(obj["context"])
            # remove \n
            obj["context"] = obj["context"].replace('\\n','')
            # shift '
            obj["context"] = obj["context"].replace(' \'','\'')
            # shift .
            obj["context"] = obj["context"].replace(' .','. ')
            # shift ,
            obj["context"] = obj["context"].replace(' ,',', ')
            # shift ?
            obj["context"] = obj["context"].replace(' ?','? ')
            # shift -
            obj["context"] = obj["context"].replace(' - ','-')
            # shift ()
            obj["context"] = obj["context"].replace(' ( ',' (')
            obj["context"] = obj["context"].replace(')',') ').replace(' )',') ')
            # shift n't
            obj["context"] = obj["context"].replace(' n\'t','n\'t ')
            # remove double space
            obj["context"] = obj["context"].replace('  ',' ')

            
            # delete lines sequence length larger than bert can handle
            if (len(tokenizer.wordpiece_tokenizer.tokenize(obj["context"]))) >= 499: continue
            
            # remove unused keys
            obj.pop('confidence', None)
            obj.pop('document_id', None)
            obj.pop('genre', None)
            obj.pop('id', None)
            obj.pop('idiom', None)
            obj.pop('judgment_count', None)
            obj.pop('label_distribution', None)
            obj.pop('non_standard_usage_explanations', None)
            obj.pop('offsets', None)
            obj.pop('sentence_no', None)
            obj.pop('split', None)
            obj.pop('variant_type', None)

            # assign number to idiomatic and literal lines
            if obj["label"] == 'i':
                idiomatic.append({"TEXT1":obj["context"],"LBL":"i"})

            elif obj["label"] == 'l':
                literal.append({"TEXT1":obj["context"],"LBL":"l"})
            else: 
                print(obj["label"])

            

    if len(idiomatic) > len(literal):
        idiomatic = idiomatic[:len(literal)]  
    else:
        literal = literal[:len(idiomatic)]
            
        
    print("idiomatic: %i literal: %i" % (len(idiomatic),len(literal)))

    # split into train, dev, test by 80%, 10%, 10%
    train_i = idiomatic[:int(len(idiomatic)*0.8)] 
    dev_i = idiomatic[int(len(idiomatic)*0.8):int(len(idiomatic)*0.9)] 
    test_i = idiomatic[int(len(idiomatic)*0.9):] 

    train_l = literal[:int(len(literal)*0.8)] 
    dev_l = literal[int(len(literal)*0.8):int(len(literal)*0.9)] 
    test_l = literal[int(len(literal)*0.9):] 

    dev = dev_i + dev_l
    test = test_i + test_l

    # shuffle lists
    random.shuffle(dev)
    random.shuffle(test)
    
    print("train size:%i dev size: %i test size: %i" % (len(train_i + train_l), len(dev), len(test)))
    
    # prepare data for ratio
    ratio_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for ratio in ratio_list:
        curr_dir_path = '{p}%'.format(p=ratio)
        os.makedirs(curr_dir_path)
        # write jsonl files
        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"train.jsonl"), mode="w") as w:
            train = train_i[:int(len(train_i)*ratio/100)] + train_l[:int(len(train_l)*ratio/100)]
            random.shuffle(train)
            w.write_all(train)

        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"dev.jsonl"), mode="w") as w:
            w.write_all(dev)
        
        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"test.jsonl"), mode="w") as w:
            w.write_all(test)

    #prepare data for num
    num_list = [10, 50, 100, 250, 500, 1000]
    for num in num_list:
        curr_dir_path = '{n}'.format(n=num)
        os.makedirs(curr_dir_path)
        # write jsonl files
        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"train.jsonl"), mode="w") as w:
            train = train_i[:num//2] + train_l[:num//2]
            random.shuffle(train)
            w.write_all(train)

        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"dev.jsonl"), mode="w") as w:
            w.write_all(dev)
        
        with jsonlines.open(os.path.join(dir_path,curr_dir_path,"test.jsonl"), mode="w") as w:
            w.write_all(test)



    # ratio of idiom and literal figure
    activities = ['idiomatic', 'literal']
    slices = [len(idiomatic), len(literal)]
    plt.pie(slices, labels = activities, startangle=90, shadow = True,
        radius = 1.2, autopct = '%1.1f%%')
    plt.legend()
    plt.title('Ratio of classes')
    plt.savefig(os.path.join(dir_path,'ratio_of_classes.png'))
    plt.show()

    # ratio of train, dev and test figure 
    left = [1, 2, 3]
    height = [len(train_i + train_l), len(dev), len(test)]
    label = ['train', 'dev', 'test']
    plt.bar(left, height, tick_label=label, width=0.8)
    plt.ylabel('data size')
    plt.title('Ratio of split')
    plt.savefig(os.path.join(dir_path,'ratio_of_split.png'))
    plt.show()
    


if __name__ == '__main__':
    main()