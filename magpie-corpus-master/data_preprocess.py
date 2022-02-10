import jsonlines
import os
import matplotlib.pyplot as plt
import nltk

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
            if (len(nltk.word_tokenize(obj["context"]))) > 300: continue

            # assign number to idiomatic and literal lines
            num = 0
            if obj["label"] == 'i':
                obj["no."] = num
                idiomatic.append(obj)
                num+= 1
            elif obj["label"] == 'l':
                obj["no."] = num
                literal.append(obj)
                num+= 1
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

    train = train_i + train_l
    dev = dev_i + dev_l
    test = test_i + test_l
    
    
    print("train size:%i dev size: %i test size: %i" % (len(train), len(dev), len(test)))
    
    # write jsonl files
    with jsonlines.open(os.path.join(dir_path,"train.jsonl"), mode="w") as w:
        w.write_all(train)

    with jsonlines.open(os.path.join(dir_path,"dev.jsonl"), mode="w") as w:
        w.write_all(dev)
    
    with jsonlines.open(os.path.join(dir_path,"test.jsonl"), mode="w") as w:
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
    height = [len(train), len(dev), len(test)]
    label = ['train', 'dev', 'test']
    plt.bar(left, height, tick_label=label, width=0.8)
    plt.ylabel('data size')
    plt.title('Ratio of split')
    plt.savefig(os.path.join(dir_path,'ratio_of_split.png'))
    plt.show()
    


if __name__ == '__main__':
    main()