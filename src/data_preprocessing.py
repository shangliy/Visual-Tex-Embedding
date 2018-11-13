"""
Author: Shanglin
Description: This script is to preprocess data.
"""
import os
import json
import glob
import pickle as pkl

key_attrs = ['style', 'color', 'season','attrs', 'class']

def build_vocab():
    words_txts = glob.glob('dataset/labels/*.txt')
    vocabulary_set = set()
    for txt_f in words_txts:
        with open(txt_f) as f:
            words_list = [l.strip() for l in f.readlines()]
        for w in words_list: vocabulary_set.add(w)
    print('Total vocabulary_set size ', len(vocabulary_set))
    vocabulary_list = list(vocabulary_set)
    vocabulary_list.sort()
    return vocabulary_list

vocabulary_list = build_vocab()

with open('dataset/sku_info_all.json') as f:
    data = json.load(f)

data_list = []

max_len = 0

for skuid in data:
    # Image path
    image_path = 'dataset/images/%s.jpg'%(skuid)
    if not os.path.isfile(image_path):continue
    
    skuinfo = data[skuid]   
    sku_txt = skuinfo['title']
    attrs = [v  for key_ in key_attrs for v in skuinfo[key_] ]
    for v in attrs: 
        sku_txt += " %s"%(v)
    
    txt_word_list = ""
    for w in vocabulary_list:
        if w in sku_txt: txt_word_list += "%s "%(w)
    
    max_len = max(max_len, len(txt_word_list.split()))
    data_pair = (image_path, txt_word_list.strip())
    data_list.append(data_pair)

print('max_len: ', max_len)
pkl.dump(data_list,open('dataset/pair_train_data.p','wb'))
pkl.dump(vocabulary_list, open('dataset/vocabulary_list.p','wb'))
    

    
        
    