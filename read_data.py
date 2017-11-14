# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:24:11 2017

@author: shuangluliu
"""
data_path=pjoin(FLAGS.data_dir,"train.ids.question")
print (data_path)
tf.gfile.Exists(data_path)
rev_vocab = []
data=[]
with tf.gfile.GFile(data_path, mode="rb") as f:
        rev_vocab.extend(f.readlines())
        rev_vocab = [line.decode("utf-8").strip('\n') for line in rev_vocab] 
        for q in rev_vocab:
            data.append([int(item) for item in q.split(" ")])
        
        
def initialize_dataset(data_path):
    if tf.gfile.Exists(data_path):
        data_str = []
        data_ids=[]
        with tf.gfile.GFile(data_path, mode="rb") as f:
            data_str.extend(f.readlines())
            data_str = [line.decode("utf-8").strip('\n') for line in data_str]
            for i in data_str:
                data_ids.append([int(j) for j in i.split(" ")])
        
        return data_ids
    else:
        raise ValueError("Vocabulary file %s not found.", data_path)
        
with tf.gfile.GFile(embed_path, mode="rb") as f:
            data_str.extend(f.readlines())        