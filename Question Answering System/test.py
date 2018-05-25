# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:07:17 2017

@author: shuangluliu
"""

def add_padding(q_batch,c_batch):
        #max length of sentence in question and context 
        q_max_length=max([len(i) for i in q_batch])
        c_max_length=max([len(i) for i in c_batch])
        
        def zero_paddings(sentence, max_length):
            mask = [True] * len(sentence)
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
                mask += [False] * pad_len
            else:
                padded_sentence = sentence[:max_length]
                mask = mask[:max_length]
            return padded_sentence, mask

        def padding_batch(data, max_len):
            padded_data = []
            padded_mask = []
            for sentence in data:
                d, m = zero_paddings(sentence, max_len)
                padded_data.append(d)
                padded_mask.append(m)
            return (padded_data, padded_mask)
       
        question, question_mask = padding_batch(q_batch, q_max_length)
        context, context_mask = padding_batch(c_batch, c_max_length)
        return  question, question_mask, context, context_mask     

data=dataset['train']
question=data[0]
context=data[1]
answer=data[2]

indices = np.random.permutation(len(question))
question_X = question[indices]
context_X=context[indices]
answer_X=answer[indices]


step=1
batch_size=10
    # Create the batch by selecting up to batch_size elements
batch_start = step * batch_size
q = question_X[batch_start:batch_start + batch_size]
c=context_X[batch_start:batch_start + batch_size]
a=answer_X[batch_start:batch_start + batch_size]

     #print (q.shape,c.shape)
question, question_mask, context, context_mask =add_padding([q,c,a][0],[q,c,a][1])

for i in range(len(c)):
    for w in c[i]:
        try:
           print (vocab[w])
        except KeyError:
            print (w)