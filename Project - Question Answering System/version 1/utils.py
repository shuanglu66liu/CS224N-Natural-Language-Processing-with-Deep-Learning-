# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:11:33 2017

@author: shuangluliu
"""
import numpy as np
def data_iterator(data, batch_size=32, shuffle=False):
  # Optionally shuffle the data before training
  question=data[0]
  context=data[1]
  answer=data[2]
  if shuffle:
    indices = np.random.permutation(len(question))
    question_X = question[indices]
    context_X=context[indices]
    answer_X=answer[indices]
  else:
    question_X = question
    context_X = context
    answer_X=answer
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(question) / float(batch_size)))
  for step in range(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    q = question_X[batch_start:batch_start + batch_size]
    c=context_X[batch_start:batch_start + batch_size]
    a=answer_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    
    ###
    yield q,c,a
    total_processed_examples += len(q)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(question), 'Expected {} and processed {}'.format(len(question), total_processed_examples)
  
  
 
