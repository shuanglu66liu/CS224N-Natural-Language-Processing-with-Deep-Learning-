from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.framework import ops

import time
import logging
import sys 

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from utils import data_iterator

logging.basicConfig(level=logging.INFO)

def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]):
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[1])
    score_mask_values = score_mask_value * array_ops.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimize
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def get_optimizer(opt, loss,config,global_step):
    
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    learning_rate = tf.train.exponential_decay(config.lr, global_step,
                                           config.decay_steps, config.decay_rate, staircase=True)                                     
    learning_step=(optfn(learning_rate).minimize(loss, global_step=global_step))
    
    grads_and_vars = optfn(learning_rate).compute_gradients(loss)
    variables = [output[1] for output in grads_and_vars]
    gradients = [output[0] for output in grads_and_vars]

    gradients = tf.clip_by_global_norm(gradients, clip_norm=config.max_grad_norm)[0]
    grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
    train_op = optfn(learning_rate).apply_gradients(grads_and_vars)

    return train_op    

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  max_grad_norm=10.0
  batch_size = 10
  evaluate_sample_size=10
  model_selection_sample_size=10
  max_epochs = 10
  early_stopping = 2
  dropout = 0.85
  lr = 0.01
  decay_steps=100000
  decay_rate=0.96

     
class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

        
    def encode(self, inputs, masks, dropout,encoder_state_input=None,bidirection=True):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        #create LSTM cell
        
        if bidirection:
            lstm_fw = tf.nn.rnn_cell.LSTMCell(self.size)
            lstm_bw = tf.nn.rnn_cell.LSTMCell(self.size)
            lstm_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_fw, input_keep_prob = dropout,dtype=tf.float32)
            lstm_bw= tf.nn.rnn_cell.DropoutWrapper(lstm_bw, input_keep_prob = dropout,dtype=tf.float32)
            
            sequence_length=tf.reshape(tf.reduce_sum(tf.cast(masks,tf.int32),axis=1),[-1,])
            outputs,final_states=tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,inputs,sequence_length=sequence_length,
                                                      initial_state_fw=encoder_state_input,initial_state_bw=encoder_state_input,dtype=tf.float32)
            encoding=tf.concat(outputs, 2)
        else:
           lstm= tf.nn.rnn_cell.LSTMCell(self.size)
           #lstm= tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob =dropout,dtype=tf.float32)
           sequence_length=tf.reshape(tf.reduce_sum(tf.cast(masks,tf.int32),axis=1),[-1,])
           outputs,final_states=tf.nn.dynamic_rnn(lstm,tf.cast(inputs,dtype=tf.float32),sequence_length=sequence_length,
                                                      initial_state=encoder_state_input,dtype=tf.float32)
                                                      
           encoding=outputs
        return encoding
        
    
class Decoder(object):
    def __init__(self, output_size,state_size):
        self.output_size = output_size
        self.state_size=state_size
        
    def run_lstm(self, encoded_rep, q_rep, masks):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks

        q_rep = tf.expand_dims(q_rep, 1) # (batch_size, 1, D)
        encoded_passage_shape = tf.shape(encoded_passage)[1]
        q_rep = tf.tile(q_rep, [1, encoded_passage_shape, 1])

        mixed_question_passage_rep = tf.concat([encoded_passage, q_rep], axis=-1)

        with tf.variable_scope("lstm_"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple = True)
            outputs,final_states = tf.nn.bidirectional_dynamic_rnn(cell, cell,mixed_question_passage_rep, dtype=tf.float32, scope ="rnn")    

        
        output_attender =tf.concat(outputs, -1)
        return output_attender
        
        
    def run_match_lstm(self, encoded_rep, masks):
        encoded_question, encoded_passage = encoded_rep
        masks_question, masks_passage = masks
        question_length=tf.reshape(tf.reduce_sum(tf.cast(masks_question,tf.int32),axis=1),[-1,])
       

        match_lstm_cell_attention_fn = lambda curr_input, state : tf.concat([curr_input, state], axis = -1)
        query_depth = encoded_question.get_shape()[-1]


        # output attention is false because we want to output the cell output and not the attention values
        with tf.variable_scope("match_lstm_attender"):
            attention_mechanism_match_lstm = tf.contrib.seq2seq.BahdanauAttention(query_depth, encoded_question, memory_sequence_length = question_length)
            cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple = True)
            lstm_attender  = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism_match_lstm, output_attention = False, cell_input_fn = match_lstm_cell_attention_fn)


            outputs,final_states = tf.nn.bidirectional_dynamic_rnn(lstm_attender, lstm_attender,encoded_passage, dtype=tf.float32, scope ="rnn")    

        
        output_attender =tf.concat(outputs, -1)
        return output_attender
        
    def run_answer_ptr(self, output_attender, masks, labels):
        #batch_size = tf.shape(output_attender)[0]
        masks_question, masks_passage = masks
        labels = tf.unstack(labels, axis=1)
        #labels_s,labels_e=tf.split(labels, num_or_size_splits=2, axis=1)
        passage_length=tf.reshape(tf.reduce_sum(tf.cast(masks_passage,tf.int32),axis=1),[-1,])
        #labels = tf.ones([batch_size, 2, 1])


        answer_ptr_cell_input_fn = lambda curr_input, context : context # independent of question
        query_depth_answer_ptr = output_attender.get_shape()[-1]

        with tf.variable_scope("answer_ptr_attender"):
            attention_mechanism_answer_ptr = tf.contrib.seq2seq.BahdanauAttention(query_depth_answer_ptr , output_attender, memory_sequence_length = passage_length)
            # output attention is true because we want to output the attention values
            cell_answer_ptr = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple = True )
            answer_ptr_attender = tf.contrib.seq2seq.AttentionWrapper(cell_answer_ptr, attention_mechanism_answer_ptr, cell_input_fn = answer_ptr_cell_input_fn)
            logits, _ = tf.nn.static_rnn(answer_ptr_attender, labels, dtype = tf.float32)

        return logits 
        
        
    def decode(self, encoded_rep,masks,labels):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
    
        
        output_lstm = self.run_lstm(encoded_rep, q_rep, masks)
        output_attender = self.run_match_lstm(encoded_rep, masks)
        logits = self.run_answer_ptr(output_attender, masks, labels)
        
        return logits

class QASystem(object):
    def __init__(self, encoder, decoder,embed,vocab):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.vocab=vocab
        self.config=Config()
        self.encoder=encoder
        self.decoder=decoder
        self.embeddings=embed
        self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(dtype=tf.bool, name="q_mask", shape=(None, None))
        self.context_placeholder = tf.placeholder(dtype=tf.int32, name="c", shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(dtype=tf.bool, name="c_mask", shape=(None, None))
        # self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, config.answer_size))
        self.answer_placeholder = tf.placeholder(dtype=tf.int32, name="a_s", shape=(None,2))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=[])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embed,self.context_embed=self.setup_embeddings()
            self.pred_logits=self.setup_system()
            self.loss=self.setup_loss()
            
            self.global_step = tf.Variable(0, trainable=False)
            
        # ==== set up training/updating procedure ====
      
        self.train_op = get_optimizer("adam", self.loss,self.config,self.global_step)
        self.saver = tf.train.Saver()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        with tf.variable_scope("encoded_question"):
             quest_rep=self.encoder.encode(self.question_embed,masks=self.question_mask_placeholder,dropout=self.dropout_placeholder,bidirection=False)
        with tf.variable_scope("encoded_passage"):      
             contxt_rep=self.encoder.encode(self.context_embed,masks=self.context_mask_placeholder,dropout=self.dropout_placeholder,bidirection=False)
        
        encoded_Rep=[quest_rep,contxt_rep]
        
        logits=self.decoder.decode(encoded_Rep,[self.question_mask_placeholder,self.context_mask_placeholder],self.answer_placeholder)
        
        return logits
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_logits[0], labels=self.answer_placeholder[:,0])
            losses += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_logits[1], labels=self.answer_placeholder[:,1])
        return tf.reduce_mean(losses)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            question_embed = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
            #question_embed_list=tf.split(question_embed,self.config.num_steps,1)
            #question_inputs=[tf.squeeze(i) for i in question_embed_list ]
            
            context_embed = tf.nn.embedding_lookup(self.embeddings, self.context_placeholder)
            #context_embed_list=tf.split(context_embed,self.config.num_steps,1)
            #context_inputs=[tf.squeeze(i) for i in context_embed_list ]
        return  question_embed, context_embed  

    def optimize(self, session, train_dataset):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        
        q=train_dataset[0]
        c=train_dataset[1]
        a=train_dataset[2]     
        q_pad,q_mask,c_pad,c_mask=self.add_padding(q,c)
          
        input_feed = {self.question_placeholder: q_pad,
                  self.question_mask_placeholder: q_mask,
                  self.context_placeholder: c_pad,
                  self.context_mask_placeholder:c_mask,
                  self.answer_placeholder:a,
                  self.dropout_placeholder: self.config.dropout
                  }                                   

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        output_feed =[self.loss,self.train_op]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_dataset):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        question=valid_dataset[0]
        context=valid_dataset[1]
        answer=valid_dataset[2]
        q_pad,q_mask,c_pad,c_mask=self.add_padding(question,context)
        
        input_feed = {self.question_placeholder: q_pad,
                  self.question_mask_placeholder: q_mask,
                  self.context_placeholder: c_pad,
                  self.context_mask_placeholder:c_mask,
                  self.answer_placeholder:answer,
                  self.dropout_placeholder: self.config.dropout}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed =self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        question=test_x[0]
        context=test_x[1]
        answer=test_x[2]
        q_pad,q_mask,c_pad,c_mask=self.add_padding(question,context)
        
        input_feed = {self.question_placeholder: q_pad,
                  self.question_mask_placeholder: q_mask,
                  self.context_placeholder: c_pad,
                  self.context_mask_placeholder:c_mask,
                  self.answer_placeholder:answer,
                  self.dropout_placeholder: self.config.dropout}    

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.pred_logits]

        outputs = session.run(output_feed, input_feed)

        return outputs[0][0], outputs[0][1]

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)
        
        def search(y1, y2):
            max_ans = -999999
            a_s, a_e= 0,0
            num_classes = len(y1)
            for i in xrange(num_classes):
                for j in xrange(15):
                    if i+j >= num_classes:
                        break

                    curr_a_s = y1[i];
                    curr_a_e = y2[i+j]
                    if (curr_a_e+curr_a_s) > max_ans:
                        max_ans = curr_a_e + curr_a_s
                        a_s = i
                        a_e = i+j

            return (a_s, a_e)


        answer=[]
        for i in xrange(yp.shape[0]):
            _a_s, _a_e = search(yp[i], yp2[i])
            answer.append((_a_s,_a_e))

        return answer

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = []
        for step, [q,c, a] in enumerate(data_iterator(valid_dataset, self.config.batch_size)):
             loss = self.test(sess, [q,c,a])
             valid_cost.append(loss)
        return np.mean(valid_cost)             
           
       


    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.
        
        question=dataset[0]
        context=dataset[1]
        answer=dataset[2]
        N = len(question)
        sampleIndices = np.random.choice(N, sample, replace=False)
        q_evaluate = question[sampleIndices]
        c_evaluate=context[sampleIndices]
        a_evaluate=answer[sampleIndices]
        evaluate_set=[q_evaluate,c_evaluate,a_evaluate]
        
        preds=self.answer(session,evaluate_set)
        for i in range(sample):
            true_s,true_e=a_evaluate[i]
            start,end=preds[i]
            c=c_evaluate[i]
            # print (start, end, true_s, true_e)
            context_words = [self.vocab[w] for w in c]

            true_answer = ' '.join(context_words[true_s : true_e + 1])
            if start <= end:
                predict_answer = ' '.join(context_words[start : end + 1])
            else:
                predict_answer = ''
            f1 += f1_score(predict_answer, true_answer)
            em += exact_match_score(predict_answer, true_answer)


        f1 = 100 * f1 / sample
        em = 100 * em / sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
            
  

        return f1, em
        
    def add_padding(self,q_batch,c_batch):
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
        
    def run_epoch(self, session, data, verbose,sample_size):
        
        
        total_loss = []
        
        for step, (q,c, a) in enumerate(data_iterator(data, self.config.batch_size)):
                                       
          loss,_= self.optimize(session, [q,c,a])
          total_loss.append(loss)
          if verbose and step % verbose == 0:
              logging.info('')
              self.evaluate_answer(session, data, sample=sample_size, log=True)
        return np.mean(total_loss)
        
    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        
        
        training_set=dataset['train'] #[question,context,answer]
        validation_set=dataset['val']
        
        f1_best=0
        for epoch in range(self.config.max_epochs):
            logging.info("\n*********************EPOCH: %d*********************\n" %(epoch+1))
            avg_loss = self.run_epoch(session, training_set,verbose=10,sample_size=self.config.evaluate_sample_size)
            logging.info("\n*********************Average Loss: %d*********************\n" %(avg_loss))
            logging.info("-- validation --")
            val_loss=self.validate(session, validation_set)
            logging.info("\n*********************Validation Loss: %d*********************\n" %(val_loss))
            f1, em = self.evaluate_answer(session, validation_set, sample=self.config.model_selection_sample_size, log=True)
            # Saving the model
            if f1>f1_best:
                f1_best = f1
                self.saver.save(session, train_dir+'/fancier_model'+ str(epoch))
                logging.info('New best f1 in val set')
                logging.info('')
    
            
            