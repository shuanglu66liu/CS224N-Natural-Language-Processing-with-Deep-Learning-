from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt,lr):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(lr)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(lr)
    else:
        assert (False)
    return optfn


      
class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def preprocess(self,inputs,masks,encoder_state_input):
        lstm= tf.nn.rnn_cell.LSTMCell(self.size)
        sequence_length=tf.reshape(tf.reduce_sum(tf.cast(masks,tf.int32),axis=1),[-1,])
        outputs,_=tf.nn.dynamic_rnn(lstm,inputs,sequence_length=sequence_length,initial_state=encoder_state_input)
        return  outputs
        
    def encode(self, inputs, masks, encoder_state_input):
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
        
        lstm_fw = tf.nn.rnn_cell.LSTMCell(self.size)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(self.size)
        sequence_length=tf.reshape(tf.reduce_sum(tf.cast(masks,tf.int32),axis=1),[-1,])
        outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,inputs,sequence_length=sequence_length,
                                                  initial_state_fw=encoder_state_input,initial_state_bw=encoder_state_input)
        enconding=tf.concat(outputs, 2)
        
        return enconding
        
    
class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
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
        lstm = tf.nn.rnn_cell.LSTMCell(self.output_size)
        outputs,states=tf.nn.dynamic_rnn(lstm,knowledge_rep ,dtype=tf.float32)
        
        return

class QASystem(object):
    def __init__(self, encoder, decoder, config,embed):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        self.config=config
        self.encoder=encoder
        self.decoder=decoder
        self.embeddings=embed
        self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(dtype=tf.bool, name="q_mask", shape=(None, None))
        self.context_placeholder = tf.placeholder(dtype=tf.int32, name="c", shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(dtype=tf.bool, name="c_mask", shape=(None, None))
        # self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, config.answer_size))
        self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a_s", shape=(None,None))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=())

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embed,self.context_embed=self.setup_embeddings()
            self.prediction=self.setup_system()
            self.loss=self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        quest_lstm=self.encoder.encode(self.question_embed,masks=self.question_mask_placeholder)
        contxt_lstm=self.encoder.encode(self.context_embed,masks=self.context_mask_placeholder)
        
        knowledge_Rep={"question":quest_lstm,"context":contxt_lstm}        
        answer_pred=self.decoder.decode(knowledge_Rep)
        
        return answer_pred
        #raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

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
        learning_rate = tf.train.exponential_decay(self.config.lr, self.config.global_step,
                                           self.config.decay_steps, self.config.decay_rate, staircase=True)
        input_feed = {"train_dataset":train_dataset}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        loss=self.setup_loss(train_dataset)
        output_feed = get_optimizer(self.opt,learning_rate).minimize(loss)

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_dataset):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {'session':session,"dataset":valid_dataset}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed =self.evaluate_answer

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        valid_cost = self.test(sess, valid_dataset)


        return valid_cost

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

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
            
        test_x={ key: dataset[key] for key in ["context","question"] }
        prediction=self.answer(session,test_x)
        actual=dataset["answer"]
        f1=f1_score(prediction,actual)
        em=exact_match_score(prediction,actual)

        return f1, em
        
    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
          train_op = tf.no_op()
          dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(
          ptb_iterator(data, config.batch_size, config.num_steps)):
          # We need to pass in the initial state and retrieve the final state to give
          # the RNN proper history
          feed = {self.input_placeholder: x,
                  self.labels_placeholder: y,
                  self.initial_state: state,
                  self.dropout_placeholder: dp}
          loss, state, _ = session.run(
              [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
          total_loss.append(loss)
          if verbose and step % verbose == 0:
              sys.stdout.write('\r{} / {} : pp = {}'.format(
                  step, total_steps, np.exp(np.mean(total_loss))))
              sys.stdout.flush()
        if verbose:
          sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))
        
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
        saver = tf.train.Saver() 
        
        with tf.Session() as session:
            
            best_val_pp = float('inf')
            best_val_epoch = 0
      
            session.run(tf.initialize_all_variables())
            for epoch in range(self.config.max_epochs):
              print ('Epoch {}'.format(epoch))
              start = time.time()
              ###
              train_pp = self.run_epoch(
                  session, dataset['train'],
                  train_op=model.train_step)
              valid_pp = self.run_epoch(session, dataset['valid'])
              print ('Training perplexity: {}'.format(train_pp))
              print ('Validation perplexity: {}'.format(valid_pp))
              if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
              if epoch - best_val_epoch > config.early_stopping:
                break
              print ('Total time: {}'.format(time.time() - start))
              
            saver.restore(session, 'ptb_rnnlm.weights')
            test_pp = model.run_epoch(session, model.encoded_test)
            print ('=-=' * 5)
            print ('Test perplexity: {}'.format(test_pp))
            print ('=-=' * 5)
            starting_text = 'in palo alto'
            while starting_text:
              print (' '.join(generate_sentence(
                  session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))
              starting_text = input('> ')
