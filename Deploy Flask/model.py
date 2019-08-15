import collections

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers import Embedding, CuDNNLSTM, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import helper

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.contrib.seq2seq import AttentionWrapper as attention_wrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder as beam_search_decoder
# print('TensorFlow Version: {}'.format(tf.__version__))
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

_START_ = "_START_"
_PAD_ = "_PAD_"
_END_ = "_END_"


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


def single_rnn_cell(cell_name,dim_size, train_phase = True, keep_prob = 0.75):
    if cell_name == "gru":
        cell = tf.contrib.rnn.GRUCell(dim_size)
    elif cell_name == "lstm":
        cell = tf.contrib.rnn.LSTMCell(dim_size)
    else:
        cell = tf.contrib.rnn.BasicRNNCell(dim_size)
    if train_phase and keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(
              cell=cell,
              input_keep_prob=keep_prob,
              output_keep_prob=keep_prob)
    return cell

def multi_rnn_cell(cell_name,dim_size,num_layers = 1, train_phase = True, keep_prob=0.75):
    cells = []
    for _ in range(num_layers):
        cell = single_rnn_cell(cell_name,dim_size, train_phase, keep_prob)
        cells.append(cell)
    
    if len(cells) > 1:
        final_cell = tf.contrib.rnn.MultiRNNCell(cells=cells)
    else:
        final_cell = cells[0]
    return final_cell

class BasicS2SModel(object):
    def __init__(self, vocab, batch_size = 2, dim_size=128, rnn_cell = 'gru', num_layers=2, max_gradient_norm=5.0, atten_size=30, 
                 learning_rate=0.001, learning_rate_decay_factor=0.98, dropout=0.2,max_inference_lenght=10,
                 max_source_len = 10, max_target_len = 10,beam_size =3, optimizer="adam", mode ='train',
                 use_beam_search = False):
        assert mode in ['train', 'inference']
        self.start_token = vocab.get(_START_)
        self.end_token = vocab.get(_END_)
        self.train_phase = True if mode == 'train' else False
        self.cell_name = rnn_cell
        self.dim_size = dim_size
        self.vocab_size = len(vocab)
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.atten_size = atten_size
        
        # decoder
        self.max_inference_lenght = max_inference_lenght
        
        # beam search
        self.beam_size = beam_size
        self.beam_search = use_beam_search
        
        # learning
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        
        # if we use beam search decoder, we need to specify the batch size and max source len
        if self.beam_search:
            self.batch_size = batch_size
            self.source_tokens = tf.placeholder(tf.int32, shape=[batch_size, max_source_len])
            self.source_length = tf.placeholder(tf.int32, shape=[batch_size,])
        else:
            self.source_tokens = tf.placeholder(tf.int32,shape=[None,None])
            self.source_length = tf.placeholder(tf.int32,shape=[None,])
            
        if self.train_phase:
            self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
            self.target_length = tf.placeholder(tf.int32, shape=[None,])
         
        with tf.variable_scope("S2S",initializer = tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            #self.setup_encoder()
            self.setup_bidirection_encoder()
            self.setup_attention_decoder()
                
        if self.train_phase:
            opt = get_optimizer(optimizer)(self.learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.gradient_norm = tf.global_norm(gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    
    def setup_embeddings(self):
        with tf.variable_scope("Embeddings"):
            with tf.device('/cpu:0'):
                self.enc_emd = tf.get_variable("encode_embedding", [self.vocab_size, self.dim_size])
                self.dec_emd = tf.get_variable("decode_embedding", [self.vocab_size, self.dim_size])
                self.encoder_inputs = tf.nn.embedding_lookup(self.enc_emd, self.source_tokens)
                if self.train_phase:
                    self.decoder_inputs = tf.nn.embedding_lookup(self.dec_emd, self.target_tokens)
    
    def setup_encoder(self):
        cell = multi_rnn_cell(self.cell_name,self.dim_size, self.num_layers, self.train_phase,self.keep_prob_config)
        outputs,state = tf.nn.dynamic_rnn(cell,inputs=self.encoder_inputs,sequence_length=self.source_length,dtype=tf.float32)
        self.encode_output = outputs
        self.encode_state = state
        # using the state of last layer of rnn as initial state
        self.decode_initial_state = self.encode_state[-1]
        
    def setup_bidirection_encoder(self):
        fw_cell = single_rnn_cell('gru',self.dim_size, train_phase=self.train_phase, keep_prob=self.keep_prob_config)
        bw_cell = single_rnn_cell('gru',self.dim_size, train_phase=self.train_phase, keep_prob=self.keep_prob_config)
        
        with tf.variable_scope("Encoder"):
            outputs,states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = fw_cell,
                cell_bw = bw_cell,
                dtype = tf.float32,
                sequence_length = self.source_length,
                inputs = self.encoder_inputs
                )
            outputs_concat = tf.concat(outputs, 2)
        self.encode_output = outputs_concat
        self.encode_state = states
        
        # use Dense layer to convert bi-direction state to decoder inital state
        convert_layer = Dense(self.dim_size,dtype=tf.float32,name="bi_convert")
        self.decode_initial_state = convert_layer(tf.concat(self.encode_state,axis=1))
        
    def setup_training_decoder_layer(self):
        max_dec_len = tf.reduce_max(self.target_length, name='max_dec_len')
        training_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs,self.target_length,name="training_helper")
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.dec_cell,
            helper = training_helper,
            initial_state = self.initial_state,
            output_layer = self.output_layer
        )
        train_dec_outputs, train_dec_last_state,_ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_dec_len)
        
        # logits: [batch_size x max_dec_len x vocab_size]
        logits = tf.identity(train_dec_outputs.rnn_output, name='logits')

        # targets: [batch_size x max_dec_len x vocab_size]
        targets = tf.slice(self.target_tokens, [0, 0], [-1, max_dec_len], 'targets')

        masks = tf.sequence_mask(self.target_length,max_dec_len,dtype=tf.float32,name="mask")
        self.losses = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=targets,weights=masks,name="losses")
        
        # prediction sample for validation
        self.valid_predictions = tf.identity(train_dec_outputs.sample_id, name='valid_preds')
    
    def setup_inference_decoder_layer(self):
        start_tokens = tf.tile(tf.constant([self.start_token],dtype=tf.int32),[self.batch_size])
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.dec_emd,
            start_tokens=start_tokens,
            end_token=self.end_token)
        
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = self.dec_cell,
            helper = inference_helper, 
            initial_state=self.initial_state,
            output_layer=self.output_layer)
        
        infer_dec_outputs, infer_dec_last_state,_ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.max_inference_lenght)
        # [batch_size x dec_sentence_length], tf.int32
        self.predictions = tf.identity(infer_dec_outputs.sample_id, name='predictions')
            
    def setup_beam_search_decoder_layer(self):
        start_tokens = tf.tile(tf.constant([self.start_token],dtype=tf.int32),[self.batch_size])
        bsd = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.dec_cell,
                    embedding=self.dec_emd,
                    start_tokens= start_tokens,
                    end_token=self.end_token,
                    initial_state=self.initial_state,
                    beam_width=self.beam_size,
                    output_layer=self.output_layer,
                    length_penalty_weight=0.0)
        # final_outputs are instances of FinalBeamSearchDecoderOutput
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            bsd, 
            output_time_major=False,
           # impute_finished=True,
            maximum_iterations=self.max_inference_lenght
        )
        beam_predictions = final_outputs.predicted_ids
        self.beam_predictions = tf.transpose(beam_predictions,perm=[0,2,1])
        self.beam_prob = final_outputs.beam_search_decoder_output.scores
        self.beam_ids = final_outputs.beam_search_decoder_output.predicted_ids
        
    def setup_attention_decoder(self):
        #dec_cell = multi_rnn_cell('gru',self.dim_size,num_layers=self.num_layers, train_phase = self.train_phase, keep_prob=self.keep_prob_config)
        dec_cell = [single_rnn_cell(self.cell_name,self.dim_size,self.train_phase,self.keep_prob_config) for i in range(self.num_layers)]
        if self.beam_search:
            memory = tf.contrib.seq2seq.tile_batch(self.encode_output,multiplier = self.beam_size)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(self.source_length,multiplier = self.beam_size)
        else:
            memory = self.encode_output
            memory_sequence_length = self.source_length
            
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            num_units = self.atten_size,
            memory = memory,
            memory_sequence_length = memory_sequence_length,
            name = "BahdanauAttention"
        )
        dec_cell[0] = tf.contrib.seq2seq.AttentionWrapper(
            cell=dec_cell[0],
            attention_mechanism=attn_mech,
            attention_layer_size=self.atten_size)
        
        if self.beam_search:
            tile_state = tf.contrib.seq2seq.tile_batch(self.decode_initial_state,self.beam_size)
            initial_state = [tile_state for i in range(self.num_layers)]
            cell_state = dec_cell[0].zero_state(dtype=tf.float32,batch_size=self.batch_size*self.beam_size)
            initial_state[0] = cell_state.clone(cell_state=initial_state[0])
            self.initial_state = tuple(initial_state)
        else:
            # we use dynamic batch size
            self.batch_size = tf.shape(self.encoder_inputs)[0]
            initial_state = [self.decode_initial_state for i in range(self.num_layers)]
            cell_state = dec_cell[0].zero_state(dtype=tf.float32, batch_size = self.batch_size)
            initial_state[0] = cell_state.clone(cell_state=initial_state[0])
            self.initial_state = tuple(initial_state)
            
        print(self.initial_state)
        self.dec_cell = tf.contrib.rnn.MultiRNNCell(dec_cell)
        self.output_layer = Dense(self.vocab_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        if self.train_phase:
            self.setup_training_decoder_layer()
        else:
            if self.beam_search:
                self.setup_beam_search_decoder_layer()
            else:
                self.setup_inference_decoder_layer()
    
    def train_one_step(self,sess,encode_input,encode_len,decode_input,decode_len):
        feed_dict = {}
        feed_dict[self.source_tokens] = encode_input
        feed_dict[self.source_length] = encode_len
        feed_dict[self.target_tokens] = decode_input
        feed_dict[self.target_length] = decode_len
        valid_predictions,loss,_ = sess.run([self.valid_predictions,self.losses,self.updates],feed_dict=feed_dict)
        return valid_predictions,loss
    
    def inference(self,sess,encode_input,encode_len):
        feed_dict = {}
        feed_dict[self.source_tokens] = encode_input
        feed_dict[self.source_length] = encode_len
        if self.beam_search:
            predictions,probs,ids = sess.run([self.beam_predictions,self.beam_prob,self.beam_ids],feed_dict=feed_dict)
            return predictions,ids
        else:
            predictions = sess.run([self.predictions],feed_dict=feed_dict)
            return predictions
    
    def save_model(self,sess,checkpoint_dir):
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess,checkpoint_dir + "model.ckpt",global_step=self.global_step)
        
    def restore_model(self,sess,checkpoint_dir):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))