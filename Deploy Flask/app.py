from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

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


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text


mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def preprocess_sentence(w):

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
#     w = re.sub(r"([?.!,¿])", r" \1 ", w)
#     w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!¿]+", " ", w)

    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
#     w = '<start> ' + w + ' <end>'
    
    return w


# with open('indextosent.pkl', 'rb') as f:
#     idx2sent = pickle.load(f)

# with open('senttoindex.pkl', 'rb') as f:
#     sent2idx = pickle.load(f)

reverse_vocab = json.load( open( "idxtosent.json" ) )
vocab = json.load( open( "senttoindex.json" ) )


enc_sentence_length = 17
dec_sentence_length = 17

_START_ = "_START_"
_PAD_ = "_PAD_"
_END_ = "_END_"

def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", str(sentence))
    return tokens

def token2idx(word, vocab):
    return vocab[word]

def sent2idx(sent, vocab=vocab, max_sentence_length=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [2] + [1] * (pad_length-1), current_length + 1
    else:
        return [token2idx(token, vocab) for token in tokens] + [1] * pad_length, current_length

def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]

def idx2sent(indices, reverse_vocab=reverse_vocab):
    return " ".join([idx2token(str(idx), reverse_vocab) for idx in indices])

# # Enc Example
# print('hi what is your name?')
# print(sent2idx('hi what is your name?'))

# # Dec Example
# print('hi this is chinmay.')
# print(sent2idx('hi this is chinmay.', max_sentence_length=dec_sentence_length, is_target=True))

# print(idx2sent([0, 16, 41, 7, 36, 3, 2, 1, 1, 1, 1]))

from model import BasicS2SModel

def respond(inp):
    tf.reset_default_graph()
    # model = BasicS2SModel(vocab=vocab,num_layers=3)

    with tf.Session() as sess:
        model = BasicS2SModel(vocab=vocab,mode="inference",use_beam_search=False,num_layers=3)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess,tf.train.latest_checkpoint('./test_s2s/'))
        inputs = ['hi what is your name', inp]


        batch_preds = []
        batch_tokens = []
        batch_sent_lens = []
        for input_sent in inputs:
            tokens, sent_len = sent2idx(input_sent)
            batch_tokens.append(tokens)
            batch_sent_lens.append(sent_len)

        batch_preds = model.inference(sess,batch_tokens,batch_sent_lens)
    #       print(batch_preds)
        
        response = []
        preds = batch_preds[0][1]
        for i in range(1,len(preds)-1):
            if preds[i] == preds[i+1]:
                response.append(preds[i])
                # preds[i+1:9] = 1
                # preds[9] = 2
                break
            else:
                response.append(preds[i])



        # for input_sent, pred in zip(input_batch,  batch_preds[0]):
        #     print('Input:', input_sent)
        #     print(pred)
        #     print('Prediction:', idx2sent(pred, reverse_vocab=reverse_vocab))
        #     print('Target:', target_sent)

        response = idx2sent(response, reverse_vocab=reverse_vocab)
        return response



def preprocessing(text):
    text = text.lower()
    text = clean_contractions(text,contraction_mapping)
    text = clean_special_chars(text, punct, punct_mapping)
    text = correct_spelling(text, mispell_dict)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\/r\/[a-z]+ ', '', text, flags=re.MULTILINE)
    text = preprocess_sentence(text)
    return text





@app.route('/ask', methods=['POST'])
def index():
        text = request.form['messageText']
        if text=='exit':
            exit()
        else:
            sentence = preprocessing(text)
            bot_response = respond(sentence)
            # print(text)
            return jsonify(({'status':'OK','answer':bot_response}))



@app.route('/')
def my_form():
    return render_template('my-form.html')


if __name__ == '__main__':
    app.run(debug=True)




