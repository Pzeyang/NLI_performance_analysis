## Deep learning models###
# import statements
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Input, concatenate, dot, Flatten, Reshape, Bidirectional, add
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.regularizers import l2
from keras.layers import dot, Dot
from keras.activations import softmax
from keras.layers import Permute, subtract, multiply, GlobalAvgPool1D, GlobalMaxPool1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adadelta
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import functions.utils as utils
from functions.utils import *
import models.deep_learning as deep_learning
from models.deep_learning import *
from functions.model_evaluation import *
import functions.model_evaluation
from sklearn.model_selection import train_test_split
import csv


def write_csv(result, id_col, path):
    result_final = pd.DataFrame(data = list(zip(id_col, result)), columns = ['id', 'prediction'])
    result_final.to_csv(path, index = False)
    

 #loading the word vector
file_name = 'word_embedding/glove.6B.300d.txt'
embeddings = load_embedding(file_name)

## Quora Question paris Data set
#Loading the data
df = pd.read_csv('Data/train.csv')
df['question1'] = df['question1'].apply(str)
df['question2'] = df['question2'].apply(str)
df.dropna(inplace = True)
df = df[:1000]

#training and testing
seed = 123
train, test = train_test_split(df)
q = list(train['question1']) + list(train['question2']) + list(test['question1']) + list(test['question2'])
#Creating the embedding matrix
NUM_WORDS = len(embeddings) #200000
tokenize = Tokenizer(num_words = NUM_WORDS)
tokenize.fit_on_texts(q)
word_index = tokenize.word_index

q1_train = tokenize.texts_to_sequences(train['question1'])
q2_train = tokenize.texts_to_sequences(train['question2'])
q1_test = tokenize.texts_to_sequences(test['question1'])
q2_test = tokenize.texts_to_sequences(test['question2'])

max_seq_length = max_seq_len(q1_train)
max_seq_length = max_seq_len(q2_train, max_seq_length)
max_seq_length = max_seq_len(q1_test, max_seq_length)
max_seq_length = max_seq_len(q2_test, max_seq_length)

q1_train_padded = pad_sequences(q1_train, max_seq_length)
q2_train_padded = pad_sequences(q2_train, max_seq_length)
q1_test_padded = pad_sequences(q1_test, max_seq_length)
q2_test_padded = pad_sequences(q2_test, max_seq_length)

#Matrix with the embedding weights
embedding_dim = 300
embedding_weights = create_embedding_weights(embeddings, embedding_dim, word_index, NUM_WORDS)

NUM_WORDS = len(embedding_weights)
print("NUM of Words:"+ str(NUM_WORDS))

# Deep learning class
dl = deep_learning.deepModels(embedding_dim = embedding_dim,
                embedding_weights = embedding_weights,
                max_seq_length = max_seq_length,
                NUM_WORDS = NUM_WORDS)

#### Siamese LSTM####
# variables
batch_size=32
epochs=15
validation_split=.1

lstm = dl.siamese_lstm()

hist_lstm = lstm.fit([q1_train_padded, q2_train_padded],
                     train['is_duplicate'],
                     batch_size=batch_size, epochs=epochs, 
                     validation_split= validation_split)

pred_lstm = lstm.predict([q1_test_padded, q2_test_padded])

result_lstm = []
for i in pred_lstm:
    if i > .5:
        result_lstm.append(1)
    else:
        result_lstm.append(0)
        
count = 0
for i in range(len(result_lstm)):
    if result_lstm[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1

print("SIAMESE LSTM:")
accuracy = (count/len(result_lstm))*100
print('accuracy: ' + str(accuracy))
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_lstm, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_lstm, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_lstm, average = 'weighted')))
plot_model_training(hist_lstm)
write_csv(result_lstm, test['id'], 'results/deep_learning/quora/lstm.csv')

####Siamese GRU####
batch_size=100
epochs=15
validation_split=.1

gru = dl.Siamese_GRU()

hist_gru = gru.fit([q1_train_padded, 
                    q2_train_padded],
                   train['is_duplicate'],
                   batch_size=batch_size,
                   epochs=epochs, 
                   validation_split=validation_split)

pred_gru = gru.predict([q1_test_padded, q2_test_padded])

result_gru = []
for i in pred_gru:
    if i > .5:
        result_gru.append(1)
    else:
        result_gru.append(0)
        
count = 0
for i in range(len(result_gru)):
    if result_gru[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1
print("SIAMESE GRU:")
accuracy = (count/len(result_gru))*100
print("accuracy: " + str(accuracy))
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_gru, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_gru, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_gru, average = 'weighted')))

write_csv(result_gru, test['id'], 'results/deep_learning/quora/gru.csv')

#####Siamese CNN#####
batch_size = 100
epochs = 15
validation_split = 0.1

cnn = dl.siamese_cnn(filters= 16, kernel_size= 3)

hist_cnn = cnn.fit([q1_train_padded, q2_train_padded], 
                   train['is_duplicate'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_cnn = cnn.predict([q1_test_padded, q2_test_padded])

result_cnn = []
for i in pred_cnn:
    if i > .5:
        result_cnn.append(1)
    else:
        result_cnn.append(0)
        
count = 0
for i in range(len(result_cnn)):
    if result_cnn[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_cnn))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_cnn, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_cnn, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_cnn, average = 'weighted')))

plot_model_training(hist_cnn)
write_csv(result_cnn, test['id'], 'results/deep_learning/quora/cnn.csv')

#### Deep NN####
batch_size = 100
epochs = 15
validation_split = 0.1

snli = dl.deep_nn()

hist_snli = snli.fit([q1_train_padded, q2_train_padded], 
                   train['is_duplicate'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_snli = snli.predict([q1_test_padded, q2_test_padded])

result_snli = []
for i in pred_snli:
    if i > .5:
        result_snli.append(1)
    else:
        result_snli.append(0)
        
count = 0
for i in range(len(result_snli)):
    if result_snli[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_snli))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_snli, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_snli, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_snli, average = 'weighted')))

write_csv(result_snli, test['id'], 'results/deep_learning/quora/deep_nn.csv')

#####Hybrid CNN-LSTM #####
batch_size = 100
epochs = 15
validation_split = 0.1

hybrid = dl.hybrid_model()

hist_hybrid = hybrid.fit([q1_train_padded, q2_train_padded], 
                   train['is_duplicate'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_hybrid = hybrid.predict([q1_test_padded, q2_test_padded])

result_hybrid = []
for i in pred_hybrid:
    if i > .5:
        result_hybrid.append(1)
    else:
        result_hybrid.append(0)
        
count = 0
for i in range(len(result_hybrid)):
    if result_hybrid[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_hybrid))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_hybrid, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_hybrid, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_hybrid, average = 'weighted')))

plot_model_training(hist_hybrid)

write_csv(result_hybrid, test['id'], 'results/deep_learning/quora/hybrid.csv')

#####Enhanced LSTM#####
batch_size = 32
epochs = 5
validation_split = .1

e_lstm = dl.enhanced_lstm()

hist_e_lstm = e_lstm.fit([q1_train_padded, q2_train_padded], 
                   train['is_duplicate'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_e_lstm = e_lstm.predict([q1_test_padded, q2_test_padded])

result_elstm = []
for i in pred_e_lstm:
    if i > .5:
        result_elstm.append(1)
    else:
        result_elstm.append(0)
        
count = 0
for i in range(len(result_elstm)):
    if result_elstm[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_elstm))*100
print("accuracy: "  + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_elstm, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_elstm, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_elstm, average = 'weighted')))

plot_model_training(hist_e_lstm)

write_csv(result_elstm, test['id'], 'results/deep_learning/quora/elstm.csv')

####### MSR Paraphrase Dataset####
# Loading the training and testing data
train = pd.read_csv(r'Data/msr_paraphrase_train.txt', sep = '\t', quoting=csv.QUOTE_NONE)
test = pd.read_csv(r'Data/msr_paraphrase_test.txt', sep = '\t', quoting=csv.QUOTE_NONE)

sent = list(train['#1 String']) + list(train['#2 String']) + list(test['#1 String']) + list(test['#2 String'])

#Creating the embedding matrix
NUM_WORDS = len(embeddings) #200000
tokenize = Tokenizer(num_words = NUM_WORDS)
tokenize.fit_on_texts(sent)
word_index = tokenize.word_index

sent1_train = tokenize.texts_to_sequences(train['#1 String'])
sent2_train = tokenize.texts_to_sequences(train['#2 String'])
sent1_test = tokenize.texts_to_sequences(test['#1 String'])
sent2_test = tokenize.texts_to_sequences(test['#2 String'])

max_seq_length = max_seq_len(sent1_train)
max_seq_length = max_seq_len(sent2_train, max_seq_length)
max_seq_length = max_seq_len(sent1_test, max_seq_length)
max_seq_length = max_seq_len(sent2_test, max_seq_length)

sent1_train_padded = pad_sequences(sent1_train, max_seq_length)
sent2_train_padded = pad_sequences(sent2_train, max_seq_length)
sent1_test_padded = pad_sequences(sent1_test, max_seq_length)
sent2_test_padded = pad_sequences(sent2_test, max_seq_length)
#Matrix with the embedding weights
embedding_dim = 300
embedding_weights = create_embedding_weights(embeddings, embedding_dim, word_index, NUM_WORDS)
NUM_WORDS = len(embedding_weights)
NUM_WORDS

#####Deep learning models####
dl = deep_learning.deepModels(embedding_dim = embedding_dim,
                embedding_weights = embedding_weights,
                max_seq_length = max_seq_length,
                NUM_WORDS = NUM_WORDS)

####Siamese LSTM####
batch_size=32
epochs=15
validation_split=.1

lstm = dl.siamese_lstm(dropout_lstm=None)

hist_lstm = lstm.fit([sent1_train_padded, sent2_train_padded],
                     train['Quality'],
                     batch_size=batch_size, epochs=epochs, 
                     validation_split= validation_split)

pred_lstm = lstm.predict([sent1_test_padded, sent2_test_padded])

result_lstm = []
for i in pred_lstm:
    if i > .5:
        result_lstm.append(1)
    else:
        result_lstm.append(0)
        
count = 0
for i in range(len(result_lstm)):
    if result_lstm[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_lstm))*100
print('accuracy:'+ str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_lstm, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_lstm, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_lstm, average = 'weighted')))

plot_model_training(hist_lstm)

write_csv(result_lstm, test['id'], 'results/deep_learning/MSR/lstm.csv')

####Siamese GRU#####
batch_size=100
epochs=15
validation_split=.1

gru = dl.Siamese_GRU()

hist_gru = gru.fit([sent1_train_padded, 
                    sent2_train_padded],
                   train['Quality'],
                   batch_size=batch_size,
                   epochs=epochs, 
                   validation_split=validation_split)

pred_gru = gru.predict([sent1_test_padded, sent2_test_padded])

result_gru = []
for i in pred_gru:
    if i > .5:
        result_gru.append(1)
    else:
        result_gru.append(0)
        
count = 0
for i in range(len(result_gru)):
    if result_gru[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_gru))*100
print('accuracy: ' + str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_gru, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_gru, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_gru, average = 'weighted')))

plot_model_training(hist_gru)

write_csv(result_gru, test['id'], 'results/deep_learning/MSR/GRU.csv')

#####Siamese CNN#####
batch_size = 100
epochs = 15
validation_split = 0.1


cnn = dl.siamese_cnn(filters= 16, kernel_size= 3)

hist_cnn = cnn.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_cnn = cnn.predict([sent1_test_padded, sent2_test_padded])

result_cnn = []
for i in pred_cnn:
    if i > .5:
        result_cnn.append(1)
    else:
        result_cnn.append(0)
        
count = 0
for i in range(len(result_cnn)):
    if result_cnn[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_cnn))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_cnn, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_cnn, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_cnn, average = 'weighted')))

plot_model_training(hist_cnn)

write_csv(result_cnn, test['id'], 'results/deep_learning/MSR/CNN.csv')

####Deep Neural Network
batch_size = 100
epochs = 15
validation_split = 0.1

snli = dl.deep_nn()

hist_snli = snli.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_snli = snli.predict([sent1_test_padded, sent2_test_padded])

result_snli = []
for i in pred_snli:
    if i > .5:
        result_snli.append(1)
    else:
        result_snli.append(0)
        
count = 0
for i in range(len(result_snli)):
    if result_snli[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_snli))*100
print("accuracy: " + str(accuracy))
batch_size = 100
epochs = 15
validation_split = 0.1

hybrid = dl.hybrid_model()

hist_hybrid = hybrid.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_hybrid = hybrid.predict([sent1_test_padded, sent2_test_padded])

result_hybrid = []
for i in pred_hybrid:
    if i > .5:
        result_hybrid.append(1)
    else:
        result_hybrid.append(0)
        
count = 0
for i in range(len(result_hybrid)):
    if result_hybrid[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_hybrid))*100
print('accuracy: '+ str(accuracy))


print('f1_socre:' + str(f1_score(test['Quality'], result_hybrid, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_hybrid, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_hybrid, average = 'weighted')))

plot_model_training(hist_hybrid)

write_csv(result_hybrid, test['id'], 'results/deep_learning/MSR/hybrid.csv')

####Enhanced LSTM
batch_size = 100
epochs = 15
validation_split = 0.1

e_lstm = dl.enhanced_lstm()

hist_e_lstm = e_lstm.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_e_lstm = e_lstm.predict([sent1_test_padded, sent2_test_padded])

result_e_lstm = []
for i in pred_e_lstm:
    if i > .5:
        result_e_lstm.append(1)
    else:
        result_e_lstm.append(0)
        
count = 0
for i in range(len(result_e_lstm)):
    if result_e_lstm[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_e_lstm))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_e_lstm, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_e_lstm, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_e_lstm, average = 'weighted')))

plot_model_training(hist_e_lstm)

write_csv(result_elstm, test['id'], 'results/deep_learning/MSR/elstm.csv')
print('f1_socre:' + str(f1_score(test['Quality'], result_snli, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_snli, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_snli, average = 'weighted')))

plot_model_training(hist_snli)

write_csv(result_snli, test['id'], 'results/deep_learning/MSR/Deep_NN.csv')

####Hybrid Model

batch_size = 100
epochs = 15
validation_split = 0.1

hybrid = dl.hybrid_model()

hist_hybrid = hybrid.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_hybrid = hybrid.predict([sent1_test_padded, sent2_test_padded])

result_hybrid = []
for i in pred_hybrid:
    if i > .5:
        result_hybrid.append(1)
    else:
        result_hybrid.append(0)
        
count = 0
for i in range(len(result_hybrid)):
    if result_hybrid[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_hybrid))*100
print('accuracy: '+ str(accuracy))


print('f1_socre:' + str(f1_score(test['Quality'], result_hybrid, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_hybrid, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_hybrid, average = 'weighted')))

plot_model_training(hist_hybrid)

write_csv(result_hybrid, test['id'], 'results/deep_learning/MSR/hybrid.csv')

####Enhanced LSTM
batch_size = 100
epochs = 15
validation_split = 0.1

e_lstm = dl.enhanced_lstm()

hist_e_lstm = e_lstm.fit([sent1_train_padded, sent2_train_padded], 
                   train['Quality'], 
                   batch_size=batch_size, 
                   epochs=epochs, 
                   validation_split=validation_split)

pred_e_lstm = e_lstm.predict([sent1_test_padded, sent2_test_padded])

result_e_lstm = []
for i in pred_e_lstm:
    if i > .5:
        result_e_lstm.append(1)
    else:
        result_e_lstm.append(0)
        
count = 0
for i in range(len(result_e_lstm)):
    if result_e_lstm[i] == test['Quality'].to_numpy()[i]:
        count = count +1

accuracy = (count/len(result_e_lstm))*100
print("accuracy: " + str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_e_lstm, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_e_lstm, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_e_lstm, average = 'weighted')))

plot_model_training(hist_e_lstm)

write_csv(result_elstm, test['id'], 'results/deep_learning/MSR/elstm.csv')