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

class deepModels():    
    def __init__(self, embedding_dim, embedding_weights, max_seq_length, NUM_WORDS):
        
        """Initializing the required variables
        """
        
        self.embedding_dim = embedding_dim
        self.embedding_weights = embedding_weights
        self.max_seq_length = max_seq_length
        self.NUM_WORDS = NUM_WORDS
        
    def embedding_layer(self):
        
        """Function to create embedding_layer, returns the embedding layer
        """
        
        embedding_layer = Embedding(
        self.NUM_WORDS,
        self.embedding_dim,
        weights = [self.embedding_weights], 
        input_length = self.max_seq_length,
        trainable = False)
        return embedding_layer
    
    def siamese_lstm(self,
                     reg_lstm = None, 
                     dropout_lstm = None,
                    dropout_embedding = None,
                    dropout_merge = None,
                    kernel_reg_lstm = None,
                    ):
        """Function to create siamese LSTM model
        """
        # Left input
        if kernel_reg_lstm != None:
            lstm = LSTM(self.embedding_dim, 
                    kernel_regularizer=l2(kernel_reg_lstm),
                    dropout = dropout_lstm
                       )
        else:
            lstm = LSTM(self.embedding_dim)
                
        lstm = LSTM(self.embedding_dim)
        embedding_layer = self.embedding_layer()
        
        left_input = Input(shape=(self.max_seq_length,), name='input_1')
        left_output = embedding_layer(left_input)
        if dropout_embedding != None:
            left_output = Dropout(dropout_embedding)(left_output)
            
        left_output = lstm(left_output)
        left_output = BatchNormalization()(left_output)        

        # Right input
        right_input = Input(shape=(self.max_seq_length,), name='input_2')
        right_output = embedding_layer(right_input)
        if dropout_embedding != None:
            right_output = Dropout(dropout_embedding)(right_output)
        right_output = lstm(right_output)
        right_output = BatchNormalization()(right_output)        
    
        # merging the networks using absolute distance
        #***to do*** adding different distance measures
        dist_normal = lambda x: 1 - K.abs(x[0] - x[1])
        
        merged = Lambda(function=dist_normal, output_shape=lambda x: x[0], 
                                       name='L1_distance')([left_output, right_output])
        if dropout_merge == True:
            merged = Dropout(dropout_merge)(merged)
        predictions = Dense(1, activation='sigmoid', name='sentence_duplicate')(merged)
        
        model = Model([left_input, right_input], predictions)

        model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])    
      
        return model
    
    #**to do** this function should be made similar to siamese lstm
    def siamese_cnn(self,
                    filters,
                    kernel_size,
                    kernel_reg = None, 
                    dropout_embedding = None,
                    dropout_merge = None):
        
        """Function to create siamese CNN model"""
        
        x = Input(shape = (self.max_seq_length,))
        
        y = Input(shape = (self.max_seq_length,))
        
        embedding_layer = self.embedding_layer()
        
        left_input = embedding_layer(x)
        right_input = embedding_layer(y)
        
        if dropout_embedding !=None:
            left_input = Dropout(dropout_embedding)(left_input)
            right_input = Dropout(dropout_embedding)(right_input)
        
        left_output = Conv1D(filters = 16, 
                             kernel_size=3,
                             padding = 'valid',
                             activation = 'relu'
                             )(left_input)
        
        left_output = MaxPooling1D()(left_output)
        
        left_output = Conv1D(filters = 32, 
                             kernel_size = 3,
                             padding = 'valid',
                             activation = 'relu'
                             )(left_output)
        
        left_output = MaxPooling1D()(left_output)
        
        left_output = Flatten()(left_output)

        right_output = Conv1D(filters = 16, 
                              kernel_size=3,
                              padding = 'valid', 
                              activation = 'relu'
                              
                             )(right_input)
        
        right_output = MaxPooling1D()(right_output)
        
        right_output = Conv1D(filters = 32,
                              kernel_size = 3,
                              padding = 'valid',
                              activation = 'relu',
                              )(right_output)
        
        right_output = MaxPooling1D()(right_output)
        
        right_output = Flatten()(right_output)
        
        l1_layers = Lambda(lambda x: K.abs(x[0] - x[1]))
        l1_distance = l1_layers([left_output, right_output])
        if dropout_merge !=None:
            l1_distance = Dropout(dropout_merge)(l1_distance)

        similarity = Dense(1, name = 'duplicate', activation = 'sigmoid')(l1_distance)

        cnn = Model([x, y], similarity)
        
        cnn.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

        return cnn       
    
    def aggregate(self, input_1, input_2, num_dense=300, dropout_rate=0.1):
        feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
        feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
        x = concatenate([feat1, feat2])
        x = BatchNormalization()(x)
        x = Dense(num_dense, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(num_dense, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        return x    

    def align(self, input_1, input_2):
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
        w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2))(attention))
        in1_aligned = Dot(axes=1)([w_att_1, input_1])
        in2_aligned = Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned

    
    def enhanced_lstm(self):
        
        """Function to enhance lstm"""
        
        q1 = Input(shape = (self.max_seq_length,))
        q2 = Input(shape = (self.max_seq_length,))
        embedding_layer = self.embedding_layer()
        left_input = embedding_layer(q1)
        right_input = embedding_layer(q2)
        left_input = BatchNormalization(axis = 2)(left_input)
        right_input = BatchNormalization(axis = 2)(right_input)
        
        ###Encoding####
        encode = Bidirectional(LSTM(self.embedding_dim, return_sequences = True))
        left_output = encode(left_input)
        right_output = encode(right_input)
      
        ###Aligning###
        left_aligned, right_aligned = self.align(left_output, right_output)

        #comparing
        q1_combined = concatenate([left_output, right_aligned, subtract([left_output, right_aligned]), multiply([left_output, right_aligned])])
        q2_combined = concatenate([right_output, left_aligned, subtract([right_output, left_aligned]), multiply([right_output, left_aligned])]) 
        compare = Bidirectional(LSTM(self.embedding_dim, return_sequences=True))
        q1_compare = compare(q1_combined)
        q2_compare = compare(q2_combined)

        #aggregating
        x = self.aggregate(q1_compare, q2_compare, dropout_rate=.1)
        x = Dense(1, activation='sigmoid')(x)

        model = Model([q1, q2], x)

        model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
        
        return model
    
    def hybrid_model(self):
        x = Input(shape = (self.max_seq_length,))
        y = Input(shape = (self.max_seq_length,))

        embedding = Embedding(len(self.embedding_weights), self.embedding_dim, weights=[self.embedding_weights], 
                         input_length=self.max_seq_length, 
                         trainable=False )
        left_input = embedding(x)
        right_input = embedding(y)

        left_output = Conv1D(filters = 16, kernel_size=3, padding = 'valid', activation = 'relu')(left_input)
        left_output = MaxPooling1D()(left_output)
        left_output = Conv1D(filters = 16, kernel_size = 3, padding = 'valid', activation = 'relu')(left_output)
        left_output = MaxPooling1D()(left_output)
        #left_output = (Flatten()(TimeDistributed((left_output))
        left_output = MaxPooling1D(pool_size=4)(left_output)
        left_output = Dropout(.2)(left_output)
        left_output = Bidirectional(LSTM(self.embedding_dim))(left_output)

        right_output = Conv1D(filters = 16, kernel_size=3, padding = 'valid', activation = 'relu')(right_input)
        right_output = MaxPooling1D()(right_output)
        right_output = Conv1D(filters = 16, kernel_size = 3, padding = 'valid', activation = 'relu')(right_output)
        right_output = MaxPooling1D()(right_output)
        right_output = MaxPooling1D(pool_size=4)(right_output)
        right_output = Dropout(.2)(right_output)
        right_output = Bidirectional(LSTM(self.embedding_dim))(right_output)


        dist_normal = lambda x: 1 - K.abs(x[0] - x[1])
        merged = Lambda(function=dist_normal, output_shape=lambda x: x[0], 
                                       name='L1_distance')([left_output, right_output])
        predictions = Dense(1, activation='sigmoid', name='sentence_duplicate')(merged)
        model = Model([x, y], predictions)

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model
    
    def Siamese_GRU(self,
                    kernel_reg = None,
                    bias_reg= None,
                    dropout_embedding = None,
                    dropout_merged = None,
                    dropout_gru = None
                   ):
        
        if kernel_reg!= None:
            gru = GRU(self.embedding_dim, dropout= dropout_gru, kernel_regularizer=l2(kernel_reg))
        else:
            gru = GRU(self.embedding_dim)
        
        embedding_layer = self.embedding_layer()
        
        left_input = Input(shape=(self.max_seq_length,), name='input_1')
        left_output = embedding_layer(left_input)
        if dropout_embedding !=None:
            left_output = Dropout(dropout_embedding)
        left_output = gru(left_output)
        left_output = BatchNormalization()(left_output)
        
        right_input = Input(shape=(self.max_seq_length,), name='input_2')
        right_output = embedding_layer(right_input)
        if dropout_embedding !=None:
            right_output = Dropout(dropout_embedding)
        right_output = gru(right_output) 
        right_output = BatchNormalization()(right_output)
            
        dist_normal = lambda x: 1 - K.abs(x[0] - x[1])
        merged = Lambda(function=dist_normal, output_shape=lambda x: x[0], 
                                       name='L1_distance')([left_output, right_output])
        if dropout_merged != None:
            merged = Dropout(dropout_merged)
        merged = BatchNormalization()(merged)

        predictions = Dense(1, activation='sigmoid', name='sentence_duplicate')(merged)
        model = Model([left_input, right_input], predictions)

        model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
        return model
    
    def deep_nn(self):
        question1 = Input(shape=(self.max_seq_length,))
        question2 = Input(shape=(self.max_seq_length,))

        q1 = Embedding(
                self.NUM_WORDS,
                self.embedding_dim,
                weights = [self.embedding_weights], 
                input_length = self.max_seq_length,
                trainable = False)(question1)
        q1 = TimeDistributed(Dense(self.embedding_dim, activation='relu'))(q1)
        q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.embedding_dim, ))(q1)

        q2 = Embedding(
                self.NUM_WORDS,
                self.embedding_dim,
                weights = [self.embedding_weights], 
                input_length = self.max_seq_length,
                trainable = False)(question2)
        q2 = TimeDistributed(Dense(self.embedding_dim, activation='relu'))(q2)
        q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(self.embedding_dim, ))(q2)

        merged = concatenate([q1,q2])
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(.1)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(.1)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(.1)(merged)
        merged = BatchNormalization()(merged)
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(.1)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[question1,question2], outputs=is_duplicate)
        model.compile(loss='binary_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
        return model