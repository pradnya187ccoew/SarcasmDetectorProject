#Word embeddings as features
#No scaling applied


from __future__ import print_function
#from keras.models import model_from_json
import os
import sys
import numpy as np
import codecs
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D,LSTM,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SpatialDropout1D
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

BASE_DIR = ''
FILENAME='wiki.hi-new.txt'
MAX_SEQUENCE_LENGTH = 100
SARC_MAX_NUM_WORDS = 15046
NONSARC_MAX_NUM_WORDS = 27339
TOTAL_MAX_NUM_WORDS = 35936 #SARC_MAX_NUM_WORDS + NONSARC_MAX_NUM_WORDS
EMBEDDING_DIM = 300 #100
VALIDATION_SPLIT = 0.2
global matrix_id
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = codecs.open('C:\\Users\\MyPC\\PycharmProjects\\SarcasmDetectorProject\\WordEmbedding\\wiki.hi-new.txt','r', encoding='utf-8')
#i=1
for line in f.readlines():
    #print ("line %d", i)
    values = line.strip().split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    #i=i+1
f.close()

print('Found %s word vectors.' % len(embeddings_index))

labels=[]
texts=[]
with codecs.open('C:\\Users\\MyPC\\PycharmProjects\\SarcasmDetectorProject\\Sarc_Data\\sarc_combined.txt', "r", encoding='utf-8') as sarcf:
    sarc_text=sarcf.readlines()
    for line in sarc_text:
        texts.append(line)
        labels.append(1)




with codecs.open('C:\\Users\\MyPC\\PycharmProjects\\SarcasmDetectorProject\\NonSarc_Data\\nonsarc_combined.txt', "r", encoding='utf-8') as nonsarcf:
    nonsarc_text=nonsarcf.readlines()
    for line in nonsarc_text:
        texts.append(line)
        labels.append(0)


tokenizer = Tokenizer(num_words=TOTAL_MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

final_word_index = tokenizer.word_index

print('Found %s total unique tokens.' % len(final_word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


print('Shape of complete data tensor:', data.shape)

labels = to_categorical(np.asarray(labels))
print('Shape of label tensor:', labels.shape)

# prepare embedding matrix
print('Preparing embedding matrix.')

sarc_num_words = min(TOTAL_MAX_NUM_WORDS, len(final_word_index))
embedding_matrix = np.zeros((TOTAL_MAX_NUM_WORDS, EMBEDDING_DIM))
for word, matrix_id in final_word_index.items():
    if matrix_id >= TOTAL_MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros.
        embedding_matrix[matrix_id] = embedding_vector
matrix_id=matrix_id+1

print ("Embedding Matrix")
print (embedding_matrix)
print (embedding_matrix.shape)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
#tot_num_words = sarc_num_words + nonsarc_num_words
print('Total number of words from both files: ', TOTAL_MAX_NUM_WORDS)

'''embedding_layer = Embedding(TOTAL_MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)'''

print('Training model.')

#divide data into traing &testing
#figure out hidden layer neurons (using cross validation)
#perform cross validation,train model, validation


import keras.backend as K

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def rec(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall = c1 / c3

    return recall


def prec(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision

lstm_out = 100 #number of neurons in the entire model
batch_size = 256


# fix random seed for reproducibility
np.random.seed(7)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


model = Sequential()
model.add(Embedding(TOTAL_MAX_NUM_WORDS, EMBEDDING_DIM,input_length = data.shape[1]))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dropout(0.35))
model.add(LSTM(100, stateful=False, dropout=0.25, recurrent_dropout=0.25))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dropout(0.35))
model.add(Dense(2,activation='softmax')) #here 2 becuase of label tensor shape
model.compile(loss = 'categorical_crossentropy', optimizer='adadelta',metrics = ['accuracy', prec, rec, f1_score])
print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

callbacks = [EarlyStopping(monitor='val_loss', mode='auto', patience=0, verbose=0)]

history=model.fit(X_train, Y_train, epochs = 12, batch_size=batch_size, validation_split=0.2, verbose = 2, callbacks=[callbacks])

import matplotlib.pyplot as plot
plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('model train vs validation loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.ylim(0,2)
plot.xlim(0,20)
plot.legend(['train', 'validation'], loc='upper right')
plot.show()
plot.savefig('lstm_mlm_set1_epoch12_adadelta.png')


#validation_size = int(VALIDATION_SPLIT * data.shape[0])#1500


score,acc,prec1, rec1, f1_score1 = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
print("prec: %.2f" % (prec1))
print("rec: %.2f" % (rec1))
print("f1-score: %.2f" % (f1_score1))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_test)):

    result = model.predict(X_test[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_test[x]):
        if np.argmax(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_mlm_combinedset_Epoch12_adadelta.h5")
print("Saved model to disk")

'''OUTPUT
Indexing word vectors.
Found 157789 word vectors.
Found 34613 total unique tokens.
Shape of complete data tensor: (15000, 100)
Shape of label tensor: (15000, 2)
Preparing embedding matrix.
Embedding Matrix
[[ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [-0.23162     0.12244    -0.37121999 ..., -0.16367     0.0086467  -0.041372  ]
 ..., 
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]]
(35936, 300)
Total number of words from both files:  35936
Training model.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 300)          10780800  
_________________________________________________________________
dropout_1 (Dropout)          (None, 100, 300)          0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               160400    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 202       
=================================================================
Total params: 10,941,402
Trainable params: 10,941,402
Non-trainable params: 0
_________________________________________________________________
None
(12000, 100) (12000, 2)
(3000, 100) (3000, 2)
Train on 9600 samples, validate on 2400 samples
Epoch 1/12
257s - loss: 0.6602 - acc: 0.5894 - prec: 0.5894 - rec: 0.5894 - f1_score: 0.5894 - val_loss: 0.6393 - val_acc: 0.5892 - val_prec: 0.5892 - val_rec: 0.5892 - val_f1_score: 0.5892
Epoch 2/12
245s - loss: 0.6026 - acc: 0.6805 - prec: 0.6805 - rec: 0.6805 - f1_score: 0.6805 - val_loss: 0.5676 - val_acc: 0.7442 - val_prec: 0.7442 - val_rec: 0.7442 - val_f1_score: 0.7442
Epoch 3/12
197s - loss: 0.4870 - acc: 0.8003 - prec: 0.8003 - rec: 0.8003 - f1_score: 0.8003 - val_loss: 0.3386 - val_acc: 0.8592 - val_prec: 0.8592 - val_rec: 0.8592 - val_f1_score: 0.8592
Epoch 4/12
181s - loss: 0.3128 - acc: 0.8776 - prec: 0.8776 - rec: 0.8776 - f1_score: 0.8776 - val_loss: 0.2528 - val_acc: 0.9012 - val_prec: 0.9012 - val_rec: 0.9012 - val_f1_score: 0.9012
Epoch 5/12
192s - loss: 0.2485 - acc: 0.9048 - prec: 0.9048 - rec: 0.9048 - f1_score: 0.9048 - val_loss: 0.2442 - val_acc: 0.9008 - val_prec: 0.9008 - val_rec: 0.9008 - val_f1_score: 0.9008
Epoch 6/12
207s - loss: 0.2059 - acc: 0.9266 - prec: 0.9266 - rec: 0.9266 - f1_score: 0.9266 - val_loss: 0.2226 - val_acc: 0.9100 - val_prec: 0.9100 - val_rec: 0.9100 - val_f1_score: 0.9100
Epoch 7/12
104s - loss: 0.1805 - acc: 0.9350 - prec: 0.9350 - rec: 0.9350 - f1_score: 0.9350 - val_loss: 0.2219 - val_acc: 0.9096 - val_prec: 0.9096 - val_rec: 0.9096 - val_f1_score: 0.9096
Epoch 8/12
91s - loss: 0.1526 - acc: 0.9444 - prec: 0.9444 - rec: 0.9444 - f1_score: 0.9444 - val_loss: 0.2212 - val_acc: 0.9125 - val_prec: 0.9125 - val_rec: 0.9125 - val_f1_score: 0.9125
Epoch 9/12
90s - loss: 0.1383 - acc: 0.9526 - prec: 0.9526 - rec: 0.9526 - f1_score: 0.9526 - val_loss: 0.2177 - val_acc: 0.9158 - val_prec: 0.9158 - val_rec: 0.9158 - val_f1_score: 0.9158
Epoch 10/12
110s - loss: 0.1204 - acc: 0.9592 - prec: 0.9592 - rec: 0.9592 - f1_score: 0.9592 - val_loss: 0.2172 - val_acc: 0.9192 - val_prec: 0.9192 - val_rec: 0.9192 - val_f1_score: 0.9192
Epoch 11/12
96s - loss: 0.1112 - acc: 0.9617 - prec: 0.9617 - rec: 0.9617 - f1_score: 0.9617 - val_loss: 0.2190 - val_acc: 0.9200 - val_prec: 0.9200 - val_rec: 0.9200 - val_f1_score: 0.9200
Epoch 12/12
94s - loss: 0.0983 - acc: 0.9680 - prec: 0.9680 - rec: 0.9680 - f1_score: 0.9680 - val_loss: 0.2194 - val_acc: 0.9175 - val_prec: 0.9175 - val_rec: 0.9175 - val_f1_score: 0.9175

(image of the fitting curve not included here)

score: 0.19
acc: 0.94
prec: 0.94
rec: 0.94
f1-score: 0.94
pos_acc 91.7948717948718 %
neg_acc 94.75409836065573 %
Saved model to disk

'''