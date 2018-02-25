from __future__ import print_function
from polyglot.downloader import downloader
import os
import sys
import numpy as np
import codecs
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D,LSTM,BatchNormalization,merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import SpatialDropout1D
from keras.models import Model
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from polyglot.text import Text


BASE_DIR = ''
FILENAME='wiki.hi-new.txt'
MAX_SEQUENCE_LENGTH = 100
#SARC_MAX_NUM_WORDS = 15046
#NONSARC_MAX_NUM_WORDS = 27339
TOTAL_MAX_NUM_WORDS = 34613 #SARC_MAX_NUM_WORDS + NONSARC_MAX_NUM_WORDS
EMBEDDING_DIM = 300 #100
VALIDATION_SPLIT = 0.2
global matrix_id
global result
# first, build index mapping words in the embeddings set
# to their embedding vector


print('Indexing word vectors.')

embeddings_index = {}
f = codecs.open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Word_Embeddings\\wiki.hi-new.txt','r', encoding='utf-8')
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

with codecs.open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Data_files\\sarc_combined.txt', "r", encoding='utf-8') as sarcf:
    sarc_text=sarcf.readlines()
    for line in sarc_text:
        texts.append(line)
        labels.append(1)





with codecs.open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Data_files\\nonsarc_combined.txt', "r", encoding='utf-8') as nonsarcf:
    nonsarc_text=nonsarcf.readlines()

    for line in nonsarc_text:
        texts.append(line)
        labels.append(0)



#text=Text("काम करो फल चिंता मत करो गनेश")


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
sentiment_texts = np.zeros(TOTAL_MAX_NUM_WORDS)

for word, matrix_id in final_word_index.items():
    if matrix_id >= TOTAL_MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros.
        embedding_matrix[matrix_id] = embedding_vector
    text=Text(word)
    for words in text.words:
        try:
            sentiment_texts[matrix_id]=words.polarity
        except ValueError:
            sentiment_texts[matrix_id]=0
matrix_id=matrix_id+1

print ("Embedding Matrix")
#print (embedding_matrix)
print ("Embeddings Matrix shape ",embedding_matrix.shape)


print('Shape of complete sentiment_features tensor:', sentiment_texts.shape)


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

#Preparing Data for training, testing and validation

# fix random seed for reproducibility
np.random.seed(7)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

#splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size = 0.2, random_state = 42)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


embedding_layer = Embedding(TOTAL_MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedded_sequences = embedding_layer(sequence_input)
x=Dropout(0.35)(embedded_sequences)
x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x=LSTM(200, stateful=False )(x)
seq_features=Dropout(0.35)(x)


# Adding hand-picked features
other_features = Input(shape=(MAX_SEQUENCE_LENGTH,))

model_final = merge([seq_features , other_features], mode='concat')

model_final = Dense(2, activation='softmax')(model_final)

model_final = Model([sequence_input, other_features], model_final)

model_final.compile(loss = 'categorical_crossentropy', optimizer='adadelta',metrics = ['accuracy', prec, rec, f1_score])
print(model_final.summary())



early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath="model_lstm_sentiment.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#checkpoint = ModelCheckpoint(filepath="model_lstm_sentiment.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history=model_final.fit(X_train, Y_train, epochs = 50, batch_size=batch_size, validation_split=0.2, verbose = 2, callbacks=[early_stop,checkpoint])


#plotting the fitting curve during the training process


import matplotlib.pyplot as plot
plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('model train vs validation loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.ylim(0,2)
plot.xlim(0,100)
plot.legend(['train', 'validation'], loc='upper right')
plot.savefig('lstm_final.png')
plot.show()



#running the trained model on test sets, evaluating the performance metrics

score,acc,prec1, rec1, f1_score1 = model_final.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("acc: %.2f" % (acc))


#printing the confusion matrix
#pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
#for x in range(len(X_test)):

# result = model.predict(X_test[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

#print(confusion_matrix(Y_test,result))
# Confusion matrix result

from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model_final.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)


cm = confusion_matrix(np.argmax(Y_test,axis=1),y_pred)
print(cm)

'''
    if np.argmax(result) == np.argmax(Y_test[x]):
        if np.argmax(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1
'''

#print("pos_acc", pos_correct / pos_cnt * 100, "%")
#print("neg_acc", neg_correct / neg_cnt * 100, "%")

# serialize model to JSON
model_json = model_final.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("model_lstm_sentiment.h5")
print("Saved model to disk")

