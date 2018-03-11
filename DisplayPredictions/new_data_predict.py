import codecs
import numpy as np
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

TOTAL_MAX_NUM_WORDS = 34613 #SARC_MAX_NUM_WORDS + NONSARC_MAX_NUM_WORDS
MAX_SEQUENCE_LENGTH= 100

texts=[]

stopwords=[]
with codecs.open("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Preprocessing\\stopwords_hindi.txt", "r","utf-8") as sourceFile:
    lines=sourceFile.readlines()
    for line in lines:
        words = line.split()
        for word in words:
            if word not in stopwords:
                stopwords.append(word)
sourceFile.close()

str=''
with codecs.open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\DisplayPredictions\\new_user_data', "r", encoding='utf-8') as new:
    new_text=new.readlines()
    for line in new_text:
        words = line.split()
        for word in words:
            if word not in stopwords:
                str+= word + " "

#stopwords removed



tokenizer = Tokenizer(num_words=34613)
tokenizer.fit_on_texts(str)
sequences = tokenizer.texts_to_sequences(str)

new_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# load json and create model
json_file = open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\WordEmoji\\model_lstm_emoji_word.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\WordEmoji\\model_lstm_emoji_word.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta')


#only for first sentence in the file
result = loaded_model.predict(np.array(new_data), batch_size=1)[0]

print (result)
'''print("Non-sarcasm score:",result[0])
print("Sarcasm score: ",result[1])
'''


if np.argmax(result)==0:
    print("Non-Sarcastic")
else:
    print("Sarcastic")



'''
इस देश की स्तिथि सुधारनी होगी sarcastic 
मौत सजा लिए क्या फांसी अलावा कोई विकल्प सकता जिसमें पीड़ादायक मृत्यु न सुप्रीम कोर्ट इस मामले आज एक जनहित याचिका सुनवाई non-sarcastic
पढ़ेगा देश तभी तोह बढेगए देश non-sarcastic
आज दिन अच्छा है  predicted as non-sarcastic
मुझे यह पुस्तक अच्छी लगी sarcastic
भारत को ३६ और राफेल जंगी विमान बेचकर पॉवरफुल बनाना चाहता है फ्रांस sarcastic
हमने महिअलों के लिए बियर बनायीं है पिंक बियर कारण उनको सब पिंक और चमकीला पसंद है न sarcastic
'''
