import os
import codecs
from builtins import str



stopwords=[]
with codecs.open("stopwords_hindi.txt", "r","utf-8") as sourceFile:
    stopwords=sourceFile.readlines()
sourceFile.close()

stopwords=[]
with codecs.open("stopwords_hindi.txt", "r","utf-8") as sourceFile:
    lines=sourceFile.readlines()
    for line in lines:
        words = line.split()
        for word in words:
            if word not in stopwords:
                stopwords.append(word)
sourceFile.close()

f1 = codecs.open("Data_files/sarcastic_stopwords_removed_output.txt", "w+", "utf-8")
with codecs.open("Data_files/sarcastic_normalized_output.txt", "r","utf-8") as sourceFile:
    lines=sourceFile.readlines()
    for line in lines:
        str=''
        words = line.split()
        for word in words:
            if word not in stopwords:
                str+= word + " "
            else:
                print (word)
        f1.write(str+'\n')
sourceFile.close()
f1.close()