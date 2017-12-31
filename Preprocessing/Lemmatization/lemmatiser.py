#!/usr/bin/env python
'''
Usage: python lemmatiser.pl lemmaFile < input > output

The input should have two colums
<html>
word1	tag1
word2	tag2
.
'''

import sys
import re
import codecs, string

global lemmaDict
lemmaDict= {}  # word : { pos : lemma}

def loadLemmatiser(file):
    for line in codecs.open(file, 'r', 'utf-8'):
        line=line.strip().split('\t')
        word=line[0]
        lemma=line[1:2]
        if word not in lemmaDict:
            lemmaDict[word]= {}
        if lemma!="":
            lemmaDict[word]=str(lemma)
    #print (lemmaDict)
'''
def lemmatise(f):
    for line in f:
        line= line.strip()
        if line=="":
            print (line)
        elif line[0]=='<':
            print (line)
        else:
            #line.sub("\t+")
            cols= line.split()
            if len(cols)!=2:
                #print cols
                print (line)
            else:
                if cols[0] in lemmaDict and cols[1] in lemmaDict[cols[0]]:
                    print ("%s\t%s" %(line, lemmaDict[cols[0]][cols[1]]))
                else:
                    print ("%s\t%s" %(line, cols[0]+"."))
'''


def lemmatise(l,f):
    for line in f:
        line=line.strip()
        line=line.split()
        str=""
        for word in line:
            if word in l:
                word=l[word]
            str=str+word+" "
        f1.write(str+'\n')

f1=codecs.open("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Preprocessing\\Lemmatization\\sarcastic_lemmatized_output.txt", "w", "utf-8")
loadLemmatiser("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Preprocessing\\Lemmatization\\hindi.lemma")
with codecs.open("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Preprocessing\\Lemmatization\\sarcastic_stopwords_removed_output.txt", "r","utf-8") as sourceFile:
    lines=sourceFile.readlines()
lemmatise(lemmaDict,lines)
