#! /usr/bin/env python3.1
#encoding=UTF8

''' Lightweight Hindi stemmer
Copyright Â© 2010 LuÃ­s Gomes <luismsgomes@gmail.com>.

Implementation of algorithm described in

    A Lightweight Stemmer for Hindi
    Ananthakrishnan Ramanathan and Durgesh D Rao
    http://computing.open.ac.uk/Sites/EACLSouthAsia/Papers/p6-Ramanathan.pdf

    @conference{ramanathan2003lightweight,
      title={{A lightweight stemmer for Hindi}},
      author={Ramanathan, A. and Rao, D.},
      booktitle={Workshop on Computational Linguistics for South-Asian Languages, EACL},
      year={2003}
    }

Ported from HindiStemmer.java, part of of Lucene.
'''
import sys
import codecs
#1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"]
suffixes = {

    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
}

def hi_stem(word):
    for L in 5, 4, 3, 2:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                if word.endswith(suf):
                    if(suf=="ुओं"):
                        word=word[:-L]+"ु"+'|'+word[-L:]
                    elif(word=="टीवी"):
                        word=word
                    else:
                        word=word[:-L]+'|'+word[-L:]
                    return word
                    #return word[:-L]
    return word+'|'+ 'null'
    #return word

if __name__ == '__main__':

    if len(sys.argv) != 1:
        sys.exit('{} takes no arguments'.format(sys.argv[0]))
    f1 = codecs.open("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Data_files\\sarcastic_stemmed_output.txt", "w+", "utf-8")
    with codecs.open("C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Data_files\\sarcastic_stopwords_removed_output.txt", "r","utf-8") as sourceFile:
        lines=sourceFile.readlines()
        str=""
        for line in lines:
            for word in line.split():
                word=hi_stem(word)
                str=str+ word + " "
            f1.write(str+'\n')
