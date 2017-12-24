from builtins import str
import codecs
import os
import json
import codecs

total_tweets = 0
total_tweets_words = 0

#filename="Data_files/sarcastic_stopwords_removed_output.txt"
with codecs.open('C:\\Users\\hp-pc\\PycharmProjects\\SarcasmProject\\Data_files\\sarcastic_stopwords_removed_output.txt', 'r','utf-8') as f:
    content= f.readlines()

    for line in content:
        total_tweets+=1
        total_tweets_words += len(line.split())

average_len_tweet = total_tweets_words/(total_tweets*1.0)

#filename_output=""
#with open(filename_output,"w+") as f:
#dumpdata = {'TotalTweet':total_tweets,'AverageTweetLength':average_len_tweet}
#json.dump(dumpdata)

print (total_tweets)
print (average_len_tweet)