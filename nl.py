#maps several words into one common root
#outputs lematisations of a proper word
#gone going should be maped into go
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.stem import wordnet, WordNetLemmatizer
from nltk.corpus import stopwords, words
import re
import itertools

stem = WordNetLemmatizer()
reviews = pd.read_csv('reviews_clean.csv')

reviews = reviews[:5]
# no_punctuation_regex = r'[-.?!,:;()|0-9]'

reviews.info()
punctuation = re.compile(r'-.?!,:;()|0-9')

# stopspanish = stopwords.words('spanish')

stop_words = set(stopwords.words('english'))
print(stop_words,'\n\n\n')
# stop_words.add("'")
# stop_words.add("’")
def convert_word(word):
    if word in words.words():
        return word
    else:
        return re.sub(r'([a-z])\1+', r'\1', word)
    
def cleanStringFile(row):
    if(row.text is np.nan):
        return []

    print(row.text)
    text_tokens = word_tokenize(row.text) 
    text_tokens = [t for t in text_tokens if  t not in stop_words and len(t) > 1]
    text_tokens = [stem.lemmatize(t) for t in text_tokens]
    print(text_tokens)
    #takes too long
    text_tokens = [convert_word(t) for t in text_tokens]

    fdist = FreqDist()
    for t in text_tokens:
        fdist[t] +=1

    li = []
    print(row.id)
    for item,count in fdist.items():
        li.append([row.id,row.key,item,count,row.stars])
        # print(item,count)

    return li


list_train_data = list(itertools.chain.from_iterable(reviews))
reviews_transformed = pd.DataFrame(reviews, columns = ['id','key','word','count','score'])
reviews = reviews.apply(cleanStringFile, axis=1)

word_lem = WordNetLemmatizer()
print(wordnet)

