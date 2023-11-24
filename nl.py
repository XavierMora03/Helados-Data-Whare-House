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
from nltk.corpus import stopwords
import re


stem = WordNetLemmatizer()
reviews = pd.read_csv('reviews_clean.csv')
no_punctuation_regex = r'[-.?!,:;()|0-9]'
reviews['text'] = reviews.text.str.replace(no_punctuation_regex,'',regex=True)
reviews['title'] = reviews.title.str.replace(no_punctuation_regex,'',regex=True)

punctuation = re.compile(r'-.?!,:;()|0-9')
'''
var = 'arremanguala arrepuajala si no se jaja ajaj okas arre nono si no claro miamor, no tengo miamor'
print(stopwords.words('english'))
tokenizado = word_tokenize(var)
fdist = FreqDist()
stopspanish = stopwords.words('spanish')
for t in tokenizado:
    if(t not in stopspanish):
        fdist[t] += 1
print(fdist.N,"HOLACOMOESTA             N",fdist.B)
'''
stop_words = set(stopwords.words('english'))
print('dont' in stop_words)
# print(stop_words)
def cleanStringFile(row):
    if(row.text is np.nan):
        return []

    print(row.text)
    text_tokens = word_tokenize(row.text) 
    text_tokens = [t for t in text_tokens if  t not in stop_words]
    text_tokens = [stem.lemmatize(t) for t in text_tokens]
    # li = []
    # for i in range(len(counts[0])):
    #     li.append([row.id,row.key,counts[0][i],counts[1][i]])
    # return li
    print(text_tokens)
    quit()


reviews = reviews.apply(cleanStringFile, axis=1)
print(stopwords.words('english'))

word_lem = WordNetLemmatizer()
print(wordnet)

