#maps several words into one common root
#outputs lematisations of a proper word
#gone going should be maped into go
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.stem import wordnet, WordNetLemmatizer
from nltk.corpus import stopwords
import re

reviews = pd.read_csv('reviews_cleanv2.csv')
reviews.info()
no_punctuation_regex = r'-.?!,:;()|0-9'
reviews['text'] = reviews.text.str.replace(no_punctuation_regex,'',regex=True)
reviews['title'] = reviews.title.str.replace(no_punctuation_regex,'',regex=True)

print(reviews.text)
punctuation = re.compile(r'-.?!,:;()|0-9')
def cleanStringFile(row):
    print(row)
    if(row.title is np.nan):
        return []

    text_tokens = word_tokenize(row.text) 
    text_tokens = [t for t in text_tokens if  t not in stopwords]
    # li = []
    # for i in range(len(counts[0])):
    #     li.append([row.id,row.key,counts[0][i],counts[1][i]])
    # return li
    print(text_tokens)


reviews = reviews.apply(cleanStringFile, axis=1)
print(stopwords.words('english'))

word_lem = WordNetLemmatizer()
print(wordnet)

