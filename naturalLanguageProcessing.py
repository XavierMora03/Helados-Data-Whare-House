#maps several words into one common root
#outputs lematisations of a proper word
#gone going should be maped into go
import pandas as pd
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
# no_punctuation_regex = r'[-.?!,:;()|0-9]'
# punctuation = re.compile(r"[-.?!,:;'()|0-9]")
punctuation = re.compile(r"['’]")
stop_words = set(stopwords.words('english'))

mystop_words = set([punctuation.sub('',w) for w in stop_words ])
stop_words = stop_words.union(mystop_words)

def convert_word(word):
    if word in words.words():
        return word
    else:
        return re.sub(r'([a-z])\1+', r'\1', word)
    

def cleanStringFile(row):
    if(row.text is np.nan):
        return []

    text_tokens = word_tokenize(row.text, language='english') 
    text_tokens = [t for t in text_tokens if  t not in stop_words and len(t) > 2 and  not t.__contains__("'")]
    text_tokens = [stem.lemmatize(t,'v') for t in text_tokens]
    fdist = FreqDist()

    for t in text_tokens:
        fdist[t] +=1

    li = []
    for item,count in fdist.items():
        new_row = [row.id,row.key,item,count,row.stars]
        li.append(new_row)

    return li


list_train_data = reviews.apply(cleanStringFile, axis=1)
list_train_data = list(itertools.chain.from_iterable(list_train_data))
reviews_transformed = pd.DataFrame(list_train_data, columns = ['id','key','word','count','score'])
reviews_transformed.to_csv('reviews_transformed.csv',index=False)

