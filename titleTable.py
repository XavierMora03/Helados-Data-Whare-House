import pandas as pd
import numpy as np
import itertools

reviews = pd.read_csv("reviews_clean.csv")

def rowTitle(row):
    if(row.title is np.nan):
        return []
    counts = np.unique(np.array(row.title.split()),return_counts=True)
    li = []
    for i in range(len(counts[0])):
        li.append([row.id,row.key,counts[0][i],counts[1][i]])
    return li

def rowText(row):
    if(row.text is np.nan):
        return []
    counts = np.unique(np.array(row.text.split()),return_counts=True)
    li = []
    for i in range(len(counts[0])):
        li.append([row.id,row.key,counts[0][i],counts[1][i]])
    return li

title = reviews.apply(rowTitle,axis=1)
text = reviews.apply(rowText,axis=1)

wordTitleTable = list(itertools.chain.from_iterable(title))
wordTextTable = list(itertools.chain.from_iterable(text))

textDf = pd.DataFrame(wordTextTable, columns=['id','key','word','count'])
titleDf = pd.DataFrame(wordTitleTable, columns=['id','key','word','count'])

textDf.to_csv('text_words.csv',index=False)
titleDf.to_csv('title_words.csv',index=False)