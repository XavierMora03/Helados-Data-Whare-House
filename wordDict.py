import pandas as pd
import numpy as np
reviews = pd.read_csv("reviews_clean.csv")

wordDict_index = 0
wordDict = {}

def update_dict(string):
    if string is np.nan:
        return
    global wordDict_index
    global wordDict
    for word in string.split():
        if word not in wordDict:
            wordDict[word] = wordDict_index
            wordDict_index += 1


def see_row(row):
    update_dict(row.text)
    update_dict(row.title)

reviews.apply(see_row,axis=1)

worddf = pd.DataFrame(wordDict.keys(),columns=['word'],index=wordDict.values())
worddf.index.name = 'id'
worddf.to_csv('words.csv')

