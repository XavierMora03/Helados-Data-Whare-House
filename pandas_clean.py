import pandas as pd
import numpy as np
reviews = pd.read_csv("./archive/combined/reviews.csv")
text_labes = ['text','title']

reviews.date = pd.to_datetime(reviews.date, format='%Y-%m-%d')
for t in text_labes:
    reviews[t] = reviews[t].str.lower()
    reviews[t] = reviews[t].str.replace('[-_]',' ',regex=True)
    reviews[t] = reviews[t].str.replace('[^\w\s]','',regex=True)

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



print(reviews.date.describe())
def count_words(string):
    a = np.unique(string.split(),return_counts=True)


reviews.apply(see_row,axis=1)
print(wordDict.items())

worddf = pd.DataFrame(wordDict.keys(),columns=['word'],index=wordDict.values())
worddf.index.name = 'id'
worddf.to_csv('words.csv')

