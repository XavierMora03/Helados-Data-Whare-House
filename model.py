import pandas as pd
import numpy as np


reviews = pd.read_csv('./ingredientes_words_per_review.csv',encoding='utf-8')

labels = ['words','ingredients']

for l in labels:
    reviews[l] = reviews[l].map(lambda x:  np.array(x.split(','),dtype='S'))

# reviews = reviews.dropna(how='any',axis=0)

print(reviews.loc[0])
