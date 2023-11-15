import pandas as pd
import numpy as np
reviews = pd.read_csv("./archive/combined/reviews.csv")
text_labes = ['text','title']

reviews.date = pd.to_datetime(reviews.date, format='%Y-%m-%d')
for t in text_labes:
    reviews[t] = reviews[t].str.lower()
    reviews[t] = reviews[t].str.replace('[-_]',' ',regex=True)
    reviews[t] = reviews[t].str.replace('[^\w\s]','',regex=True)

reviews.to_csv('reviews_clean.csv',index=False)
