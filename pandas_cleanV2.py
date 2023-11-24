import pandas as pd
import numpy as np
reviews = pd.read_csv("./archive/combined/reviews.csv",encoding='utf-8')
products = pd.read_csv("./archive/combined/products.csv")

reviews_text_labels = ['text','title']
products_text_labels = ['name','subhead','description','ingredients']

reviews.date = pd.to_datetime(reviews.date, format='%Y-%m-%d')

#only words and spaces

def cleanAfterTokenize(df, labels):
    for l in labels:
        df[l] = df[l].str.lower()
        df[l] = df[l].str.replace(r'[-.?!’:;|0-9]',' ',regex=True)
    return df



reviews = cleanAfterTokenize(reviews,reviews_text_labels)
products = cleanAfterTokenize(products,products_text_labels)

reviews.index.name = 'id'
reviews.reset_index(inplace=True)

products.to_csv('products_cleanv2.csv',index=False)
reviews.to_csv('reviews_cleanv2.csv',encoding='utf-8-sig',index=False)
