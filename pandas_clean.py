import pandas as pd
import numpy as np
reviews = pd.read_csv("./archive/combined/reviews.csv",encoding='utf-8')
products = pd.read_csv("./archive/combined/products.csv")

reviews_text_labels = ['text','title']
products_text_labels = ['name','subhead','description','ingredients']

reviews.date = pd.to_datetime(reviews.date, format='%Y-%m-%d')
pd.set_option('display.max_colwidth', None)
#only words and spaces
def tolower(df,label):
        df[label] = df[label].str.lower()
        # df[label] = df[label].str.replace("[-_=/']",' ',regex=True)
        return df



def transformReviews(df,lables):
    for t in lables:
        df = tolower(df,t)
        # df[t] = df[t].str.replace("[^\w\s\']",'',regex=True)
        # df[t] = df[t].str.replace(r'([a-z])\1+', r'\1', regex=True)
        # df[t] = df[t].str.replace(r"['-_/\|’]",' ',regex=True)
        df[t] = df[t].str.replace("[^\w\s'’]",'',regex=True)
        df[t] = df[t].str.replace(r"[\\\\]",'',regex=True)
        # df[t] = df[t].str.replace(r"[-.?'!,:;()|0-9]",'',regex=True)

    
    return df


def transformProducts(df,lables):
    for t in lables:
        df = tolower(df,t)
        # df[t] = df[t].str.replace("[^\w\s,']",'',regex=True)
    
    return df

print(reviews.text[10:15])

reviews = transformReviews(reviews,reviews_text_labels)
reviews = tolower(reviews,'likes')
products = transformProducts(products,products_text_labels)

reviews.index.name = 'id'
reviews.reset_index(inplace=True)

print("\n\n\n\n",reviews.text[10:15].to_string())
products.to_csv('products_clean.csv',index=False)
reviews.to_csv('reviews_clean.csv',encoding='utf-8-sig',index=False)
