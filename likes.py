import pandas as pd
import numpy as np
import itertools

products = pd.read_csv("./reviews_clean.csv")
products = products.dropna(axis=0,subset=['likes'])
products.likes = products.likes.astype(str)
print(products.head())
def get_ingredients(row):
    if row.likes == np.nan:
        return []
    li = [] 
    for ingredient in row.likes.split(','):
        li.append([row.key,ingredient.strip().replace('"','')])
    return li

ingredients = products.apply(get_ingredients, axis=1)
ingredients = list(itertools.chain.from_iterable(ingredients))

ingredients = pd.DataFrame(ingredients, columns=['key','likes'])
ingredients.to_csv('./likes.csv', index=False)
