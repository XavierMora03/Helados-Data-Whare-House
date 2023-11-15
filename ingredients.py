import pandas as pd
import itertools

products = pd.read_csv("./products_clean.csv")

def get_ingredients(row):
    """
    returns a list of the ingredients of a product
    """
    li = [] 
    for ingredient in row.ingredients.split(','):
        li.append([row.key,ingredient.strip().replace('"','')])
    return li

ingredients = products.apply(get_ingredients, axis=1)
ingredients = list(itertools.chain.from_iterable(ingredients))

ingredients = pd.DataFrame(ingredients, columns=['key','ingredient'])
ingredients.to_csv('./ingredients.csv', index=False)
