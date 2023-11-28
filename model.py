import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import average_precision_score

df = pd.read_csv('./ingredientes_words_per_review.csv',encoding='utf-8')

labels = ['words','ingredients']

for l in labels:
    # df[l] = df[l].map(lambda x:  np.array(x.split(','),dtype='S'))
    df[l] = df[l].astype('S')


#words
Tfid = TfidfVectorizer(min_df = 1,stop_words='english')
cv = CountVectorizer()
#stars
scale = MinMaxScaler()

df.ingredients = cv.fit_transform(df.ingredients)
df.words = Tfid.fit_transform(df.words)

df.stars = scale.fit_transform(df[['stars']])

print(df.stars)

preprocessing = ColumnTransformer([
    ("ingredients", make_pipeline(cv), ['ingredients']),
    ("words", make_pipeline(Tfid), ['words']),
    ])
    # ("stars",MinMaxScaler(), ['stars'])
    # ])

dd = preprocessing.fit_transform(df)
df_x = df[['ingredients','stars']]
df_y = df['words']


x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size = 0.2, random_state=4)

# x_trainCv = cv.fit_transform(x_train['ingredients'])
print(cv.get_feature_names_out())
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predictions = mnb.predict(x_test)
score = average_precision_score(y_test,predictions)
print(score)
# print(x_trainCv, x_trainCv.toarray())
quit()

print(df.loc[0],type(df[0][0]))
encoder = OneHotEncoder()

x = encoder.fit(np.array(df.words).reshape(1,-1))
print(encoder.categories_)
# print(df.loc[0],type(df[0][0][0]))
