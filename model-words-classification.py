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

from sklearn.svm import LinearSVC

df = pd.read_csv('./ingredientes_words_per_review.csv',encoding='utf-8')

labels = ['words','ingredients']

for l in labels:
    # df[l] = df[l].map(lambda x:  np.array(x.split(','),dtype='S'))
    df[l] = df[l].astype(str)


#words
Tfid = TfidfVectorizer(min_df = 1,stop_words='english')
cv = CountVectorizer()
#stars
scale = MinMaxScaler()


#the columns transofrmer does not work rigth now, I dont know why
words = Tfid.fit_transform(np.array(df['words']))
ingredientes = cv.fit_transform(df['ingredients'])
# stars = np.array(scale.fit_transform(df[['stars']]),dtype='i')
stars = np.array(df.stars,dtype='i')
print(np.info(stars),np.info(words))



preprocessing = ColumnTransformer([
    ("ingredients", cv, ['ingredients']),
    ("words", Tfid, ['words']),
    ])
    # ("stars",MinMaxScaler(), ['stars'])
    # ])


# df_x = df[['ingredients','stars']]
# df_y = df['words']


type(stars)
x_train, x_test, y_train, y_test = train_test_split(words,stars, test_size = 0.2, random_state=4)

print(x_train,y_train)


# x_trainCv = cv.fit_transform(x_train['ingredients'])
# print(cv.get_feature_names_out())
# mnb = LinearSVC()
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predictions = mnb.predict(x_test)
score = average_precision_score(y_test,predictions)
print(score)
# print(x_trainCv, x_trainCv.toarray())


