import pandas as pd
import numpy as np
import flet as ft


#importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report , mean_squared_error,r2_score,mean_absolute_error
import warnings



from sklearn.model_selection import train_test_split

df = pd.read_csv('./archive/IceCreamData.csv')

X = df[['Temperature']]
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

print(f'Total # of sample in whole dataset: {len(X)}')
print("*****"*10)
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Shape of X_train: {X_train.shape}')
print("*****"*10)
print(f'Total # of sample in test dataset: {len(X_test)}')
print(f'Shape of X_test: {X_test.shape}')





from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True)

model.fit(X_train, y_train)
# print(X_test,type(X_test))
pred = model.predict(X_test)
p = model.predict(np.array(float('40')).reshape((1,-1)))
print("ESTO SE VENDE COLEGA", p)


train_score = model.score(X_train, y_train)
print(f'Train score of trained model: {train_score*100}')

test_score = model.score(X_test, y_test)
print(f'Test score of trained model: {test_score*100}')





def getPredictions(temp,model):
    return f'Con {temp}° se predice ${round(model.predict(np.array(float(temp)).reshape((1,-1)))[0],3)}'


def main(page: ft.Page):
    def add_clicked(e):
        print(new_task.value,type(new_task.value))
        page.add(ft.Text(value=getPredictions(new_task.value, model),size=30, weight=ft.FontWeight.BOLD))
        new_task.value = ""
        new_task.focus()
        new_task.update()

    new_task = ft.TextField(hint_text="Introduzca la temperatura en un cierto día", width=400)
    page.add(ft.Row([new_task, ft.ElevatedButton("Predecir", on_click=add_clicked)]))

ft.app(target=main)

