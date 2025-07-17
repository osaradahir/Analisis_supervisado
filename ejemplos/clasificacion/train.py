import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    "tiempo_m": [5, 8, 12, 2, 20, 18, 30],
    "paginas_visitadas": [6, 4, 7, 3, 10, 9, 12],
    "compro": [1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
X = df[["tiempo_m", "paginas_visitadas"]]
y = df["compro"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Datos de prueba:", X_test)
print("Predicciones:", y_pred)
