import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset  = pd.read_csv("LR.csv")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

accuracy = regressor.score(x,y)*100
print(accuracy)

y_pred = regressor.predict([[10]])
print(y_pred)

hours = int(input("Enter hours"))
eq = regressor.coef_*hours+regressor.intercept_
print(eq[0])

plt.plot(x,y,'o')
plt.plot(x,regressor.predict(x))
plt.show()