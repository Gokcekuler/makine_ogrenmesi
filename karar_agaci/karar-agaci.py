import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


veriler = pd.read_csv('odul.csv')


x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(x.values,y.values)

plt.scatter(x.values,y.values, color='red')
plt.plot(x,r_dt.predict(x.values), color='blue')
plt.show()


