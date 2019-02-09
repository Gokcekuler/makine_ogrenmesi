import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
veriler = pd.read_csv('odul.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.ensemble import RandomForestRegressor
rassal_agac= RandomForestRegressor(n_estimators=10, random_state=0)
rassal_agac.fit(x.values,y.values)
plt.scatter(x.values,y.values, color= 'red')
plt.plot(x,rassal_agac.predict(x.values), color='blue')
plt.show()
