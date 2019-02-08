import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('odul.csv')
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x.values)
print(x_poly)
lr=LinearRegression()
lr.fit(x_poly,y)
plt.scatter(x.values,y.values, color='red')
plt.plot(x.values,lr.predict(poly.fit_transform(x.values)), color='blue')
plt.show()

