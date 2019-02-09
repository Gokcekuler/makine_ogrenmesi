import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


veriler = pd.read_csv('odul.csv')


x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X=scx.fit_transform(x.values)
scy=StandardScaler()
Y=scy.fit_transform(y.values)
print (X)
print(Y)

from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,svr.predict(X),color='blue')

plt.show()

