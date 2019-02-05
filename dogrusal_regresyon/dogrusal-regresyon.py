import pandas as pd
import matplotlib.pyplot as plt
veriler=pd.read_csv('hava-durumu.csv')
aylar=veriler[['gun']]
print(aylar)
sicaklik=veriler[['sicaklik']]
print(sicaklik)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,sicaklik,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

print(tahmin)

print(y_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.show()

















