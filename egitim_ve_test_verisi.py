#eksikveri.csv dosyasındaki kategorik verileri numerik veriye donusturduk ve bunları birlestirdik
import pandas as pd
veriler=pd.read_csv("eksikveri.csv")

meyve=veriler.iloc[:,0:1].values
print(meyve)

miktar= veriler.iloc[:,1:3].values
print(miktar)

tur= veriler.iloc[:,-1].values
print(tur)


from sklearn.preprocessing import OneHotEncoder
onehotencoder= OneHotEncoder(categorical_features='all')
Meyve=onehotencoder.fit_transform(meyve).toarray()
print(Meyve)

sonuc=pd.DataFrame(data=Meyve, index=range(12), columns=['erik', 'kiraz', 'muz'])
print(sonuc)

sonuc2= pd.DataFrame(data=miktar, index=range(12), columns=['kilo', 'fiyat'])
print(sonuc2)

sonuc3= pd.DataFrame(data=tur, index=range(12), columns=['tur'])
print(sonuc3)




s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verileri egitim ve test verisi olarak bolduk

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(s,sonuc3,test_size=0.33,random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)




#verilere standartlastirma uyguladik

from sklearn.preprocessing import StandardScaler


standart=StandardScaler()

X_train=standart.fit_transform(x_train)
X_test=standart.fit_transform(x_test)

print(X_train)
print(X_test)


