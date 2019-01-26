#kategorik verileri numerik veri haline getirip olusan sonucu birlesirmek
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




