#kategorik verilerin numerik verilere donusturulmesi

import pandas as pd
veriler=pd.read_csv("veri.csv")

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder #LabelEncoder her deger için sayisal deger vermemizi saglar
encoder=LabelEncoder() 
ulke[:,0]= encoder.fit_transform(ulke[:,0]) 
print(ulke)

from sklearn.preprocessing import OneHotEncoder    #burada numara verme islemini ilgilenilen deger icin 1 diger degerler icin 0 alarak yapiyorum. 
onehotencoder= OneHotEncoder(categorical_features='all')
ulke=onehotencoder.fit_transform(ulke).toarray() 
print(ulke)



