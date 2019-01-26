#eksik olan verilerin ortalama deger ile doldurulmasi
import pandas as pd

veriler=pd.read_csv("eksikveri.csv")
print(veriler)


from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values='NaN', strategy= 'mean', axis=0 ) 

eksik= veriler.iloc[:,1:3].values 
print(eksik)
imputer= imputer.fit(eksik[:,1:3]) 
eksik[:,1:4] = imputer.transform(eksik[:,1:4]) 
print(eksik)



