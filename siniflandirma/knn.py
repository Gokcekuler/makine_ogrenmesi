import pandas as pd

veriler=pd.read_csv('veri.csv')
x=veriler.iloc[:,1:3].values
y=veriler.iloc[:,3:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
standart=StandardScaler()
X_train=standart.fit_transform(x_train)
X_test=standart.transform(x_test)

#En yakin komsu yontemiyle siniflama
from sklearn.neighbors import KNeighborsClassifier
k=KNeighborsClassifier(n_neighbors=1, metric='minkowski')
k.fit(X_train,y_train)
y_pred=k.predict(X_test)

#karmasiklik matrisi
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


