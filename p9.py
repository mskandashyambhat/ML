import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv("olivetti_faces.csv")

Xtr,Xte,ytr,yte=train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=.3,random_state=42)

m=GaussianNB().fit(Xtr,ytr)

print("Accuracy:",m.score(Xte,yte))

for i in [0,5,10]:
    p=m.predict([Xte.iloc[i]])[0]
    print("Predicted:",p,"Actual:",yte.iloc[i])
    plt.imshow(Xte.iloc[i].values.reshape(64,64),cmap="gray")
    plt.show()