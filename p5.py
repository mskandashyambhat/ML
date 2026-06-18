import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x=np.random.rand(100)
X,y=x[:50].reshape(-1,1),['Class1' if i<=0.5 else 'Class2' for i in x[:50]]
t=x[50:].reshape(-1,1)

for k in [1,2,3,4,5,20,30]:
    p=KNeighborsClassifier(k).fit(X,y).predict(t)
    print("k =",k,"\n",p,"\n")