import pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

b=pd.read_csv("BostonHousing.csv")
X,y=b.iloc[:,:-1],b.iloc[:,-1]
m=LinearRegression().fit(X,y)
plt.scatter(y,m.predict(X)); plt.title("Linear Regression"); plt.show()

a=pd.read_csv("auto-mpg.csv")
a['horsepower']=pd.to_numeric(a['horsepower'],errors='coerce')
a=a.dropna()

X,y=a[['horsepower']],a['mpg']
p=PolynomialFeatures(4)
m=LinearRegression().fit(p.fit_transform(X),y)

plt.scatter(X,y)
plt.plot(X,m.predict(p.fit_transform(X)),'r.')
plt.title("Polynomial Regression")
plt.show()