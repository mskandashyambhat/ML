import pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df=pd.read_csv("breast_cancer_data.csv")

X=StandardScaler().fit_transform(df.iloc[:,:-1])

c=KMeans(2,n_init=10).fit_predict(X)

p=PCA(2).fit_transform(X)

plt.scatter(p[:,0],p[:,1],c=c,cmap='viridis')
plt.title("K-Means Clustering")
plt.show()