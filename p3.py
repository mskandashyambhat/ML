import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("iris_data.csv")
X=StandardScaler().fit_transform(df.iloc[:,:4])
pca=PCA(n_components=2).fit_transform(X)

sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=df["Species"])
plt.show()