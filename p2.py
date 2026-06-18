import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

df=pd.read_csv("California Housing.csv").select_dtypes("number")
sns.heatmap(df.corr(),annot=True,cmap="coolwarm"); plt.show()
sns.pairplot(df); plt.show()