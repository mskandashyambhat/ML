import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

df=pd.read_csv("California Housing.csv")
df.hist(figsize=(15,10)); plt.show()

for c in df.select_dtypes("number"):
    sns.boxplot(x=df[c]); plt.show()
    q1,q3=df[c].quantile([.25,.75]); i=q3-q1
    print(c,len(df[(df[c]<q1-1.5*i)|(df[c]>q3+1.5*i)]))