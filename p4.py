import pandas as pd

d=pd.read_csv("tennis.csv")
h=['0']*(d.shape[1]-1)

for r in d.values:
    if r[-1]=='Yes':
        for i in range(len(h)):
            if h[i]=='0': h[i]=r[i]
            elif h[i]!=r[i]: h[i]='?'

print(h)