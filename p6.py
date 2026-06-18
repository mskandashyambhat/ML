import numpy as np, matplotlib.pyplot as plt

x=np.linspace(0,1,100)
y=2*x**2+0.3*np.random.randn(100)

for t in [0.1,0.3,0.5]:
    yp=[np.sum(np.exp(-(x-i)**2/(2*t*t))*y)/np.sum(np.exp(-(x-i)**2/(2*t*t))) for i in x]
    plt.plot(x,yp,label=f"τ={t}")

plt.scatter(x,y)
plt.legend()
plt.show()