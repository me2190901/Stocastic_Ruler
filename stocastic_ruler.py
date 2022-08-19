import numpy as np
np.random.seed(0)

# ---------------------Initialization-----------------
n=10
# Defining f(X)
f = [0.3, 0.7, 0.9, 0.5, 1.0, 1.4, 0.7, 0.8, 0.0, 0.6]

# Defining Discrete Set S
S = [i for i in range(1, n+1)]

# Defining parameters for Theta
a = -0.5
b = 1.9

# Defining Transition Probability Matrix(R)
R = np.random.rand(n, n)
R = R / R.sum(axis=1, keepdims=True)
print(R)

# Defining Neighbourhood Structure(N)
N=[[i for i in range(1, n+1) if i!=j] for j in range(1, n+1)]

# Defining X0
x0=np.random.randint(1,n+1)

# Initializing K
k=0
