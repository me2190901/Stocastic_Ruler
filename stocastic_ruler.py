import numpy as np
import math

np.random.seed(1234)

def define_R(size):
    R = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i == j):
                R[i, j] = 0
            else:
                R[i, j] = 1/(size-1)
    return R


def define_N():
    return [[i for i in range(1, n+1) if i != j] for j in range(1, n+1)]

def Mk(k):
    return math.floor(math.log(k+10,5))


# ---------------------Initialization-----------------
n = 10
# Defining f(X)
f = [0.3, 0.7, 0.9, 0.5, 1.0, 1.4, 0.7, 0.8, 0.0, 0.6]

# Defining Discrete Set S
S = [i for i in range(1, n+1)]

# Defining parameters for Theta
a = -0.5
b = 1.9

# Defining Transition Probability Matrix(R)
R = define_R(n)

# Defining Neighbourhood Structure(N)
N = define_N()

# Defining X0
x0 = np.random.randint(1, n+1)

# Initializing K
k = 0

# Limit on number of iterations for termination
limit_k=100

#---------------Implementing Stochastic Ruler Algorithm------------

def step_1(x):
    Nx= N[x-1]
    P= [R[x-1][i-1] for i in Nx]
    z=np.random.choice(Nx, p=P)
    return z

# def step_2(x):


def step_3():
    global k
    k = k + 1