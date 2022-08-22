import numpy as np
import math
import time

np.random.seed(1234)

# ---------------------Initialization-----------------


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
    return math.floor(math.log(k+10, 5))

def f(x):
    f_vals = [0.3, 0.7, 0.9, 0.5, 1.0, 1.4,
              0.7, 0.8, 0.0, 0.6]  # 1 based indexing
    return f_vals[x-1]

def find_x0():
    x0 = np.random.randint(1, n+1)
    return x0

def get_h_val(f_x):
    temp = np.random.uniform(low=f_x - 0.5, high=f_x +
                             0.5, size=1)  # [low,high)
    return temp[0]

def get_theta_val(a, b):
    temp = np.random.uniform(low=a, high=b, size=1)  # [low,high)
    return temp[0]

n = 10

# Defining Discrete Set S
S = [i for i in range(1, n+1)]

# Defining parameters for Theta
a = -0.5
b = 1.9

# Defining Transition Probability Matrix(R)
R = define_R(n)

# Defining Neighbourhood Structure(N)
N = define_N()

# Limiting number of times x can be repeated
limit_unchanged_x = 100

# Number of iterations whole algo is to be run
iter_count = 500

# Optimal f(x) as per the paper
optimal_f = 0

# ---------------Implementing Stochastic Ruler Algorithm------------


def step_1(x):
    Nx = N[x-1]
    P = [R[x-1][i-1] for i in Nx]
    z = np.random.choice(Nx, p=P)
    return z


def step_3():
    global k
    k = k + 1


def step_2(k, xk, z, a, b):
    total_tests_to_do = Mk(k)
    test_count = 0
    while (test_count < total_tests_to_do):
        # realization for h(z) i.e. uniform random between f(z)-0.5 , f(z)+0.5
        h_z = get_h_val(f(z))
        theta_val = get_theta_val(a, b)
        if (h_z > theta_val):
            return "Failed"  # go to next step i.e. step-3
        else:
            pass
        test_count += 1
    return "Accept"


depth_went = []
Optimized_correctly = 0

start = time.perf_counter_ns()

for i in range(iter_count):
    print("Iteration", i+1)
    # Initializing K
    k = 0
    # Defining X0
    xk = find_x0()
    z = None
    unchanged_count = 0
    while (True):
        z = step_1(xk)
        #print("k = ",k ," x = ",xk , " z = ",z ,"No neighbour found count = " ,unchanged_count)
        check = step_2(k, xk, z, a, b)
        if (check == "Failed"):
            step_3()
            xk = xk
            unchanged_count += 1
        else:
            step_3()
            xk = z
            unchanged_count = 0
        if (unchanged_count == limit_unchanged_x):
            depth_went.append(k)
            break

    print("Final x_k = ", xk, "where f(x) = ", f(xk))
    if (f(xk) == optimal_f):
        Optimized_correctly += 1
    # go to step-1

end = time.perf_counter_ns()

print("Total Time = ", (end - start)/(10**9), " seconds")
print("Average_Time per iteration = ",
      (end - start)/(iter_count*(10**9)), "seconds")
print("Average cycle depth = ", sum(depth_went)/iter_count)
print("Probability for optimizing correctly = ", Optimized_correctly/iter_count)
