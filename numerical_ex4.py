import numpy as np
import math
import time
from itertools import product

np.random.seed(1234)

# ---------------------Initialization-----------------
n = 6

def define_R():
    R = [1/64 for i in range(64)]
    return R

def define_N_Candidates():
    Neighbours = {}
    for i in range(n):
        for j in range(n):
            Neighbours[(i,j)]=[((i-1)%n,j%n),((i+1)%n,j%n),(i%n,(j-1)%n),(i%n,(j+1)%n)]
    return Neighbours

def N(x):
    return list(product(N_Candidates[x[0]],N_Candidates[x[1]],N_Candidates[x[2]]))

def Mk(k):
    return math.floor(math.log(k+10, 5))

# def f(x):
#     f_vals = [0.3, 0.7, 0.9, 0.5, 1, 1.4, 0.7, 0.8, 0.7, 0.6, 2.1, 2.2, 1.8, 0.1, 0.3, 0.7, 0.9, 1.1, 1.2, 1.4, 1.9, 0.2, 2.1, 2.2, 2.1, 2, 1.9, 1.2, 1.1, 0.7, 0.3, 0.1, 0.1, 1.5, 1.3, 1.2, 1.2, 0.3, 1.6, 0.8, 0.6, 0.7, 1.3, 1.9, 2, 0, 0.8, 0.3, 0.5, 0.6, 1.3, 1.5, 1.5, 1.8, 0.8, 2, 1.9, 1.1, 0.2, 1.3, 0.6, 0.3, 0.5, 0.6, 0.7, 0.9, 1.3, 1.4, 1.6, 0.2, 0.1, 0.8, 0.7, 0.8, 0.9, 0.7, 0.6, 0.5, 0.9, 0.1, 1.4, 1.6, 0.3, 1.1, 0.6, 0.9, 0.1, 2.1, 0.9, 1.8, 1.7, 1.7, 1.3, 1.2, 0.4, 0.9, 1.3, 1.1, 1.9, 1.4]
#     return f_vals[x-1]

def find_x0():
    x0 = list(map(tuple,np.random.randint(0, n,[3,2])))
    return x0

# Defining parameters for Theta
a = 250
b = 800

# Defining Neighbourhood Structure(N)
N_Candidates = define_N_Candidates()

# Defining Transition Probability Matrix(R)
R = define_R()

# Number of iterations whole algo is to be run
iter_count = 1

# Parameters for normal distribution
T0=100
U=180
Sigma=30

# # Optimal f(x) as per the paper
# optimal_f = 0

##################################Implementing Helper functions for algorithm############################################

def nearest_distance(i,j,x):
    ### nearest travelling needed from (i,j) to list of locations in x
    return min([abs(item[0]-i)+abs(item[1]-j)  for item in x])


def get_theta_val(a, b):
    temp = np.random.uniform(low=a, high=b, size=1)  # [low,high)
    return temp[0]

def get_h_val( x, To= T0, n=n, u=U, sigma= Sigma):
    ### x is a list of three tuples with 0-based indexing
    avg_dist_daywise = []
    for t in range(To):
        total_day = 0                                    ##### total distance travelled by people
        ###### now finding nearest facility and saving total distance travelled in each entry of data
        for i in range(n):
            for j in range(n):
                demand=-1
                while(demand<0):
                    demand= np.random.normal(u, sigma, size=1)[0]
                total_day += demand*nearest_distance(i,j,x)  ### total distance from i,j th location to nearest facility
                 
        avg_dist_daywise.append(total_day/(n*n))    
    return sum(avg_dist_daywise)/len(avg_dist_daywise)

##################################Implementing Stochastic Ruler Algorithm############################################

def step_1(x):
    Nx = N(x)
    P = R
    Represent_Nx=[i for i in range(len(Nx))]
    i = np.random.choice(Represent_Nx, p=R)
    return Nx[i]

def step_3():
    global k
    k = k + 1

def step_2(k, xk, z, a, b):
    total_tests_to_do = Mk(k)
    test_count = 0
    while (test_count < total_tests_to_do):
        # realization for h(z) i.e. uniform random between f(z)-0.5 , f(z)+0.5
        h_z = get_h_val(z)
        theta_val = get_theta_val(a, b)
        if (h_z > theta_val):
            return 0  # go to next step i.e. step-3
        else:
            pass
        test_count += 1
    return 1

# def stocastic_ruler():
#     # Defining X0
#     xk = find_x0()
#     z = None
#     while (True):
#         z = step_1(xk)
#         check = step_2(k, xk, z, a, b)
#         if (check == 0):
#             step_3()
#             xk = xk
#         else:
#             step_3()
#             xk = z
#         if (f(xk)==optimal_f):
#             depth_went.append(k)
#             break

# depth_went = []

# total_time=0
# for i in range(iter_count):
#     print("Iteration", i+1)
#     k=0
#     start = time.perf_counter_ns()
#     stocastic_ruler()
#     end = time.perf_counter_ns()
#     total_time+=end-start
#     # go to step-1

# print("Total Time = ", (total_time)/(10**9), " seconds")
# print("Average_Time per iteration = ",
#       (total_time)/(iter_count*(10**9)), "seconds")
# print("Average k = ", sum(depth_went)/iter_count)
