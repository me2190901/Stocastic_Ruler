from glob import glob
from pickle import TRUE
from matplotlib import markers
import numpy as np
import math
import time
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(1234)

# ---------------------Initialization-----------------
N = 6

def define_R():
    R = [1/64 for i in range(64)]
    return R

def define_N_Candidates(n=N):
    Neighbours = {}
    for i in range(n):
        for j in range(n):
            Neighbours[(i,j)]=[((i-1)%n,j%n),((i+1)%n,j%n),(i%n,(j-1)%n),(i%n,(j+1)%n)]
    return Neighbours

def Mk(k):
    return math.floor(math.log(k+10, 5))

def find_x0(n=N):
    x0 = list(map(tuple,np.random.randint(0, n,[3,2])))
    return x0

# Defining parameters for Theta
a = 250
b = 800

# Defining Neighbourhood Structure(N)
N_Candidates = define_N_Candidates(N)
def N(x):
    return list(product(N_Candidates[x[0]],N_Candidates[x[1]],N_Candidates[x[2]]))

# Defining Transition Probability Matrix(R)
R = define_R()

# Number of iterations whole algo is to be run
iter_count = 1

# Parameters for normal distribution
U = 180
Sigma = 30

# Time period for demand measurement
T0 = 30

# Defining percentage reduction for objective function
per_reduction_list=[5,15,25,40,50]
global df
df = pd.DataFrame(columns = ["k","obj_value","per_red"])

# Defining limit on the value of k
limit_k = 300

##################################Implementing Helper functions for algorithm############################################

def define_demand(**para):
    # return np.random.normal(para["u"], para["sigma"], size=1)[0] #for normal distribution of demand
    # return np.random.uniform(para["low"], para["high"], size=1)[0] #for uniform distribution of demand
    return np.random.triangular(para["low"], para["mode"], para["high"], size=1)[0] #for triangular distribution of demand

def nearest_distance(i,j,x):
    ### nearest travelling needed from (i,j) to list of locations in x
    return min([abs(item[0]-i)+abs(item[1]-j)  for item in x])

def get_theta_val(a, b):
    temp = np.random.uniform(low=a, high=b, size=1)  # [low,high)
    return temp[0]

def get_h_val( x, **params):
    ### x is a list of three tuples with 0-based indexing
    avg_dist_daywise = []
    for t in range(params["To"]):
        total_day = 0                                    ##### total distance travelled by people
        ###### now finding nearest facility and saving total distance travelled in each entry of data
        for i in range(n):
            for j in range(n):
                demand=-1
                while(demand<0):
                    # demand = define_demand(u=u,sigma=sigma)  #for normal distribution of demand
                    # demand = define_demand(low=u-3*sigma,high=u+3*sigma)  #for uniform distribution of demand
                    # demand = define_demand(low=u-3*sigma,mode=u      ,high=u+3*sigma)  #for triangular distribution of demand
                    # demand = define_demand(low=u-3*sigma,mode=u-sigma,high=u+3*sigma)  #for triangular distribution of demand (right skewed)
                    demand = define_demand(low=params["u"]-3*params["sigma"],mode=params["u"]+params["sigma"],high=params["u"]+3*params["u"])  #for triangular distribution of demand (left skewed)
                total_day += demand*nearest_distance(i,j,x)  ### total distance from i,j th location to nearest facility
        avg_dist_daywise.append(total_day/(n*n))    
    return sum(avg_dist_daywise)/T0

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

def step_2(k, xk, z, a, b,**params):
    sum_hz=0
    total_tests_to_do = Mk(k)
    test_count = 0
    while (test_count < total_tests_to_do):
        h_z = get_h_val(z,**params)
        theta_val = get_theta_val(a, b)
        if (h_z > theta_val):
            return 0,sum_hz  # go to next step i.e. step-3
        else:
            pass
        test_count += 1
        sum_hz+=h_z
    sum_hz=sum_hz/total_tests_to_do
    return 1, sum_hz

global no_failures
global obj_value
no_failures=0
obj_value=0

def stocastic_ruler(per_reduction,**params):
    np.random.seed(1234)
    global no_failures
    global obj_value
    global df
    no_failures=0
    obj_value=0
    # Defining X0
    xk = find_x0()
    z = None
    fz=[]
    while (True):
        if(obj_value!=0 and (len(df)==0 or k>= df.iloc[-1]["k"]) ):
            df = df.append({'k' : k, 'obj_value' : obj_value, 'per_red' : per_reduction},ignore_index = True)
        z = step_1(xk)
        check,sum_hz = step_2(k, xk, z, a, b,**params)
        if (check == 0):
            step_3()
            xk = xk
            no_failures+=1
        else:
            step_3()
            xk = z
            fz.append(sum_hz)
            obj_value=min(fz)
            if((fz[0]-fz[-1])/fz[0]>=per_reduction/100):
                if(obj_value!=0 and (len(df)==0 or k>= df.iloc[-1]["k"])):
                    df = df.append({'k' : k, 'obj_value' : obj_value, 'per_red' : per_reduction},ignore_index = True)
                break
        if (k>=limit_k):
            break
    matrix  = np.zeros((n,n))
    for loc in xk:
        matrix[loc[0]][loc[1]]=1
    return matrix

total_time=0
avg_no_failures=0
avg_obj_value=0
for per_red in per_reduction_list:
    global k
    k=0
    total_time=0
    avg_no_failures=0
    avg_obj_value=0
    for i in range(iter_count):
        print("Iteration", i+1)
        k=0
        no_failures=0
        obj_value=0
        start = time.perf_counter_ns()
        matrix=stocastic_ruler(per_red,  To=T0, n=N, u=U, sigma=Sigma)
        end = time.perf_counter_ns()
        total_time+=end-start
        avg_no_failures+=no_failures
        avg_obj_value+=obj_value
        # go to step-1
    print("Percentage Reduction:", per_red)
    # print("Total Time = ", (total_time)/(10**9), " seconds")
    print("Average_Time per iteration = ",
        (total_time)/(iter_count*(10**9)), "seconds")
    print("Average Objective function value = ",avg_obj_value/iter_count)
    # print("Average number of failures = ",no_failures/iter_count)
    plt.imshow(matrix , cmap= "Greens" )
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    plt.grid(visible= True,which = "minor", color ="black")
    # plt.title("Distribution: Normal, %red = " +str(per_red)+ "%")
    # plt.title("Distribution: Uniform, %red = " +str(per_red)+ "%")
    # plt.title("Distribution: Triangular(Symmetric), %red = " +str(per_red)+ "%")
    plt.title("Distribution: Triangular(Left Skewed), %red = " +str(per_red)+ "%")
    # plt.title("Distribution: Triangular(Right Skewed), %red = " +str(per_red)+ "%")
    # plt.savefig("./images/Normal_"+str(per_red)+".png")
    # plt.savefig("./images/Uniform_"+str(per_red)+".png")
    # plt.savefig("./images/Triangular_Symmetric_"+str(per_red)+".png")
    plt.savefig("./images/Triangular_Left_Skewed_"+str(per_red)+".png")
    # plt.savefig("./images/Triangular_Right_Skewed_"+str(per_red)+".png")

sns.relplot(data = df, x="k", kind ="line", y ="obj_value", hue = "per_red", markers = True, palette = "husl")
# ---------------------------------------------------title------------------------------------------
# plt.title("Distribution: Normal")
# plt.title("Distribution: Uniform")
# plt.title("Distribution: Triangular(Symmetric)")
plt.title("Distribution: Triangular(Left Skewed)")
# plt.title("Distribution: Triangular(Right Skewed)")
# ---------------------------------------------------file name------------------------------------------
# plt.savefig("./images/Normal_k="+str(limit_k)+".png")
# plt.savefig("./images/Uniform_k="+str(limit_k)+".png")
# plt.savefig("./images/Triangular_Symmetric_k="+str(limit_k)+".png")
plt.savefig("./images/Triangular_Left_Skewed_k="+str(limit_k)+".png")
# plt.savefig("./images/Triangular_Right_Skewed_k="+str(limit_k)+".png")