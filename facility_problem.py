from matplotlib import markers
import numpy as np
import math
import time
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bimodal

np.random.seed(1234)

# ---------------------Inputs------------------------
### Demand_distribution_name **params (space separated)

### Normal U Sigma limit_k per_reduction_values
### Uniform U Sigma limit_k per_reduction_values
### Triangular_Symmetric U Sigma limit_k per_reduction_values
### Triangular_Left_Skewed U Sigma limit_k per_reduction_values
### Triangular_Right_Skewed U Sigma limit_k per_reduction_values
### Normal_bimodal U1 Sigma1 U2 Sigma2 mixing_prob limit_k per_reduction_values

### After limit_k value, per_reduction values are present with space separated format, for which computation need to be done



# ------------------ Reading Input ---------------------

Input_list = input().strip().split()
global Distribution_type
Distribution_type = Input_list[0]

global limit_k,per_reduction_list
if(Distribution_type!="Normal_bimodal"):
    global U, Sigma 
    U, Sigma, limit_k = int(Input_list[1]) , int(Input_list[2]) , int(Input_list[3])
    per_reduction_list = list(map(int, Input_list[4:]))
elif(Distribution_type=="Normal_bimodal"):
    global U1, Sigma1, U2, Sigma2, mixing_prob
    U1, Sigma1, U2, Sigma2, mixing_prob, limit_k = int(Input_list[1]) , int(Input_list[2]) , int(Input_list[3]), int(Input_list[4]) , float(Input_list[5]) , int(Input_list[6])
    per_reduction_list = list(map(int, Input_list[7:]))
else:
    print("Incorrect_Input")

global df
df = pd.DataFrame(columns = ["k","obj_value","per_red"])



# ---------------------Initialization-----------------
global n
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

def Mk(k):
    return math.floor(math.log(k+10, 5))+5

def find_x0():
    x0 = list(map(tuple,np.random.randint(0, n,[3,2])))
    return x0

# Defining parameters for Theta
global a, b
a = 250
b = 800

# Defining Neighbourhood Structure(N)
N_Candidates = define_N_Candidates()
def N(x):
    return list(product(N_Candidates[x[0]],N_Candidates[x[1]],N_Candidates[x[2]]))

# Defining Transition Probability Matrix(R)
R = define_R()

# Number of iterations whole algo is to be run
iter_count = 1

# Time period for demand measurement
global T0
T0 = 30

##################################Implementing Helper functions for algorithm############################################

def define_demand(**para):
    if(Distribution_type=="Normal"):
        return np.random.normal(para["u"], para["sigma"], size=1)[0] #for normal distribution of demand
    elif(Distribution_type=="Uniform"):
        return np.random.uniform(para["low"], para["high"], size=1)[0] #for uniform distribution of demand
    elif(Distribution_type in ["Triangular_Symmetric","Triangular_Left_Skewed" ,"Triangular_Right_Skewed" ] ):
        return np.random.triangular(para["low"], para["mode"], para["high"], size=1)[0] #for triangular distribution of demand

def nearest_distance(i,j,x):
    ### nearest travelling needed from (i,j) to list of locations in x
    return min([abs(item[0]-i)+abs(item[1]-j)  for item in x])

def get_theta_val(a, b):
    temp = np.random.uniform(low=a, high=b, size=1)  # [low,high)
    return temp[0]

def get_h_val( x ):
    ### x is a list of three tuples with 0-based indexing
    avg_dist_daywise = []
    for t in range(T0):
        total_day = 0                                    ##### total distance travelled by people
        ###### now finding nearest facility and saving total distance travelled in each entry of data
        for i in range(n):
            for j in range(n):
                demand=-1
                while(demand<0):    
                    if(Distribution_type=="Normal"):
                        demand = define_demand(u=U,sigma=Sigma)   #for normal distribution of demand
                    elif(Distribution_type=="Uniform"):
                        demand = define_demand(low=U-3*Sigma,high=U+3*Sigma)  #for uniform distribution of demand
                    elif(Distribution_type=="Triangular_Symmetric"):
                        demand = define_demand(low=U-3*Sigma,mode=U      ,high=U+3*Sigma)  #for triangular distribution of demand
                    elif(Distribution_type=="Triangular_Left_Skewed"):
                        demand = define_demand(low=U-3*Sigma,mode=U+Sigma,high=U+3*Sigma)  #for triangular distribution of demand (left skewed)
                    elif(Distribution_type=="Triangular_Right_Skewed"):
                        demand = define_demand(low=U-3*Sigma,mode=U-Sigma,high=U+3*Sigma)  #for triangular distribution of demand (right skewed)
                    elif(Distribution_type=="Normal_bimodal"):
                        demand = bimodal.get_bimodal_sample(mixing_prob, per_red, U1, Sigma1, U2, Sigma2,n*n*T0*limit_k) ## for bimodal distribution of demand
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

def step_2(k, xk, z, a, b):
    sum_hz=0
    total_tests_to_do = Mk(k)
    test_count = 0
    while (test_count < total_tests_to_do):
        h_z = get_h_val(z)
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

def stocastic_ruler(per_reduction):
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
            df = pd.concat([df, pd.DataFrame([[k, obj_value, per_reduction]], columns = ["k","obj_value","per_red"])], ignore_index = True)
        z = step_1(xk)
        check,sum_hz = step_2(k, xk, z, a, b)
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
                    df = pd.concat([df, pd.DataFrame([[k, obj_value, per_reduction]], columns = ["k","obj_value","per_red"])], ignore_index = True)
                break
        if (k>=limit_k):
            break
    print("Solution_locations",xk)
    matrix  = np.zeros((n,n))
    for loc in xk:
        matrix[loc[0]][loc[1]]=1
    return matrix



################--------- Creating Results ------------------------

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
        matrix=stocastic_ruler(per_red)
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
    
    plt.imshow(matrix , cmap= "Greens" )                             #### Saving solution of locations
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n, 1))
    ax.set_yticks(np.arange(0, n, 1))
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    plt.grid(visible= True,which = "minor", color ="black")
    
    if (Distribution_type=="Normal_bimodal"):
        plt.title("Distribution: Bimodal, N1~({},{}), N2~({},{}), Weight1={}, Limit_k={}".format(U1,Sigma1,U2,Sigma2,mixing_prob,limit_k))
        plt.savefig("./images/Bimodal_{}_{}_{}_{}_{}_{}".format(U1,Sigma1,U2,Sigma2,mixing_prob,limit_k)+".png", bbox_inches='tight')
    else:
        plt.title("Distribution: "+Distribution_type+", %red = "+str(per_red) + "%")
        plt.savefig("./images/"+Distribution_type +"_"+str(per_red)+".png", bbox_inches='tight')
    plt.close()

sns.relplot(data = df, x="k", kind ="line", y ="obj_value", hue = "per_red", markers = True, palette = "husl")
# set x limit
plt.xlim(0, limit_k)
# add horizontal line for max value of objective function value and minimum value of objective function value and show its value
plt.text(0, max(df["obj_value"])+0.5, "Initial = {}".format(max(df["obj_value"])))
plt.axhline(y = max(df["obj_value"]), color = "green", linestyle = "--")
plt.text(0, min(df["obj_value"])+0.5, "Final = {}".format(min(df["obj_value"])))
plt.axhline(y = min(df["obj_value"]), color = "green", linestyle = "--")
if (Distribution_type!="Normal_bimodal"):
    # ---------------------------------------------------title------------------------------------------
    plt.title("Distribution: "+Distribution_type)
    # ---------------------------------------------------file name------------------------------------------
    plt.savefig("./images/"+Distribution_type +"_k="+str(limit_k)+".png", bbox_inches='tight')
else:
    # ---------------------------------------------------title------------------------------------------
    plt.title("Distribution: Bimodal, N1~({},{}), N2~({},{}), Weight1={}, Limit_k={}".format(U1,Sigma1,U2,Sigma2,mixing_prob,limit_k))
    # ---------------------------------------------------file name------------------------------------------
    plt.savefig("./images/Bimodal_{}_{}_{}_{}_{}_{}_k".format(U1,Sigma1,U2,Sigma2,mixing_prob,limit_k)+".png", bbox_inches='tight')
plt.close()