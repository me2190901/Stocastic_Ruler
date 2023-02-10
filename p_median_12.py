import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Create a new model
m = gp.Model("ex4")

def find_u_triangle(a,c,b):
    return ((c-a)*(2*c+a)/(3*(b-a)))-((c-b)*(2*c+b)/(3*(b-a)))

def find_u_bimodal(u1,u2,weight):
    if u1>u2:
        u1, u2 = u2, u1
    return (weight*u1 + (1-weight)*u2)

# # initiate a dataframe to store the results
# df= pd.DataFrame(columns = ['u1','sigma1','u2','sigma2','weight','obj','time'])

def distance(location, facility, N):
    row1 = location // N
    col1 = location % N
    row2 = facility // N
    col2 = facility % N
    return abs(row1 - row2) + abs(col1 - col2)

def solver(n,p,u,T0,**params):
    h = [u*T0 for i in range(n)]
    dij = [[distance(i,j, pow(n,0.5)) for i in range(n)] for j in range(n)]

    Demand = set(i for i in range(n))
    Facility = set(j for j in range(n))

    # Create variables
    y = m.addVars(Demand, Facility, vtype=GRB.BINARY, name="y")
    x = m.addVars(Facility, vtype=GRB.BINARY, name="x")

    # constraints
    m.addConstrs((gp.quicksum(y[i,j] for j in Facility) == 1 for i in Demand), name ="c1")
    m.addConstrs((y[i,j]-x[j] <= 0 for i in Demand for j in Facility), name= "c2")
    m.addConstr((gp.quicksum(x[j] for j in Facility) == p), name="c3")

    # Set objective
    m.setObjective(gp.quicksum(h[i]*y[i,j]*dij[i][j] for i in Demand for j in Facility), GRB.MINIMIZE)

    # Optimize model
    start_time = time.perf_counter_ns()
    m.optimize()
    end_time = time.perf_counter_ns()
    # print("Time taken to solve the model is ", (end_time-start_time)/1000000000, "seconds")

    # # PRINTING computation time
    # print("Computation time: ", m.Runtime)

    # print solution
    output_mat = np.zeros((round(pow(n,0.5)),round(pow(n,0.5))))
    for v in m.getVars():
        if(v.x == 1 and v.varName[0] == 'x'):
            id_ = int(v.varName[2:-1])
            output_mat[id_//round(pow(n,0.5))][id_%round(pow(n,0.5))] = 1

    objec = m.getObjective()
    obj=objec.getValue()/(n*T0)
    # # append the results to the dataframe
    # df.loc[len(df)] = [params["u1"],params["sigma1"],params["u2"],params["sigma2"],params["weight"],obj,(end_time-start_time)/1000000000]

    # plt.imshow(output_mat , cmap= "Greens" )
    # ax = plt.gca()
    # ax.set_xticks(np.arange(0, round(pow(n,0.5)), 1))
    # ax.set_yticks(np.arange(0, round(pow(n,0.5)), 1))
    # ax.set_xticks(np.arange(-.5, round(pow(n,0.5)), 1), minor=True)
    # ax.set_yticks(np.arange(-.5, round(pow(n,0.5)), 1), minor=True)
    # ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
    # plt.grid(visible=True,which = "minor", color ="black")
    # plt.title("Distribution: Normal, N~({},{})".format(u,sigma))
    # plt.savefig("./images_12/gurobi_normal_{}_{}_12.png".format(u,params["sigma"]))
    # plt.title("Distribution: Uniform, U~({},{})".format(u,sigma))
    # plt.savefig("./images_12/gurobi_uniform_{}_{}_12.png".format(u,params["sigma"]))
    # plt.title("Distribution: Triangular_Symmetric, T~({},{})".format(u,sigma))
    # plt.savefig("./images_12/gurobi_tri_sym_{}_{}_12.png".format(u,params["sigma"]))
    # plt.title("Distribution: Triangular_Left_Skewed, T~({},{})".format(u,sigma))
    # plt.savefig("./images_12/gurobi_tri_left_{}_{}_12.png".format(u,params["sigma"]))
    # plt.title("Distribution: Triangular_Right_Skewed, T~({},{})".format(u,sigma))
    # plt.savefig("./images_12/gurobi_tri_right_{}_{}_12.png".format(u,params["sigma"]))

    # plt.title("Distribution: Bimodal, N1~({},{}), N2~({},{}), Weight1={}".format(params["u1"],params["sigma1"],params["u2"],params["sigma2"],params["weight"]))
    # plt.savefig("./images_12/gurobi_bimodal_{}_{}_{}_{}_{}_12.png".format(params["u1"],params["sigma1"],params["u2"],params["sigma2"],params["weight"]))
    # plt.close()
    return obj
# n = 12*12
# p = 3
# u = 180
# sigma=30
# # u = find_u_triangle(u-3*sigma,u,u+3*sigma) # u for symmetric triangle distribution
# # u = find_u_triangle(u-3*sigma,u-sigma,u+3*sigma) # u for right skewed triangle distribution
# u = find_u_triangle(u-3*sigma,u+sigma,u+3*sigma) # u for left skewed triangle distribution
# T0 = 30
# print(solver(n, p, u, T0,sigma=sigma))

# Parameters for normal distribution
# Mean1 = [180, 180, 180, 180, 180, 180]
# Sigma1 = [30, 30, 30, 30, 30, 30]
# Mean2 = [200, 120,300, 300, 120, 120] 
# Sigma2 = [30, 20, 60, 40, 30, 10]
# Weight1 = [0.2,0.4,0.5,0.6,0.8]
# for w in Weight1:
#     for i in range(len(Mean1)):
#         u1 = Mean1[i]
#         sigma1 = Sigma1[i]
#         u2 = Mean2[i]
#         sigma2 = Sigma2[i]
    # u = find_u_bimodal(u1,u2,w)
    # solver(n,p,u,T0,u1=u1,sigma1=sigma1,u2=u2,sigma2=sigma2,weight=w)

# df.to_csv("bimodal_gurobi_results.csv")