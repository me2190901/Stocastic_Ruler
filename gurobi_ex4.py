import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Create a new model
m = gp.Model("ex4")

n = 6*6
p = 3
u = 180
T0 = 30

def distance(location, facility, N):
    row1 = location // N
    col1 = location % N
    row2 = facility // N
    col2 = facility % N
    return abs(row1 - row2) + abs(col1 - col2)

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
m.optimize()
# print solution
output_mat = np.zeros((round(pow(n,0.5)),round(pow(n,0.5))))
for v in m.getVars():
    if(v.x == 1 and v.varName[0] == 'x'):
        id_ = int(v.varName[2:-1])
        output_mat[id_//round(pow(n,0.5))][id_%round(pow(n,0.5))] = 1

print(output_mat)