import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, pi
import sys

bin_size = 0.1

def bimodal_distribution(x, **params):
    return (params["weight1"])*np.exp(-0.5*((x-params["u1"])/params["sigma1"])**2)/sqrt(2*pi*params["sigma1"]**2) + (1-params["weight1"])*np.exp(-0.5*((x-params["u2"])/params["sigma2"])**2)/sqrt(2*pi*params["sigma2"]**2)

global batch_size , i, Batch, curr_weight, curr_per_red
Batch = []
batch_size = 50000
i = batch_size
curr_weight = -1
curr_per_red = 0
o
def create_batch(n, weight1 , u1, sigma1, u2, sigma2):
    x = np.arange(u1-3*sigma1, u2+3*sigma2, bin_size)
    data = [bimodal_distribution( xi, weight1=weight1, u1=u1, u2=u2, sigma1=sigma1, sigma2=sigma2) for xi in x]
    # generate sample with probability distribution as bimodal_distribution
    bin_nos = np.random.choice( range(floor((u2-u1+3*sigma2+3*sigma1)/bin_size)), n, p=data/np.sum(data))
    # take uniform random sample from the bin
    sample = []
    for bin_no in bin_nos:
        sample.append(np.random.uniform((u1-3*sigma1)+bin_no*bin_size, u1-3*sigma1+(bin_no+1)*bin_size))
    return sample

def get_bimodal_sample(mixing_prob, per_red , u1 , sigma1 , u2, sigma2, Batch_Size):
    global batch_size , i, Batch, curr_weight, curr_per_red
    if(mixing_prob!=curr_weight or curr_per_red!=per_red):
        print("SEED-RESETING..........", file= sys.stderr)
        curr_weight = mixing_prob
        curr_per_red = per_red
        np.random.seed(1234)
        batch_size = Batch_Size
        i = batch_size
    
    if(i==Batch_Size):
        print("Creating a Batch....", file= sys.stderr)
        Batch = create_batch(batch_size, mixing_prob, u1, sigma1, u2, sigma2)
        i = 1
        return Batch[0]
    else:
        i = i + 1
        return Batch[i-1]

# weight1 = 0.5
# u1 = 180
# u2 = 300
# sigma1 = 30
# sigma2 = 50
# data = bimodal_sample(5000, weight1, u1, sigma1, u2, sigma2)
# plt.hist(data, bins=100)
# plt.show()