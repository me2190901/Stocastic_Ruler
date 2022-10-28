import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, pi

np.random.seed(1234)
bin_size = 0.1
# define bimodal distribution function with mean 150 and 200 and std 30 and 50 respectively


def bimodal_distribution(x, **params):
    return (params["weight1"])*np.exp(-((x-params["u1"])/params["sigma1"])**2)/sqrt(2*pi*params["sigma1"]**2) + (1-params["weight1"])*np.exp(-((x-params["u2"])/params["sigma2"])**2)/sqrt(2*pi*params["sigma2"]**2)


def bimodal_sample(n, weight1, u1, sigma1, u2, sigma2):
    x = np.arange(u1-3*sigma1, u2+3*sigma2, bin_size)
    data = [bimodal_distribution(
        xi, weight1=weight1, u1=u1, u2=u2, sigma1=sigma1, sigma2=sigma2) for xi in x]
    # generate sample with probability distribution as bimodal_distribution
    bin_nos = np.random.choice(
        range(floor((u2-u1+3*sigma2+3*sigma1)/bin_size)), n, p=data/np.sum(data))
    # take uniform random sample from the bin
    sample = []
    for bin_no in bin_nos:
        sample.append(np.random.uniform((u1-3*sigma1)+bin_no *
                      bin_size, u1-3*sigma1+(bin_no+1)*bin_size))
    return sample


weight1 = 0.5
u1 = 180
u2 = 300
sigma1 = 30
sigma2 = 50
data = bimodal_sample(5000, weight1, u1, sigma1, u2, sigma2)
