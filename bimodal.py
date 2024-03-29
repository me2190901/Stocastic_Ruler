import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, pi
import sys
import seaborn as sns

class BimodalDistribution:
    bin_size = 0.1
    u1=0
    u2=0
    sigma1=0
    sigma2=0
    weight1=0
    data = []
    def __init__(self, u1, sigma1, u2, sigma2, weight1):
        np.random.seed(1234)
        self.u1 = u1
        self.u2 = u2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.weight1 = weight1
        x = np.arange(self.u1-3*self.sigma1, self.u2+3*self.sigma2, self.bin_size)
        self.data = [self.bimodal_distribution(xi) for xi in x]

    def bimodal_distribution(self, x):
        return (self.weight1)*np.exp(-0.5*((x-self.u1)/self.sigma1)**2)/sqrt(2*pi*self.sigma1**2) + (1-self.weight1)*np.exp(-0.5*((x-self.u2)/self.sigma2)**2)/sqrt(2*pi*self.sigma2**2)

    def get_sample(self, n=1):
        # generate sample with probability distribution as bimodal_distribution
        bin_nos = np.random.choice( range(floor((self.u2-self.u1+3*self.sigma2+3*self.sigma1)/self.bin_size)), n, p=self.data/np.sum(self.data))
        # take uniform random sample from the bin
        sample = []
        for bin_no in bin_nos:
            sample.append(np.random.uniform((self.u1-3*self.sigma1)+bin_no*self.bin_size, self.u1-3*self.sigma1+(bin_no+1)*self.bin_size))
        return sample

def bimodal_distribution(x, **params):
    return (params["weight1"])*np.exp(-0.5*((x-params["u1"])/params["sigma1"])**2)/sqrt(2*pi*params["sigma1"]**2) + (1-params["weight1"])*np.exp(-0.5*((x-params["u2"])/params["sigma2"])**2)/sqrt(2*pi*params["sigma2"]**2)

if __name__ == "__main__":
    weight1 = 0.5
    u1 = 180
    u2 = 300
    sigma1 = 30
    sigma2 = 30

    bimodal= BimodalDistribution(u1, sigma1, u2, sigma2, weight1)
    data=[]
    for i in range(50000):
        data.append(bimodal.get_sample(1)[0])
    # data=bimodal.get_sample(1)

    # plt.show()
    x = np.arange(u1-3*sigma1, u2+3*sigma2, 0.1)
    plt.plot(x, [bimodal_distribution( xi, weight1=weight1, u1=u1, u2=u2, sigma1=sigma1, sigma2=sigma2) for xi in x], label="True Distribution")
    plt.hist(data, bins=100, density=True, label="Sampled Distribution")
    plt.legend()
    plt.show()