# -*- coding: utf-8 -*-

'''
@Time    : 2021/3/2 23:31
@Author  : Qiushi Wang
@FileName: mcmc.py
@Software: PyCharm
'''

import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


class MC(object):
    def __init__(self, loss, T, mean, std):
        self.loss = loss
        self.T = T
        self.mean = mean
        self.std = std

    def norm_dist_prob(self, theta):
        y = norm.pdf(theta, loc=self.mean, scale=self.std)
        return y

    def MCMC(self):
        pi = [0 for _ in range(self.T)]
        sigma = 1
        t = 0
        while t < self.T - 1:
            t = t + 1
            pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
            alpha = min(1, (self.norm_dist_prob(pi_star[0]) / self.norm_dist_prob(pi[t - 1])))

            u = random.uniform(0, 1)
            if u < alpha:
                pi[t] = pi_star[0]
            else:
                pi[t] = pi[t - 1]
        res = []
        for pre_lo in pi:
            for raw_lo in self.loss:
                if abs(pre_lo - raw_lo) < 1e-4:
                    res.append(raw_lo)

        return res

# plt.scatter(pi, norm.pdf(pi, loc=loss.mean(), scale=loss.std()))
# num_bins = 50
# plt.hist(pi, num_bins, density=1, stacked=True, facecolor='red', alpha=0.7)
# plt.show()
