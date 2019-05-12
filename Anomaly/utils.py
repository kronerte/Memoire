#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:30:42 2019

@author: kronert
"""
import numpy as np
from scipy.stats import norm 


# Probability law
normal = lambda mu, sigma : lambda size : sigma*np.random.randn(size).reshape(-1,1) + mu
uniform = lambda a, b: lambda size : np.random.uniform(low=a, high=b, size=(size,1))

inv_phi = lambda x : norm.ppf(x)


#
def bootstrap(X, n, with_replacement = True):
    size, dim = X.shape
    indices = np.random.choice(range(size), size = n, replace = with_replacement)
    return X[indices]
    