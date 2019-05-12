# Importing the useful libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm 
from scipy.stats import shapiro

# Kernel functions

gaussian = lambda sigma: lambda x,y : np.exp(-np.linalg.norm(x-y,axis=1)**2/(2*sigma))
laplacian = lambda sigma: lambda x,y : np.exp(-sigma*np.linalg.norm(x-y,axis=1,ord=1))

def CCK(kernels_list, weights_list):
    """ Compute the convex sum of two kernels
    - Params:
    kernels_list : list of all kernel of the sum
    weights_list : The weights of the kernel sum
    
    - Return :
    A kernel, If all kernels of the sum are characteristics,
    then the returned kernel is also characterisitic"""
    def kernel(x,y):
        res = 0
        for k,w in zip(kernels_list,weights_list):
            res += w*k(x,y)
        return res
    return  kernel

# Probability law
normal = lambda mu, sigma : lambda size : sigma*np.random.randn(size).reshape(-1,1) + mu
uniform = lambda a, b: lambda size : np.random.uniform(low=a, high=b, size=(size,1))

inv_phi = lambda x : norm.ppf(x)

# MMD Estimators
#Have to be build as scikit-learn estimator
def MMDq(X,Y,kernel):
    m = X.shape[0]
    res = 0
    
    X_1,X_2 = np.meshgrid(X,X)
    m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
    X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
    res += kernel(X_,Y_).sum()/(m*(m-1))
    
    X_1,X_2 = np.meshgrid(Y,Y)
    m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
    X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
    res += kernel(X_,Y_).sum()/(m*(m-1))
    
    X_1,X_2 = np.meshgrid(X,Y)
    m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
    X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
    res -= 2*kernel(X_,Y_).sum()/(m*(m-1))
    
    return res

def MMDl(X,Y,kernel):
    m = X.shape[0]
    res = 0
    
    X_, Y_ = X[0::2,:],X[1::2,:]
    res += kernel(X_,Y_).sum()
    
    
    X_, Y_ = Y[0::2,:],Y[1::2,:]
    res += kernel(X_,Y_).sum()
    
    X_, Y_ = X[0::2,:],Y[1::2,:]
    res -= kernel(X_,Y_).sum()
    
    X_, Y_ = Y[0::2,:],X[1::2,:]
    res -= kernel(X_,Y_).sum()
    
    return 2*res/m

class OMMD():
    def __init__(self,kernel,X):
        self.X = X
        self.Y = []
        self.m = X.shape[0]
        self.n = 0
        self.kernel = kernel
        
        X_, Y_ = X[0::2,:],X[1::2,:]
        self.estim_normP = 2/self.m*self.kernel(X_,Y_).sum()
        
        self.sumNormQ = 0
        self.sumPS = 0
        
        self.MMD = self.estim_normP 
           
    def update(self,y1,y2):
        self.sumNormQ += self.kernel(y1,y2)
        
        self.sumPS += self.kernel(self.X,y1).sum() + self.kernel(self.X,y2).sum()
        
        self.n += 2
        
        self.MMD = self.estim_normP + 2/self.n * self.sumNormQ - 2/(self.n*self.m)*self.sumPS
        
    def fit(self, Y):
        for i in range(0,Y.shape[0]//2):
            self.update(Y[2*i].reshape((-1,1)),Y[2*i+1].reshape((-1,1)))
            
 # Estimators 
def sampleMMD(kernel, law_p, law_q, m, n,repeat = 1000):
    
    Liste_mmdl = []
    for _ in range(repeat):
        if (_%(repeat//100)==0):
            print(_/repeat*100, '% achevé',end='\r',flush=True)
        X = law_p(m)
        Y = law_q(n)

        MMD =  OMMD(kernel, X)
        MMD.fit(Y)
        Liste_mmdl.append(MMD.MMD[0])
    return np.r_[Liste_mmdl]

def moyenne(kernel, X_, Y, repeat=100):
    mmd = np.zeros(repeat)
    for i in range(repeat):
        MMD = OMMD(kernel,np.random.choice(X_.flatten(), size = 100, replace=True).reshape((-1,1)))
        MMD.fit(np.random.choice(Y.flatten(), size = 100, replace=True).reshape((-1,1)))
        mmd[i] = MMD.MMD
        #print(f"Progression {(i/repeat)*100:.0f}%", end="\r", flush=True)
    return mmd.mean()

def est_partial_mean(kernel, X, y):
    return np.mean(kernel(y, X))


def estim_V(kernel, X, Y, repeat=1000):
    echantillions = np.zeros(repeat)
    for i in range(repeat):
        x1 = np.random.choice(X.flatten(), size = 1, replace=True).reshape((-1,1))
        x2 = np.random.choice(X.flatten(), size = 1, replace=True).reshape((-1,1))
        Y_1 = np.random.choice(Y.flatten(), size = 100, replace=True).reshape((-1,1))
        Y_2 = np.random.choice(Y.flatten(), size = 100, replace=True).reshape((-1,1))
        echantillions[i] = kernel(x1,x2) - est_partial_mean(kernel, x1, Y_1) - est_partial_mean(kernel, x2, Y_2)
    return echantillions.var()

def var_H(kernel, X, Y, alpha, size=100):
    return estim_V(kernel, X, Y) + 1/alpha*estim_V(kernel, Y, X)

def V1V0(sigma_H0, alpha, sigma_H1):
    """ Renvoie le premier terme de l'élément à optimiser"""
    return sigma_H0*inv_phi(1 - alpha)/sigma_H1
def MV(m, MMD_H1, sigma_H1):
    """ Renvoie le deuxième terme de l'élément à optimiser"""
    return np.sqrt(m/2)*MMD_H1/sigma_H1


        

