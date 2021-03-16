import numpy as np
import random
import math
import sys
from scipy.stats import bernoulli

class LogReg:
    def __init__(self, A, b, reg_coef, n, m, d):
        '''
        A - data points
        b - targets
        reg_coef - L2 regularization coefficient
        n - number of nodes
        m - number of local data points
        '''
        self.A = A
        self.b = b
        self.lmb = reg_coef 
        self.n = n
        self.m = m
        self.d = A.shape[1]
        
    def function_value(self, x):
        '''
        x - current model weights
        return: F(x) + lmb/2||x||^2
        '''
        ans = 0
        N = self.n*self.m
        for i in range(N):
            ans += np.log(1+math.exp(-self.b[i]*self.A[i].dot(x)))/N
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def local_function_value(self, x, i):
        ans = 0
        left = i*self.m
        right = (i+1)*self.m
        for j in range(left, right):
            ans += np.log(1+math.exp(-self.b[j]*self.A[j].dot(x)))/self.m
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def Hessian(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        return: Hessian of f_ij(x)
        '''
        l = i*self.m + j
        alpha = self.b[l]**2*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))**2
        ans = alpha*self.A[l].reshape((self.d,1)).dot(self.A[l].reshape(1,self.d))
        return ans
    
    def alpha(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        return: alpha_ij(x)
        '''
        l = i*self.m + j
        alpha = self.b[l]**2*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))**2
        return alpha
    
    def gradient(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        return: gradient of f_ij(x)
        '''
        l = self.m*i + j
        alpha = -self.b[l]*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))
        ans = alpha*self.A[l]
        return ans
    
    def local_gradient(self, x, i):
        '''
        x - current model weights
        i - node number
        return: gradient of f_i(x)
        '''
        m = self.m
        g = np.zeros(self.d)
        for j in range(m):
            g += 1/m*self.gradient(x, i, j)
        return g
    
    def local_Hessian(self, x, i):
        m = self.m
        H = np.zeros((self.d, self.d))
        for j in range(m):
            H += 1/m*self.Hessian(x, i, j)
        return H
    
    def full_Hessian(self, x):
        '''
        x - current model weights
        return: full Hessian of f(x)
        '''
        N = self.m*self.n
        d = self.d 
        H = np.zeros((d,d))
        for i in range(self.n):
            for j in range(self.m):
                H += 1/N*self.Hessian(x, i, j)
        return H
    
    def full_gradient(self, x):
        '''
        x - current model weights
        return: full gradient of f(x)
        '''
        N = self.m*self.n
        g = np.zeros(self.d)
        for i in range(self.n):
            for j in range(self.m):
                g += 1/N*self.gradient(x, i, j)
        return g


    def alphas(self, x, i):
        '''
        x - current model weights
        i - node number
        return: vector alpha_i(x), i.e. [alpha_i(x)]_j = alpha_ij(x)
        '''
        answer = np.zeros(self.m)
        for j in range(self.m):
            answer[j] = self.alpha(x, i, j)
        return answer
    
    def get_reg_coef(self):
        '''
        return: regularization coefficient
        '''
        return self.lmb
    
    def get_number_of_weights(self):
        '''
        return: the dimension of weights space 
        '''
        return self.d
    
    def get_number_of_nodes(self):
        '''
        return: number of nodes n
        '''
        return self.n
    
    def get_number_of_local_data_points(self):
        '''
        return: number of data points m in local dataset
        '''
        return self.m
    
    def get_optimum(self):
        '''
        return: optimal solution of the problem
        '''
        return self.x_opt
    
    def set_optimum(self, x_opt):
        '''
        set the optimal solution of the problem
        '''
        self.x_opt = x_opt