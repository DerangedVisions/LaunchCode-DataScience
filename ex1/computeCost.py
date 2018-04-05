import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and 
    """
    
    m = y.size
    J = 0
    s = 0
    for r in range(X.shape[0]):
        Xr = X[r,] 
        yr = y[r]
        s = s + np.power(Xr.dot(theta) - yr,2)
    J = s * (1.0/(2 * m))
   

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.


# ========================================================================
    return J




#'''X = [ones(m, 1), data(:,1)]; ''' % Add a column of ones to x'''
#theta = zeros(2, 1); '''% initialize fitting parameters'''
#iterations = 1500;
##alpha = 0.01;