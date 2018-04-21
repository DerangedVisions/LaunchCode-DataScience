from costFunctionReg import costFunctionReg

def lrCostFunction(theta, X, y, Lambda):
    """computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    import numpy as np
    from sigmoid import sigmoid
    m=len(y)
    X_theta = X.dot(theta)
    mask= np.eye(len(theta))
    
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#
# Hint: The computation of the cost function and gradients can be
#       efficiently vectorized. For example, consider the computation
#
#           sigmoid(X * theta)
#
#       Each row of the resulting matrix will contain the value of the
#       prediction for that example. You can make use of this to vectorize
#       the cost function and gradient computations. 
#
    
    J= 1/m * (-y* np.transpose(np.log(sigmoid(np.dot(X,theta)))))\
    + lambda/(2*m)*np.sum(np.square(mask.dot(theta)))
    grad = 1.0 / m * (sigmoid(X_theta) - y).T.dot(X).T + 1.0 * lambda \
    / m * (mask.dot(theta))
    
    grad= 1/m * sigmoid(X_theta)-y).T.dot(X).T+1*1/m *(mask.dot(theta))
    # =============================================================

    return J,grad
