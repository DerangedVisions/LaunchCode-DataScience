from costFunction import costFunction


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    
    import numpy as np 
    from sigmoid import sigmoid
    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
    element1 = y*np.transpose(np.log(sigmoid(np.dot(X,theta))))
    element2 = (1-y) * np.transpose(np.log(1-sigmoid(np.dot(X,theta))))
    r = (float(lambda_reg)/(2*m)) *np.power(theta[1:theta.shape[0]],2).sum()
    J= (1/m)* (element1+element2).sum() + reg
    
    grad = (1/m)*np.dot
# =============================================================
    J = 0
    return J
