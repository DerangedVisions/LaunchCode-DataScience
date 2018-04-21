import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    m = len(y) # number of training examples
    J=0


# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#theta.shape=(3,), X.Shape=(100,3), y.shape=(100,) 

    gradient_1=y*np.transpose(np.log(1-sigmoid(np.dot(X,theta))))
    gradient_2=(1-y) * np.transpose(np.log(sigmoid(np.dot(X,theta) ) ))
    J = -(1./m)*(gradient_1+gradient_2).sum()
    
    return J