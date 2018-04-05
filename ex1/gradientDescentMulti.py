from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        s0 = 0
        s1 = 0
        for r in range(X.shape[0]):
           Xr = X[r,] 
           yr = y[r]
           s0 = s0 + (theta[0]+theta[1]*Xr[1]-yr)*Xr[0]
           s1 = s1 + (theta[0]+theta[1]*Xr[1]-yr)*Xr[1]
    s = np.array([s0,s1])   
    theta= theta- alpha * (s)/m
    J_history.append(computeCost(X, y, theta))

    return theta, J_history
        #   ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #



        # ============================================================

        # Save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history