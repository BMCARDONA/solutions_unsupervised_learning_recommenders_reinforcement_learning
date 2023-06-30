# GRADED FUNCTION: cofi_cost_func
# UNQ_C1

def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    ### START CODE HERE ### 
    
    for j in range(nu):
        for i in range(nm):
            is_rated = R[i, j] 
            dot_product = np.dot(W[j], X[i])
            parentheses_term = (dot_product + b[0, j] - Y[i, j]) ** 2
            J += (is_rated * parentheses_term)
    J /= 2
    
    # add regularization
    n = len(X[0])
    
    first_regularization_term = 0
    for j in range(nu):
        for k in range(n):
            term = W[j, k] ** 2
            first_regularization_term += term
    first_regularization_term *= (lambda_ / 2)
    J += first_regularization_term
    
    second_regularization_term = 0
    for i in range(nm):
        for k in range(n):
            term = X[i, k] ** 2
            second_regularization_term += term
    second_regularization_term *= (lambda_ / 2)
    J += second_regularization_term
    
    ### END CODE HERE ### 
    

    return J