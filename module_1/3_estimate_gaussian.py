# UNQ_C1
# GRADED FUNCTION: estimate_gaussian

def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    
    ### START CODE HERE ### 
    
    # means for each of the n features
    mu = np.zeros(n)
    for i in range(n):
        feature_mean = 0
        for j in range(m):
            feature_mean += X[j, i]
        feature_mean /= m
        mu[i] = feature_mean
    
    # Variances for each of the n features
    var = np.zeros(n)
    for i in range(n):
        feature_variance = 0
        for j in range(m):
            feature_variance += (X[j, i] - mu[i]) ** 2
        feature_variance /= m
        var[i] = feature_variance

    ### END CODE HERE ### 
        
    return mu, var