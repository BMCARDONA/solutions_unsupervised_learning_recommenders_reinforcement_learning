# UNQ_C1
# GRADED FUNCTION: find_closest_centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]
    m = X.shape[0] # number of training examples
    k = centroids.shape[0] # number of cluster centroids
    n = X.shape[1] # number of features for training examples and cluster centroids

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    
    for i in range(m):
        curr_min = float('inf')
        curr_cluster = 0
        for j in range(k):
            curr_sum = 0
            for r in range(n):
                given_term = (X[i, r] - centroids[j, r])
                curr_sum += given_term ** 2
            curr_L2 = np.sqrt(curr_sum)
            if curr_L2 < curr_min:
                curr_cluster = j
                curr_min = curr_L2
#             curr_min = min(curr_min, curr_L2)
        idx[i] = curr_cluster
        
     ### END CODE HERE ###
    
    return idx