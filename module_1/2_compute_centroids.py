# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    for i in range(K):  # Loop over index of each cluster centroid
        cluster_sum = np.zeros((1, n))
        cluster_count = 0   
        for j in range(m): # Loop over each training example
            if idx[j] == i:
                cluster_sum = np.add(X[j], cluster_sum)
                cluster_count += 1
        cluster_sum /= cluster_count
        centroids[i] = cluster_sum
    ### END CODE HERE ## 
    
    return centroids