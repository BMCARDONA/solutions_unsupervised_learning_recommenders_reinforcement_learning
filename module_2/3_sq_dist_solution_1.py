# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    ### START CODE HERE ###     
    n = len(a)
    d = 0
    for i in range(n):
        d += (a[i] - b[i]) ** 2
    ### END CODE HERE ###     
    
    return d