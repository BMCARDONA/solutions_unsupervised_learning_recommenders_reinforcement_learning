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
    diff = a - b
    squared_diff = np.square(diff)
    d = np.sum(squared_diff)
    ### END CODE HERE ###   
    
    return d