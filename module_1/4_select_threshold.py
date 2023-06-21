# UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        ### START CODE HERE ### 
        
        # Check if probability is less than epsilon
        predictions = (p_val < epsilon)
        prec = 0
        rec = 0
        tp = 0
        fp = 0
        fn = 0
        # true positives
        tp = np.sum((predictions == 1) & (y_val == 1))
        # false positives
        fp = np.sum((predictions == 1) & (y_val == 0))
        # false negatives
        fn = np.sum((predictions == 0) & (y_val == 1))
    
        # Precision measures the accuracy of positive predictions
        prec = tp / (tp + fp)
        # recall measures the completeness of positive predictions
        rec = tp / (tp + fn)
        # F1-score is the harmonic mean of precision and recall
        F1 = (2 * prec * rec) / (prec + rec)
        
        ### END CODE HERE ### 
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1