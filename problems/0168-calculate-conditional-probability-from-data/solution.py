def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    #assuming a discrete probability distribution
    pmf = {}
    indiv_groups = {}
    n  = len(data)
    for group in data:
      pmf[group] = pmf.get(group, 0) + 1
      p, q = group
      indiv_groups[p] = indiv_groups.get(p, 0) + 1
    
    
    prob = pmf[(x, y)] / indiv_groups[x] if (x, y) in pmf else 0
    return  prob

