def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    event_count = {}
    for s in samples:
        event_count[s] = event_count.get(s, 0) + 1
    
    num_events = len(samples)
    pmf = []
    for key, value in event_count.items():
        pmf.append((key, value / num_events))
    return  pmf