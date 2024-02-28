def safe_normalize_row(arr, tol=1e-16):
    """ 
    Normalizes rows (in place) so rows sum to 1 then rounds
    
    :param tol: cut off smallest value for row sum
    """
    assert tol <= 1e-8, f"tol={tol} too large (<= 1e-8)"
    assert np.min(arr) >= 0, "Given arr with negative value"
    policy_row_sum = np.atleast_2d(np.sum(arr, axis=1)).T
    rows_whose_sum_is_almost_zero = np.where(policy_row_sum < tol)
    arr[rows_whose_sum_is_almost_zero] += tol
    np.divide(arr, np.atleast_2d(np.sum(arr, axis=1)).T, out=arr)
    np.round(arr, decimals=10*int(1-np.log(tol)/np.log(10)), out=arr)

def remap_vec_to_int(obs, space):
    """ Map discrete spaces (from Discrete or Box) to a single integer """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return obs[0]
    elif isinstance(space, gym.spaces.box.Box):
        low  = space.low
        high = space.high
        diff = high-low+1
        multiplier = np.cumprod(diff)[:len(obs)-1]
        obs_as_int = (obs[0]-low[0]) + np.dot((obs-low)[1:], multiplier)
        return obs_as_int
    else:
        raise Exception("Unsupported space {type(space)} for finite spaces")
