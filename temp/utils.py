import numpy as np
import numpy.linalg as la
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge

def get_data(N, type, how_X=""):
    if how_X == "random":
        X = 20*np.random.random(N)[:,None]-10
    else:
        X = np.linspace(-10, 10, N)[:, None]
        
    if type == "step":
        return _get_step_data(X)
    elif type == "periodic":
        return _get_periodic_data(X)
    elif type == "quadratic":
        return _get_quadratic_data(X)
    else:
        return _get_random_data(X)

def _get_random_data(X):
    N = len(X)
    mean  = np.zeros(N)
    cov   = RBF()(X.reshape(N, -1))
    y     = np.random.multivariate_normal(mean, cov)
    noise = np.random.normal(0, 0.5, N)
    y    += noise
    
    # Finer resolution for smoother curve visualization.
    X_test = np.linspace(-10, 10, N*2)[:, None]

    return (X,y,X_test)

def _get_step_data(X):
    N = len(X)
    y     = (np.abs(X) <= 5).astype("int")
    
    # Finer resolution for smoother curve visualization.
    X_test = np.linspace(-10, 10, N*2)[:, None]

    return (X,y,X_test)

def _get_periodic_data(X):
    N = len(X)
    y     = np.sin(X)
    
    # Finer resolution for smoother curve visualization.
    X_test = np.linspace(-10, 10, N*2)[:, None]

    return (X,y,X_test)

def _get_quadratic_data(X):
    N = len(X)
    y = np.square(X)
    
    # Finer resolution for smoother curve visualization.
    X_test = np.linspace(-10, 10, N*2)[:, None]

    return (X,y,X_test)

def get_rbf_affine(n, dim, rng):
    """ Get affine transformation for rbf """
    C = 2**0.5 * rng.normal(size=(dim, n))
    s = 2*np.pi*rng.random(size=dim)
    return (C,s)

class CustomFitter:
    def __init__(self, n, dim, alpha=1., seed=None):
        self.n = n
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        (C,s) = get_rbf_affine(n, dim, self.rng)
        self.C = C
        self.s = s
        self.model = Ridge(alpha=alpha)
        self.has_fitted = False

    def fit(self, X,y):
        assert len(X) == len(y), f"len(X)={len(X)} and len(y)={len(y)} do not match"
        assert X.shape[1] == self.n, f"data set dim {X.shape[1]} does not match feature input dim n={self.n}"

        print("Fitting using CustomFitter")
        A = np.zeros((X.shape[0], self.dim))
        for i,x in enumerate(X): 
            A[i] = (2./self.dim)**0.5 * np.cos(self.C@x+self.s)
        self.model.fit(A, y)
        self.has_fitted = True

    def predict(self, X):
        if not self.has_fitted:
            print("Run fit before predict... returning zero array")
            return np.zeros(len(X))
            
        print("Predicting using CustomFitter")
        A = np.zeros((X.shape[0], self.dim))
        for i,x in enumerate(X): 
            A[i] = (2./self.dim)**0.5 * np.cos(self.C@x+self.s)

        return self.model.predict(A)