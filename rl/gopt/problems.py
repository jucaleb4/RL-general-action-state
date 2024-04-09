import numpy as np
import numpy.linalg as la

from rl.gopt import FOO

class LeastSquares(FOO):
    def __init__(self, m, n, A=None):
        super().__init__()
        rng = np.random.default_rng()
        if A is None:
            self.A = rng.uniform(size=(m,n))
        else:
            self.A = np.copy(A)
        assert self.A.shape[1] == n
        x_star = rng.uniform(size=n)
        self.b = self.A@x_star
        self.ATA = self.A.T@self.A
        self.ATb = self.A.T@self.b

    def f(self, x):
        return la.norm(self.A@x-self.b, ord=2)**2

    def df(self, x):
        return 2*(self.ATA@x-self.ATb)

class BlackBox(FOO):
    def __init__(self, f, df):
        super().__init__()
        self.f = f
        self.df = df

    def f(self, x):
        return self.f(x)

    def df(self, x):
        return self.df(x)
