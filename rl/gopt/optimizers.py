from abc import ABC
from abc import abstractmethod

import warnings

import numpy as np
import numpy.linalg as la

# GD
# SGD
# AGD
# AC-FGM

class Optimizer(ABC):
    """ 
    Convex smooth optimization 
    """
    def __init__(self, oracle, x_init, projection, n_iter, tol, **kwargs):    
        self.oracle = oracle
        self.n_iter = n_iter
        self.tol = tol
        self._x = np.copy(x_init)
        self.projection = projection
        self._f = self.oracle.f(self._x)
        self._grad = np.copy(self.oracle.df(self._x))
        self.early_stop = False

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_x(self, x):
        self._x = np.copy(x)

    @abstractmethod
    def step(self, t=1):
        """ 
        Single step of the optimization algorithm.
        """
        raise NotImplemented

    def solve(self, n_iter=-1, verbose=0):
        """ 
        Optimizers until `n_iter` iterations or gradient tolerance is met,
        whichever is first. Returns last iterate and its gradient.
        """
        n_iter = self.n_iter if n_iter <= 0 else n_iter
        f_arr = np.zeros(n_iter, dtype=float)
        grad_arr = np.zeros(n_iter, dtype=float)
        t = 1
        while t <= n_iter:
            self.step(t)
            f_arr[t-1] = self._f
            grad_arr[t-1] = la.norm(self._grad)
            if la.norm(self._grad) <= self.tol or self.early_stop:
                # print(f"Early termination at iteration {t+1}/{n_iter}")
                f_arr = f_arr[:t]
                break
            t+=1

        return np.copy(self._x), f_arr[:t], grad_arr[:t]

class GradDescent(Optimizer):
    def __init__(self, oracle, x_init, n_iter=100, tol=1e-10, stepsize="constant", eta_0=0.001, **kwargs):    
        super().__init__(oracle, x_init, n_iter, tol, *kwargs)
        self.stepsize = stepsize
        self.eta_0 = eta_0

    def step(self, t=1):
        eta = self.eta_0
        if(not (self.stepsize == "constant")):
            eta = self.eta_0 * np.sqrt(1./t)
        self._x -= eta*self._grad
        self._f = self.oracle.f(self._x)
        self._grad = self.oracle.df(self._x)

class AccGradDescent(Optimizer):
    def __init__(self, oracle, x_init, n_iter=100, tol=1e-10, eta_0=0.001, mu=0, **kwargs):    
        super().__init__(oracle, x_init, n_iter, tol, *kwargs)
        self._xt = np.copy(self._x)
        self._uxt = np.copy(self._x)
        self._oxt = np.copy(self._x)
        self.eta_0 = eta_0
        self.mu = mu

    def step(self, t=1):
        a_t = q_t = 2./(t+1)
        # divide by /2 is important here
        eta_t = t*self.eta_0/2

        self._uxt = (1-q_t)*self._oxt + q_t*self._xt
        grad = self.oracle.df(self._uxt)
        self._xt  = 1./(1.+self.mu*eta_t) * (self._xt + self.mu*eta_t*self._uxt - eta_t*grad)
        self._oxt = (1-a_t)*self._oxt + a_t*self._xt

        self._x = self._oxt
        self._f = self.oracle.f(self._x)
        self._grad = self.oracle.df(self._x)

class ACFastGradDescent(Optimizer):
    def __init__(self, oracle, x_init, projection, alpha, n_iter=100, tol=1e-10, stop_nonconvex=False, **kwargs):    
        super().__init__(oracle, x_init, projection, n_iter, tol, **kwargs)
        assert 0<=alpha<=1.

        self._z = np.copy(self._x)
        self._y = np.copy(self._x)
        self.beta = 1.-np.sqrt(3)/2
        self.alpha = alpha
        self.L_t = ...
        self.eta_t = ...
        self.stop_nonconvex = stop_nonconvex
        self._detected_nonconvex = False
        self._first_eta = kwargs.get("first_eta", -1)

    @property
    def detected_nonconvex(self):
        return self._detected_nonconvex

    @property
    def first_eta(self):
        return self._first_eta

    def step(self, t=1):
        if t == 1:
            self._first_eta = self.eta_t = self.line_search_eta()
            self.tau_t = 0
        elif t == 2:
            self.eta_t = self.beta/(2*self.L_t)
            self.tau_prev_t = self.tau_t
            self.tau_t = 2
        else:
            min_a = (self.tau_prev_t+1)/self.tau_t * self.eta_t
            min_b = min_a if self.L_t == 0 else self.beta*self.tau_t/(4*self.L_t)
            self.eta_t = min(min_a, min_b)
            self.tau_prev_t = self.tau_t
            self.tau_t = self.tau_prev_t + self.alpha/2 
            self.tau_t += 2*(1-self.alpha)*self.eta_t * self.L_t/(self.beta * self.tau_prev_t)

        self._z = self.projection(self._y - self.eta_t*self._grad)
        self._y = (1-self.beta)*self._y + self.beta*self._z
        next_x = (self._z + self.tau_t*self._x)/(1.+self.tau_t)
        next_f = self.oracle.f(next_x)
        next_grad = self.oracle.df(next_x)

        self.L_t = self.est_L(next_x, next_f, next_grad, first_iter=(t==1))
        self._x = next_x
        self._f = next_f
        self._grad = next_grad

        self.early_stop = self.stop_nonconvex and self._detected_nonconvex
        # warnings.warn(f"Detected non-convex function (res: {linearization_diff:.4e})")

    def line_search_eta(self, first_eta=-1):
        """
        Line search for $\eta$ so that
        \begin{align*}
            \frac{\beta}{4(1-\beta)L} \leq \eta \leq \frac{1}{3L}
        \end{align*}
        """
        eta = self._first_eta if self._first_eta > 0 else 1
        tau_t = 0

        phase_I = True
        first_iter = True
        eta_lb = ...
        eta_ub = ...
        phase_I_incr = ...

        while 1:
            z = self.projection(self._y - eta*self._grad)
            y = (1-self.beta)*self._y + self.beta*z
            next_x = (z + tau_t*self._x)/(1.+tau_t)
            next_grad = self.oracle.df(next_x)

            L = self.est_L(next_x, ..., next_grad, first_iter=True)
            if self.stop_nonconvex and self._detected_nonconvex:
                return 0

            lb = self.beta/(4*(1.-self.beta)*L)
            ub = 1./(3*L)
            if lb <= eta <= ub:
                return eta
            elif phase_I:
                # check if we should start/continue doubling search
                if (first_iter or phase_I_incr) and eta < lb:
                    eta *= 2
                    phase_I_incr = True
                elif (first_iter or not phase_I_incr) and ub < eta:
                    eta /= 2
                    phase_I_incr = False
                # below prepares line search for binary search
                elif phase_I_incr and ub < eta:
                    eta_ub = eta
                    eta_lb = eta/2
                    eta = (eta_lb+eta_ub)/2
                    phase_I = False
                else:
                    eta_ub = eta*2
                    eta_lb = eta
                    eta = (eta_lb+eta_ub)/2
                    phase_I = False
                first_iter = False
            else:
                if eta < lb:
                    eta_lb = eta
                else:
                    eta_ub = eta
                eta = (eta_lb+eta_ub)/2

        return eta

    def est_L(self, next_x, next_f, next_grad, first_iter=False, tol=1e-10):
        """
        :param tol: tolerance for non-convexity/linearization
        """
        if first_iter:
            return la.norm(next_grad - self._grad)/la.norm(next_x-self._x)

        linearization_diff = self._f - next_f - np.dot(next_grad, self._x-next_x)
        if linearization_diff > 0:
            L = la.norm(self._grad - next_grad)**2/(2*linearization_diff)
        elif abs(linearization_diff) <= tol:
            L = 0
        else:
            self._detected_nonconvex = True
            L = 0

        return L
