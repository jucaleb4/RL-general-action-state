import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

from enum import Enum

class FitMode(Enum):
    SAA = 0
    SA = 1

class FunctionApproximator():
    """
    Function approximator. 

    See: https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
    """
    def __init__(
            self, 
            num_models: int, 
            X: np.ndarray, 
            params,
        ):
        """
        :param X: initial set of states of fit the model
        """
        assert num_models > 0
        self.fit_mode = FitMode(params.get("fit_mode", 0))
        self.normalize = params.get("normalize", False)
        self.alpha = params.get("alpha", 0.1)

        if self.normalize:
            self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler.fit(X)

        # TODO: Allow custom features and more customizability
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])

        self.featurizer.fit(X)

        self.models = []
        for _ in range(num_models):
            if self.fit_mode == FitMode.SAA:
                assert self.alpha >= 0
                model = Ridge(alpha=self.alpha)
                model.fit(self.featurize([X[0]]), [0])
            else:
                model = SGDRegressor(learning_rate="constant")
                model.partial_fit(self.featurize([X[0]]), [0])

            self.models.append(model)
    
    def featurize(self, X):
        if self.normalize:
            X = self.scaler.transform(X)
        return self.featurizer.transform(X)
    
    def predict(self, x, i=0):
        assert 0 <= i < len(self.models)

        features = self.featurize(x)
        return self.models[i].predict(features)[0]
    
    def update(self, X, y, i=0):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        assert 0 <= i < len(self.models)

        if self.fit_mode == FitMode.SAA:
            # solve regularized SAA problem
            features = self.featurize(X)
            self.models[i].fit(features, y)
        else: 
            # an SGD step
            for x in X:
                features = self.featurize_state(x)
                self.models[i].partial_fit([features], [y])

    def set_coef(self, coef, i):
         assert 0 <= i < len(self.models)
         self.models[i].coef_ = coef

    def get_coef(self, i):
         assert 0 <= i < len(self.models)
         return self.models[i].coef_

    @property
    def dim(self):
        return 400
