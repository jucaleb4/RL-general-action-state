from abc import ABC
from abc import abstractmethod

import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge

import torch
from torch import nn
from torchvision.transforms import ToTensor

from enum import Enum

class FunctionApproximator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x, i=0):
        raise NotImplemented
    
    @abstractmethod
    def update(self, X, y, i=0):
        raise NotImplemented

    @abstractmethod
    def save_model(self):
        raise NotImplemented

class FitMode(Enum):
    SAA = 0
    SA = 1

class LinearFunctionApproximator():
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
        super().__init__()
        assert num_models > 0

        self.fit_mode = 1 # FitMode(params.get("fit_mode", 0))
        self.normalize = params["normalize_obs"]
        self.alpha = params["sgd_alpha"]

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
                model = SGDRegressor(learning_rate=params["sgd_stepsize"], max_iter=params["sgd_n_iter"])
                model.partial_fit(self.featurize([X[0]]), [0])

            self.models.append(model)
    
    def featurize(self, X):
        if self.normalize:
            print("here")
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

        features = self.featurize(X)
        self.models[i].fit(features, y)
        # for x_i,y_i in zip(X,y):
        #     features = self.featurize(np.atleast_2d(x_i))
        #     self.models[i].partial_fit(features, [y_i])
        # if self.fit_mode == FitMode.SAA:
        #     # solve regularized SAA problem
        #     features = self.featurize(X)
        #     self.models[i].fit(features, y)
        # else: 
        #     # an SGD step
        #     for x_i,y_i in zip(X,y):
        #         features = self.featurize(np.atleast_2d(x_i))
        #         self.models[i].partial_fit(features, [y_i])

    def set_coef(self, coef, i):
         assert 0 <= i < len(self.models)
         self.models[i].coef_ = coef

    def set_intercept(self, intercept, i):
         assert 0 <= i < len(self.models)
         self.models[i].intercept_ = intercept

    def get_coef(self, i):
         assert 0 <= i < len(self.models)
         return self.models[i].coef_

    def get_intercept(self, i):
         assert 0 <= i < len(self.models)
         return self.models[i].intercept_

    @property
    def dim(self):
        # TODO: Make this customizable
        return 400

    def save_model(self):
        raise NotImplemented

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        if len(x.shape) > 1:
            x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NNFunctionApproximator(FunctionApproximator):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=1e-3,
            # momentum=5e-1
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def predict(self, X, i=0):
        # Eval mode (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
        self.model.eval()

        y = []
        with torch.no_grad():
            for i,X_i in enumerate(X):
                X_i = torch.from_numpy(X_i).to(self.device).float()
                y_pred = self.model(X_i)
                y.append(y_pred.numpy())

        # TODO: Detect if we ever want multi-dimensional...
        return np.squeeze(np.array(y))

    def update(self, X, y, i=0):
        # Training mode (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train)
        # TODO: Allow shuffling and mini-batch and tuning
        self.model.train()

        # split into batches...
        for batch, (X_i, y_i) in enumerate(zip(X,y)):
            X_i = torch.from_numpy(X_i).to(self.device).float()
            y_i = torch.from_numpy(y_i).to(self.device).float()

            # Compute prediction error
            pred = self.model(X_i)
            loss = self.loss_fn(pred, y_i)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                # TODO: Save loss...

    def grad(self, X):
        """ 
        https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/6
        """
        grad_X = []
        for i, X_i in enumerate(X):
            X_i = torch.from_numpy(X_i).to(self.device).float()
            X_i.requires_grad = True
            # TODO: Better way to remove this?
            X_i.retain_grad()
            y_i = self.model(X_i)
            y_i.backward(torch.ones_like(y_i)) 
            grad_X.append(X_i.grad.numpy())

        return np.array(grad_X)

    def save_model(self):
        raise NotImplemented
