from abc import ABC
from abc import abstractmethod

import warnings

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
        # if len(x.shape) > 1:
        #     x = torch.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NNFunctionApproximator(FunctionApproximator):
    def __init__(self, input_dim, output_dim, **kwargs):
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
            lr=1e-4,
            # momentum=5e-1
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.max_grad_norm = max(1e-10, kwargs.get("max_grad_norm", np.inf))

    def predict(self, X, i=0):
        # Eval mode (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
        self.model.eval()

        y = []
        with torch.no_grad():
            for i,X_i in enumerate(X):
                X_i = torch.from_numpy(X_i).to(self.device).float()
                y_pred = self.model(X_i)
                # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
                # y.append(y_pred.numpy())
                y.append(y_pred.detach().numpy())

        # TODO: Detect if we ever want multi-dimensional...
        return np.squeeze(np.array(y))

    def update(self, X, y, i=0, batch_size=32, n_epochs=10):
        try:
            X = np.array(X)
            y = np.array(y)
        except Exception:
            raise RuntimeError("Inputs X,y are not list or numpy arrays")
        if len(X) != len(y):
            raise RuntimeError("len(X) does not match len(y)")
        if batch_size < 1:
            warnings.warn("Batch size not positive, setting to 1")
            batch_size = 1
        if n_epochs < 1:
            warnings.warn("n_epochs not positive, setting to 1")
            n_epochs = 1

        batch_size = min(len(X), batch_size)

        training_set = list(zip(X,y))
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

        # TODO: Cross-validation
        # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

        self.model.train()
        for epoch_i in range(n_epochs):
            self._train_one_epoch(training_loader, epoch_i)

    def _train_one_epoch(self, training_loader, epoch_idx):
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            X_i, y_i = data

            # https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-must-have-the-same-dtype/166759
            X_i = X_i.float() # X_i = torch.from_numpy(X_i).to(self.device).float()
            y_i = y_i.float() # y_i = torch.from_numpy(y_i).to(self.device).float()

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            pred_i = self.model(X_i)
            loss = self.loss_fn(pred_i, y_i)
            loss.backward()

            if self.max_grad_norm < np.inf:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

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

        return np.squeeze(np.array(grad_X))

    def save_model(self):
        raise NotImplemented
