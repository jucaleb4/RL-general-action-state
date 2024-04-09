from abc import ABC
from abc import abstractmethod

import warnings
import random

import numpy as np
import numpy.linalg as la

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

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

def custom_SGD(solver, X, y, minibatch=32):
    n_epochs = solver.max_iter
    n_consec_regress_epochs = 0
    max_regress = solver.n_iter_no_change
    frac_validation = solver.validation_fraction
    tol = solver.tol
    early_stopping = solver.early_stopping

    train_losses = []
    test_losses = []

    for i in range(n_epochs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, shuffle=True, test_size=frac_validation)
        num_batches = int(np.ceil(len(X_train)/ minibatch))
        for j in range(num_batches):
            k_s = minibatch*j
            k_e = min(len(X_train), minibatch*(j+1))
            # mini-batch update
            solver.partial_fit(X_train[k_s:k_e], y_train[k_s:k_e])

        y_train_pred = solver.predict(X_train)
        y_test_pred = solver.predict(X_test)

        train_losses.append(la.norm(y_train_pred - y_train)**2/len(y_train))
        test_losses.append(la.norm(y_test_pred - y_test)**2/len(y_test))

        if early_stopping and len(test_losses) > 1 and test_losses[-1] > np.min(test_losses)-tol:
            n_consec_regress_epochs += 1
        else:
            n_consec_regress_epochs = 0
        if n_consec_regress_epochs == max_regress:
            print("Early stopping (stagnate)")
            break
        if train_losses[-1] <= tol:
            print("Early stopping (train loss small)")
            break

    return np.array(train_losses), np.array(test_losses)

class LinearFunctionApproximator(FunctionApproximator):
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

        # self.normalize = params.get("normalize_obs", False)
        self.alpha = params.get("sgd_alpha", 1e-4)
        self.feature_type = params.get("feature_type", "rbf")
        self._deg = params.get("deg", 1)

        # if self.normalize:
        #     self.scaler = sklearn.preprocessing.StandardScaler()
        #     self.scaler.fit(X)


        if self.feature_type == "poly":
            # TODO: Allow custom features and more customizability
            # self.featurizer = sklearn.pipeline.FeatureUnion([
                # ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                # ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            #     ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                # ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            # ])
            self.featurizer = PolynomialFeatures(self._deg)
        elif self.feature_type == "rbf":
            self.featurizer = sklearn.pipeline.FeatureUnion([
                # ("rbf1", RBFSampler(gamma=20, n_components=100)),
                # ("rbf2", RBFSampler(gamma=10, n_components=100)),
                ("rbf3", RBFSampler(gamma=5.0, n_components=100)),
                # ("rbf4", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf5", RBFSampler(gamma=1.0, n_components=100)),
                # ("rbf6", RBFSampler(gamma=0.5, n_components=100)),
                # ("rbf7", RBFSampler(gamma=0.1, n_components=100)),
                # ("rbf8", RBFSampler(gamma=0.05, n_components=100)),
            ])
        else:
            print(f"Unknown feature type {self.feature_type}")
            raise RuntimeError

        self.featurizer.fit(X)

        self.models = []
        for _ in range(num_models):
            model = SGDRegressor(
                learning_rate=params.get("sgd_stepsize", "constant"),
                max_iter=params.get("sgd_n_iter", 11),
                alpha=params.get("sgd_alpha",1e-4),
                warm_start=params.get("sgd_warmstart", False),
                tol=0.0,
                n_iter_no_change=params.get("sgd_n_iter", 1000),
            )

            # model = Lasso(
            #     alpha=0.0001,
            #     precompute=True,
            #     max_iter=params["sgd_n_iter"],
            #     positive=True, 
            #     random_state=9999, 
            #     selection='random'
            # )

            model.partial_fit(self.featurize([X[0]]), [0])
            self.models.append(model)
    
    def featurize(self, X):
        # if self.normalize:
        #     X = self.scaler.transform(X)
        return self.featurizer.transform(X)
    
    def predict(self, x, i=0):
        assert 0 <= i < len(self.models)

        features = self.featurize(x)
        # return self.models[i].predict(features)[0]
        return np.squeeze(self.models[i].predict(features))
    
    def update(self, X, y, i=0, use_custom_sgd=False):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.

        :return train_loss: over each epoch
        :return val_loss: over each epoch
        """
        assert 0 <= i < len(self.models)

        features = self.featurize(X)
        if use_custom_sgd:
            return custom_SGD(self.models[i], features, y)

        self.models[i].fit(features, y)

        pred = self.predict(X, i)
        loss = la.norm(pred-y, ord=2)**2/len(y)
        return loss, 0

    def set_coef(self, coef, i):
         assert 0 <= i < len(self.models)
         self.models[i].coef_ = np.copy(coef)

    def set_intercept(self, intercept, i):
         assert 0 <= i < len(self.models)
         self.models[i].intercept_ = np.copy(intercept)

    def get_coef(self, i):
         assert 0 <= i < len(self.models)
         return np.copy(self.models[i].coef_)

    def get_intercept(self, i):
         assert 0 <= i < len(self.models)
         return np.copy(self.models[i].intercept_)

    @property
    def dim(self):
        if self.feature_type == "poly":
            return self.featurizer.n_output_features_
        elif self.feature_type == "rbf":
            return 100*len(self.featurizer.transformer_list) # self.featurizer.n_components 
        else:
            raise RuntimeError

    def save_model(self):
        raise NotImplemented

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, params, seed=None):
        super().__init__()
        # n_hidden_layers = 2
        # layer_width = 128
        n_hidden_layers = 2
        layer_width = 16
        if params.get("network_type", "small") == "deep":
            n_hidden_layers = 8
        elif params.get("network_type", "small") == "shallow":
            layer_depth = 512

        modules = nn.ModuleList([nn.Linear(input_dim, layer_width)])
        modules.extend([nn.Tanh()])
        for _ in range(1, n_hidden_layers):
            modules.extend([nn.Linear(layer_width, layer_width)])
            modules.extend([nn.Tanh()])
        modules.extend([nn.Linear(layer_width, output_dim)])
        
        # https://discuss.pytorch.org/t/notimplementederror-module-modulelist-is-missing-the-required-forward-function/175049
        self.linears = nn.Sequential(*modules)

        # zero initialization (rather than random, which can yield non-uniform behavior)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # g_cpu = torch.Generator()
                # g_cpu.manual_seed(seed)
                # torch.nn.init.xavier_normal_(m.weight, generator=g_cpu)
                # torch.nn.init.xavier_normal_(m.weight)
                # stdv = 1. / np.sqrt(m.weight.size(1))
                # torch.nn.init.normal_(m.bias, std=stdv, generator=g_cpu)
                # torch.nn.init.xavier_uniform_(m.weight, generator=g_cpu)
                torch.nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.linears.apply(init_weights)

    def forward(self, x):
        # if len(x.shape) > 1:
        #     x = torch.flatten(x)
        logits = self.linears(x)
        return logits

class NNFunctionApproximator(FunctionApproximator):
    def __init__(self, num_models, input_dim, output_dim, params):
        super().__init__()
        assert num_models > 0

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = (
            "cuda" if torch.cuda.is_available()
            # else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.models = []
        self.optimizers = []
        self.loss_fn = nn.MSELoss()
        import time
        t_0 = int(time.time())
        for _ in range(num_models):
            model = NeuralNetwork(input_dim, output_dim, params, seed=t_0).to(self.device)
            pe_update = params.get("pe_update", "sgd")
            lr = params.get("sgd_base_stepsize", 0.001)
            weight_decay = params.get("sgd_alpha", 1e-3)

            if pe_update in ["sgd", "sgd_mom"]:
                dampening = momentum = 0 
                if params.get("pe_update", "sgd") == "sgd_mom":
                    dampening = 0 # 0.1
                    momentum = 0.9
                optimizer = torch.optim.SGD(
                    model.parameters(), 
                    momentum=momentum,
                    dampening=dampening,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=1e-8,
                )

            self.models.append(model)
            self.optimizers.append(optimizer)

        self.max_grad_norm = params.get("max_grad_norm", np.inf)
        if self.max_grad_norm <= 0:
            self.max_grad_norm = np.inf
        self.sgd_n_iter = params.get("sgd_n_iter", 11)
        self.batch_size = params.get("batch_size", 32)

    def predict(self, X, i=0):
        # Eval mode (https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval)
        self.models[i].eval()

        y = []
        with torch.no_grad():
            for j,X_j in enumerate(X):
                X_j = torch.from_numpy(X_j).to(self.device).float()
                y_pred = self.models[i](X_j)
                # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
                # y.append(y_pred.numpy())
                y.append(y_pred.detach().numpy())

        # TODO: Detect if we ever want multi-dimensional...
        return np.squeeze(np.array(y))

    def update(self, X, y, i=0, batch_size=-1, sgd_n_iter=-1, validation_frac=0.1, skip_losses=False):
        try:
            X = np.array(X)
            y = np.array(y)
        except Exception:
            raise RuntimeError("Inputs X,y are not list or numpy arrays")
        if len(X) != len(y):
            raise RuntimeError("len(X) does not match len(y)")
        if batch_size < 1:
            batch_size = self.batch_size

        # TODO: Make these customizable
        dataset = list(zip(X,y))
        val_idx= int(len(dataset)*(1.-validation_frac))

        train_losses = []
        test_losses = []
        batch_size = min(len(X), batch_size)
        sgd_n_iter = sgd_n_iter if sgd_n_iter > 0 else self.sgd_n_iter

        for _ in range(sgd_n_iter):
            # TODO: Better way to do cross validation
            random.shuffle(dataset)
            X_train, y_train = list(zip(*dataset[:val_idx]))
            train_set = list(zip(X_train, y_train))

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            self._train_one_epoch(train_loader, i=i)

            if skip_losses: 
                continue

            y_train_pred = self.predict(X_train, i)
            train_losses.append(la.norm(y_train-y_train_pred)**2/len(y_train))
            if val_idx < len(dataset):
                X_test, y_test = list(zip(*dataset[val_idx:]))
                y_test_pred = self.predict(X_test, i)
                test_losses.append(la.norm(y_test-y_test_pred)**2/len(y_test))

        return np.array(train_losses), np.array(test_losses)

    def _train_one_epoch(self, train_loader, i):
        running_loss = 0.
        last_loss = 0.

        self.models[i].train()
        for j, data in enumerate(train_loader):
            X_j, y_j = data

            # https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-must-have-the-same-dtype/166759
            X_j = X_j.float() # X_i = torch.from_numpy(X_i).to(self.device).float()
            y_j = y_j.float() # y_i = torch.from_numpy(y_i).to(self.device).float()

            pred_j = self.models[i](torch.atleast_2d(X_j))
            # TODO: Is this the right way to do it?
            if len(pred_j.shape) > 1:
                pred_j = torch.squeeze(pred_j)
            if len(pred_j.shape) == 0:
                continue
            # loss = self.loss_fn(pred_j, y_j)
            loss = (pred_j - y_j).pow(2).mean()

            # Zero your gradients for every batch!
            self.optimizers[i].zero_grad()
            loss.backward()

            if self.max_grad_norm < np.inf:
                nn.utils.clip_grad_norm_(self.models[i].parameters(), self.max_grad_norm)

            self.optimizers[i].step()

            # Gather data and report
            last_loss = loss.item()


        # TEMP: Print the weights
        # for name, param in self.models[i].named_parameters():
        #     print(f"params: {name}:\n{param}")

        return last_loss

    def grad(self, X, max_grad_norm=np.inf, i=0):
        """ 
        https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/6
        """
        grad_X = []
        for j, X_j in enumerate(X):
            X_j = torch.from_numpy(X_j).to(self.device).float()
            X_j.requires_grad = True
            # TODO: Better way to remove this?
            X_j.retain_grad()
            y_j = self.models[i](X_j)
            y_j.backward(torch.ones_like(y_j)) 
            grad_X.append(X_j.grad.numpy())

        grad_X = np.squeeze(np.array(grad_X))
        scale = 1
        if  max_grad_norm < np.inf:
            grad_norm = np.max(np.abs(grad_X))
            scale = max(1, max_grad_norm/(grad_norm+1e-10))
        return scale * grad_X

    def save_model(self):
        raise NotImplemented
