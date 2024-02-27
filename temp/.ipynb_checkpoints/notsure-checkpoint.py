import numpy as np
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from rffridge import RFFRidgeRegression

from sklearn.gaussian_process.kernels import RBF

from utils import get_data
from utils import CustomFitter

# get data
N = 100
(X,y,X_test) = get_data(N, type="quadratic", how_X="random")

# predict use kernel ridge regression
clf    = KernelRidge(kernel=RBF())
clf    = clf.fit(X, y)
y_krr_pred = clf.predict(X_test)

# now use random Fourier features
rff_dim = 100
clf     = RFFRidgeRegression(rff_dim=rff_dim)
clf.fit(X, y)
y_rff_pred  = clf.predict(X_test)

# our custom one
clf     = CustomFitter(n=X.shape[1], dim=rff_dim, alpha=0.1)
import ipdb; ipdb.set_trace()
clf.fit(X, y)
y_cus_pred  = clf.predict(X_test)

# Set up figure and plot data.
fig, axes = plt.subplots(3, 1)
fig.set_size_inches(10, 5)
cmap      = plt.cm.get_cmap('Blues')

# plot training data
for i in range(3):
    axes[i].scatter(X, y, s=30, c=[cmap(0.3)])

# plot testing data
axes[0].plot(X_test, y_krr_pred, c=cmap(0.9))
axes[1].plot(X_test, y_rff_pred, c=cmap(0.9))
axes[2].plot(X_test, y_cus_pred, c=cmap(0.9))

names = ["RBF kernel", "RFF ridge", "Custom RFF ridge"]
for i in range(3):
    axes[i].margins(0, 0.1)
    axes[i].set_title(f'{names[i]} regression')
    axes[i].set_ylabel(r'$y$', fontsize=14)
    if i < 2:
        axes[i].set_xticks([])
    else:
        axes[i].set_xticks(np.arange(-10, 10.1, 1))
        axes[i].set_xlabel(r'$x$', fontsize=14)
        
plt.tight_layout()
