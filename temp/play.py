from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
import ipdb; ipdb.set_trace()
clf = SGDClassifier(max_iter=5, tol=1e-3)
clf.fit(X_features, y)
clf.score(X_features, y)
