import numpy as np
import numpy.linalg as la

from rl import NNFunctionApproximator

# Try to learn just summation
fa = NNFunctionApproximator(3, 1)

# Learn x_1+x_2
print("Training on x_1+x_2...")
n_epochs = 1
m = 200
for _ in range(n_epochs):
    X_train = 10*np.random.random((m,3))
    y_true = np.sum(X_train[:,:2], axis=1)
    y_train = y_true + 1e-1*(np.random.random(m)-0.5)
    y_train = np.atleast_2d(y_train).T

    y_pred = fa.predict(X_train)
    print(f"Pre-train Error: {la.norm(y_pred-y_true)/min(la.norm(y_pred), la.norm(y_true)):.2e}")

    fa.update(X_train,y_train, n_epochs=10)

print("\nTesting on x_1+x_2...")
X_test = 10*np.random.random((m,3))
y_test = np.sum(X_test[:,:2], axis=1) 
y_pred = fa.predict(X_test)
print(f"Post-train Error: {la.norm(y_pred-y_test)/min(la.norm(y_pred), la.norm(y_test)):.2e}")

print("\nTesting on x_1+x_2+x_3")
X_test = 10*np.random.random((m,3))
y_test = np.sum(X_test, axis=1) 
y_pred = fa.predict(X_test)
print(f"Post-train Error: {la.norm(y_pred-y_test)/min(la.norm(y_pred), la.norm(y_test)):.2e}")

print("\nBoostrapped training to learn x_1+x_2")
# Learn x_1+x_2 by boot-strapping solution from neural network and subtracting x_3
if True:
    X_train = 10*np.random.random((m,3))
    y_train = fa.predict(X_train) + X_train[:,2] + 1e-2*(np.random.random(m)-0.5)
    y_train = np.atleast_2d(y_train).T
    fa.update(X_train,y_train, n_epochs=10)

print("Testing...")
X_test = 10*np.random.random((m,3))
y_test = np.sum(X_test, axis=1) 
y_pred = fa.predict(X_test)
print(f"Post-train Error: {la.norm(y_pred-y_test)/min(la.norm(y_pred), la.norm(y_test)):.2e}")

# grad_X = fa.grad(X_train)
# print(f"Gradient w.r.t. input X:\n{grad_X}")
