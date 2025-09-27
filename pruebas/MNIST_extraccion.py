from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target'].astype(int)

# Dividir dataset
n_muestras = 12000
indices = np.random.choice(X.shape[0], n_muestras, replace=False)
X, y = X[indices], y[indices]

X_entr, y_entr = X[:10000], y[:10000]
X_prueba, y_prueba = X[10000:], y[10000:]

np.savez('../datos/mnist.npz',
         X_entr = X_entr,
         y_entr = y_entr,
         X_prueba = X_prueba,
         y_prueba = y_prueba)
