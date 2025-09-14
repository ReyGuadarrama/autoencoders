import numpy as np

def lineal(z):
    return z

def derivada_lineal(z):
    return np.ones(z.shape)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def derivada_sigmoide(z):
    g = sigmoide(z)
    return g * (1 - g)

mapa_activaciones = {
    'lineal': (lineal, derivada_lineal),
    'sigmoide': (sigmoide, derivada_sigmoide),
}