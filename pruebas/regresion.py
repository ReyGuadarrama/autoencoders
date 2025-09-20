import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna

np.random.seed(37)

X = np.linspace(-1, 1, 100).reshape(-1, 1)
T = 2 * np.cos(X*2) + X**3 + np.random.normal(0, 0.4, X.shape)


configuracion_red = {
    'input': 1,
    'capas_ocultas': [5, 10, 1],
    'activaciones': ['sigmoide', 'sigmoide', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 1000,
    'tamano_lote': 100,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(X, T, configuracion_red)
X_test = np.linspace(-1, 1, 102).reshape(-1, 1)
pred = rna.predecir(X_test, parametros_entrenados, configuracion_red['activaciones'])

plt.figure()
plt.plot(historial)

plt.figure()
plt.scatter(X, T)
plt.plot(X_test, pred, 'r-')
plt.show()
