import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna

np.random.seed(37)

X = np.linspace(-2, 2, 100).reshape(-1, 1)
T = 2 * np.cos(X*2) +  X**3 + np.random.normal(0, 0.4, X.shape)

configuracion_red = {
    'input': 1,
    'capas_ocultas': [5, 10, 1],
    'activaciones': ['sigmoide', 'sigmoide', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 5000,
    'lr': 0.05
}

parametros_entrenados, historial = rna.entrenar_red(X, T, configuracion_red)
pred = rna.predecir(X, parametros_entrenados, configuracion_red['activaciones'])

plt.figure()
plt.plot(historial)

plt.figure()
plt.scatter(X, T, label='datos entrenamiento')
plt.plot(X, pred, 'r-',  label='prediccion', linewidth=2)
plt.legend()

plt.show()


