import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils

np.random.seed(37)

img_MNIST = np.load('../datos/mnist.npz')

X = img_MNIST['X_entr']
X = X / 255.0
X_ruidosas = utils.agregar_ruido(X, 0.5)

configuracion_red = {
    'input': 784,
    'capas_ocultas': [64, 784],
    'activaciones': ['relu', 'sigmoide'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 200,
    'tamano_lote': 250,
    'lr': 0.4
}

parametros_entrenados, historial = rna.entrenar_red(X_ruidosas, X, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validaciones')
plt.legend()
plt.show()

np.savez('modelos/MNIST.npz',
         params=parametros_entrenados,
         historial=historial,
         config=configuracion_red)




