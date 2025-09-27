import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils

img_MNIST = np.load('../datos/MNIST.npz')
X = img_MNIST['X']
y = img_MNIST['y']

img_ceros = X[(y==0)]
img_unos = X[(y==1)]


configuracion_red = {
    'input': 784,
    'capas_ocultas': [64, 784],
    'activaciones': ['relu', 'sigmoide'],
    'costo': 'mse',
    'optimizador': 'gdm',
    'epocas': 400,
    'tamano_lote': 250,
    'lr': 0.4
}

parametros_entrenados, historial = rna.entrenar_red(img_ceros, img_ceros, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')
plt.legend()
plt.show()

np.savez('modelos/MNIST_prueba1.npz',
         params=parametros_entrenados,
         historial=historial,
         config=configuracion_red)