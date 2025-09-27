import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils

# Extraccion de modelo entrenado
modelo = np.load('modelos/MNIST.npz', allow_pickle=True)

parametros = modelo['params'].item()
historial = modelo['historial']
config = modelo['config'].item()

# Carga de imagenes MNIST
datos = np.load('../datos/MNIST.npz')
X = datos['X_prueba']
X = X/255.0

img_ruidosas = utils.agregar_ruido(X, 0.5)

reconstruccion = rna.predecir(X, parametros, config['activaciones'])

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validaciones')
plt.legend()
plt.show()

fig, axes = plt.subplots(5, 6)

for i, ax in enumerate(axes.reshape(-1, 3)):
    n = np.random.randint(len(X))

    # Ruidosa
    ax[0].imshow(img_ruidosas[n].reshape(28, 28), cmap="gray")
    ax[0].set_title("Entrada")
    ax[0].axis("off")

    # Original
    ax[1].imshow(X[n].reshape(28,28), cmap="gray")
    ax[1].set_title("Original")
    ax[1].axis("off")

    # Reconstruida
    ax[2].imshow(reconstruccion[n].reshape(28,28), cmap="gray")
    ax[2].set_title("Reconstruida")
    ax[2].axis("off")

plt.show()


